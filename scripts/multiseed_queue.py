#!/usr/bin/env python3
"""Resumable queue for the additional-training-seed full runs.

Works on multi-GPU servers with MIG partitions and on ordinary CUDA GPUs (one or more devices).
The queue never shares an accelerator with another workload: it waits until
every visible MIG instance or GPU reports at most ``--max-used-mib`` of used memory
for two consecutive polls, so that training fits and the recorded latency and
memory are not contaminated by a co-tenant. Seeds run strictly one after
another; a seed that does not finish with ``status == complete`` and worker
exit codes ``[0, 0]`` stops the queue. Completed seeds are skipped on restart
and an interrupted seed is resumed with ``--resume`` (identity-guarded).

After the last seed the CPU-only extension analyses are run on each new run and
the between-seed summary is aggregated with the archived seed-42 run.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from cxrquant.paths import REPO_ROOT
from cxrquant.runio import atomic_json

COMPAT = "/usr/local/cuda-12.0/compat"
MIG_ROW = re.compile(r"^\|\s+(\d+)\s+\d+\s+\d+\s+\d+\s+\|\s+(\d+)MiB\s*/\s*(\d+)MiB", re.M)


def mig_memory():
    """Used/total memory per MIG instance, or per full GPU when MIG is not enabled."""
    out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True).stdout
    rows = [{"gpu": int(g), "used_mib": int(u), "total_mib": int(t)} for g, u, t in MIG_ROW.findall(out)]
    if rows:
        return rows
    query = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, check=True).stdout
    return [{"gpu": int(i), "used_mib": int(u), "total_mib": int(t)}
            for i, u, t in (line.split(",") for line in query.strip().splitlines())]


def devices():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("NVIDIA_VISIBLE_DEVICES", "")
    values = [x.strip() for x in visible.split(",") if x.strip() and x.strip() != "all"]
    return values or [str(row["gpu"]) for row in mig_memory()]


class Queue:
    def __init__(self, args):
        self.args = args
        self.state_path = args.state.resolve()
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self.state.setdefault("seeds", {})
        self.state.update(pid=os.getpid(), seeds_requested=args.seeds, tag=args.tag)

    def save(self, **kw):
        self.state.update(kw, updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        atomic_json(self.state_path, self.state)

    def log(self, msg):
        print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)

    def wait_for_idle_gpus(self, label):
        consecutive = 0
        while True:
            mem = mig_memory()
            free_disk = shutil.disk_usage(REPO_ROOT).free // 2**30
            idle = len(mem) >= self.args.min_devices and all(m["used_mib"] <= self.args.max_used_mib for m in mem)
            enough_disk = free_disk >= self.args.min_free_disk_gib
            consecutive = consecutive + 1 if idle and enough_disk else 0
            self.save(status="waiting_for_idle_gpus", waiting_for=label, mig_memory=mem,
                      free_disk_gib=free_disk)
            if consecutive >= 2:
                return
            if idle and not enough_disk:
                self.log(f"GPUs idle but only {free_disk} GiB free disk; waiting")
            time.sleep(self.args.poll)

    def env(self, device=None):
        env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"))
        if Path(COMPAT).is_dir() and os.environ.get("CXRQUANT_USE_CUDA_COMPAT", "1") == "1":
            # Container workaround for a broken libcuda (see RUN_MULTI_GPU.md); ignored where the directory does not exist.
            env["LD_LIBRARY_PATH"] = COMPAT
        env.pop("CUDA_VISIBLE_DEVICES", None)
        if self.args.devices:
            # Launcher order decides which MIG receives which round-robin model group.
            env["CUDA_VISIBLE_DEVICES"] = self.args.devices
        if device:
            env["CUDA_VISIBLE_DEVICES"] = device
        return env

    def run(self, cmd, env, log_name):
        log_path = self.state_path.parent / "logs" / log_name
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log("RUN " + " ".join(cmd))
        with log_path.open("a") as log:
            return subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode

    def preflight(self):
        if self.state.get("preflight") == "complete":
            return
        run_dir = REPO_ROOT / "runs" / f"preflight_multiseed_{self.args.tag}"
        cfg = REPO_ROOT / "configs" / f"full_seed{self.args.seeds[0]}.json"
        code = self.run(
            [sys.executable, "-m", "cxrquant.pipeline", "--mode", "preflight", "--run-dir",
             str(run_dir), "--config", str(cfg)],
            self.env(devices()[0]), "preflight.log",
        )
        if code != 0:
            self.save(status="stopped", error=f"preflight exit {code}; see {run_dir}/preflight.json")
            raise SystemExit(code)
        self.save(preflight="complete")

    def full_run(self, seed):
        run_dir = REPO_ROOT / "runs" / f"full_seed{seed}_{self.args.tag}"
        status_path = run_dir / "status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text())
            if status.get("status") == "complete" and status.get("worker_exit_codes") == [0, 0]:
                return run_dir
        cmd = [sys.executable, "-m", "cxrquant.pipeline", "--mode", "full", "--run-dir", str(run_dir),
               "--config", str(REPO_ROOT / "configs" / f"full_seed{seed}.json"),
               "--data", str(REPO_ROOT / "data" / "iuxray_paired.json")]
        if (run_dir / "identity.json").exists():
            cmd.append("--resume")
        started = time.time()
        self.state["seeds"][str(seed)] = {"status": "running", "run_dir": str(run_dir), "started_unix": started}
        self.save(status="running", active_seed=seed)
        code = self.run(cmd, self.env(), f"full_seed{seed}.log")
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        ok = code == 0 and status.get("status") == "complete" and status.get("worker_exit_codes") == [0, 0]
        self.state["seeds"][str(seed)].update(
            status="complete" if ok else "failed", exit_code=code, duration_s=time.time() - started,
            matrix=status.get("matrix"), error=status.get("error"),
        )
        self.save(active_seed=None)
        if not ok:
            self.save(status="stopped", error=f"seed {seed} did not complete; queue halted")
            raise SystemExit(1)
        return run_dir

    def analyses(self, run_dirs):
        for run_dir in run_dirs:
            for script in ("efficiency_summary", "ablation_uncertain_policy", "ablation_stratum"):
                code = self.run([sys.executable, str(REPO_ROOT / "scripts" / (script + ".py")),
                                 "--run-dir", str(run_dir), "--overwrite"], self.env(), "analyses.log")
                if code != 0:
                    self.save(status="stopped", error=f"{script} failed on {run_dir.name}")
                    raise SystemExit(code)
        seeds = [42] + self.args.seeds
        out = REPO_ROOT / "runs" / ("multiseed_seed" + "-".join(map(str, seeds)) + "_" + self.args.tag)
        if not (out / "multiseed_summary.json").exists():
            all_runs = [self.args.seed42_run.resolve()] + list(run_dirs)
            code = self.run([sys.executable, str(REPO_ROOT / "scripts" / "aggregate_seeds.py"),
                             "--run-dirs", ",".join(map(str, all_runs)), "--out-dir", str(out)],
                            self.env(), "analyses.log")
            if code != 0:
                self.save(status="stopped", error="aggregate_seeds failed")
                raise SystemExit(code)
        self.save(aggregate=str(out))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="123,777,2024,31415")
    ap.add_argument("--tag", required=True, help="fixed run-directory suffix, reused on resume")
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--seed42-run", type=Path, default=REPO_ROOT / "runs" / "full_seed42_corrected_20260911")
    ap.add_argument("--max-used-mib", type=int, default=1024)
    ap.add_argument("--min-devices", type=int, default=1, help="devices that must be idle (2 on a two-MIG server)")
    ap.add_argument("--devices", help="ordered MIG UUIDs for the launcher (first receives distilgpt2/gpt2-medium/pythia-1b)")
    ap.add_argument("--min-free-disk-gib", type=int, default=150)
    ap.add_argument("--poll", type=int, default=120)
    args = ap.parse_args(argv)
    args.seeds = [int(s) for s in args.seeds.split(",")]
    if args.devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    if len(devices()) < args.min_devices:
        raise SystemExit(f"at least {args.min_devices} CUDA device(s) are required")
    queue = Queue(args)
    queue.save(status="starting", started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               launcher_devices=devices(), max_used_mib=args.max_used_mib)
    run_dirs = []
    for i, seed in enumerate(args.seeds):
        queue.wait_for_idle_gpus(f"seed {seed}")
        if i == 0:
            queue.preflight()
        run_dirs.append(queue.full_run(seed))
    queue.analyses(run_dirs)
    queue.save(status="complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
