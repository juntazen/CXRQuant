# Checkpoint Weights Excluded

The local execution workspace retains six final checkpoints. Weight files are intentionally excluded from release ZIPs. Model revisions, configuration, checkpoint hashes, measured FP32 weight sizes and training metadata are under models/<model>/training.json. Recreate weights with configs/full.json and the command in RUN_MULTI_GPU.md.
