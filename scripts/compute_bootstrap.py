"""Regenerate the complete validated CPU artifact set for one explicit run."""
import argparse
from cxrquant.replay import replay
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    replay(args.run_dir)
