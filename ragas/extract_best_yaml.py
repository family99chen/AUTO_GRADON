from autorag.deploy import extract_best_config
from pathlib import Path
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--trial_dir',
        type=str,
        help="Output dir for storing generated single-experiment config files",
        default="/home/lyb/RAG/data/eli5_data"
    )
    parser.add_argument(
        '--output_file',
        type=str,
        help="Output dir for storing generated logs",
        default="../experiments/eli5_runner/best.yaml"
    )

    return parser.parse_args()

def main():
    args = parse_args()
    extract_best_config(trial_path=args.trial_dir, output_path=args.output_file)


if __name__ == '__main__':
    main()