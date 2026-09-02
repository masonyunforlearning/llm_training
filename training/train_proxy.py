import argparse

from training.cli import add_common_arguments, run_training

def main():
    parser = argparse.ArgumentParser(
        description="Token-budget bilingual proxy pretraining"
    )
    parser.add_argument("--model", default="50M", choices=["50M", "100M"])
    add_common_arguments(parser)
    parser.add_argument(
        "--output-dir",
        default="experiments/proxy/run",
    )
    args = parser.parse_args()

    run_training(args, args.model)


if __name__ == "__main__":
    main()
