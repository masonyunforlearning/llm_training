import argparse

from training.cli import add_common_arguments, run_training


def main():
    parser = argparse.ArgumentParser(
        description="Token-budget final bilingual pretraining"
    )
    parser.add_argument("--model", default="1.2B", choices=["1.2B"])
    add_common_arguments(parser)
    parser.add_argument(
        "--output-dir",
        default="experiments/final_1.2B",
    )
    args = parser.parse_args()

    run_training(args, args.model)


if __name__ == "__main__":
    main()
