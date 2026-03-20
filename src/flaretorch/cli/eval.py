import argparse
from loguru import logger


def main():
    """
    Main entry point for evaluating a FlareTorch model.
    Usage: flare-eval --model_path <path_to_checkpoint>
    """
    parser = argparse.ArgumentParser(description="FlareTorch Evaluation CLI")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model checkpoint"
    )

    args = parser.parse_args()

    logger.info(f"Starting evaluation for model: {args.model_path}")

    # Stub for evaluation logic
    logger.warning("Evaluation logic is not fully implemented in this entry point yet.")
    logger.info(f"Model path provided: {args.model_path}")

    # In a real scenario, we might call something from flaretorch.tasks.calibration
    # or a dedicated evaluation module.

    # TODO: Implement actual evaluation call


if __name__ == "__main__":
    main()
