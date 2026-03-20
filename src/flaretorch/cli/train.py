import argparse
import sys
from loguru import logger
from flaretorch.tasks.training import train


def main():
    """
    Main entry point for training a FlareTorch model.
    Usage: flare-train --config <config_name>
    """
    parser = argparse.ArgumentParser(description="FlareTorch Training CLI")
    parser.add_argument(
        "--config", type=str, help="Name of the Hydra config file (without .yaml)"
    )

    # Parse known args to allow Hydra to handle the rest
    args, unknown = parser.parse_known_args()

    if args.config:
        logger.info(f"Starting training with config: {args.config}")
        # Hydra expects --config-name=<name>
        # We modify sys.argv so that hydra.main in train() picks it up
        sys.argv = [sys.argv[0]] + unknown + [f"--config-name={args.config}"]
    else:
        logger.info("Starting training with default config")

    try:
        train()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
