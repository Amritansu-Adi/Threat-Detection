#!/usr/bin/env python3
"""Main entry point for Threat Detection System.

Usage:
    python main.py                    # Start with default config.yaml
    python main.py --config custom.yaml  # Start with custom config
    python main.py --help             # Show help

The system will:
1. Load configuration
2. Initialize all components
3. Start processing threads
4. Run until interrupted (Ctrl+C)
5. Gracefully shutdown

Signal Handling:
- SIGINT (Ctrl+C): Graceful shutdown
- SIGTERM: Graceful shutdown (Unix only)
"""

import argparse
import logging
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core.system_manager import SystemManager


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("threat_detection.log")
        ]
    )

    # Set specific log levels
    logging.getLogger("ultralytics").setLevel(logging.WARNING)  # Reduce YOLO noise
    logging.getLogger("transformers").setLevel(logging.WARNING)  # Reduce HF noise


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Threat Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --config production.yaml
  python main.py --verbose
        """
    )

    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to configuration YAML file (default: config.yaml)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging (DEBUG level)"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    setup_logging()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting Threat Detection System")
    logger.info(f"Config file: {args.config}")

    # Validate config file exists
    if not Path(args.config).exists():
        logger.error(f"Config file not found: {args.config}")
        logger.info("Creating default config.yaml...")
        # Config will be created by ConfigLoader if it doesn't exist
        pass

    try:
        # Initialize system manager
        logger.info("Creating SystemManager...")
        system = SystemManager(config_path=args.config)
        logger.info("SystemManager created, calling system.start()...")

        # Start system
        system.start()
        logger.info("System.start() completed")

        # Log system status
        logger.info("Getting system status...")
        status = system.get_status()
        logger.info("System components status:")
        for component, comp_status in status.items():
            if component not in ["running", "uptime_sec", "config", "shared_state"]:
                logger.info(f"  {component}: {comp_status}")

        # Wait for shutdown signal
        logger.info("✅ System running. Press Ctrl+C to stop.")
        system.wait_for_shutdown()

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"System error: {e}")
        sys.exit(1)
    finally:
        # Ensure clean shutdown
        if 'system' in locals():
            system.stop()

    logger.info("👋 Threat Detection System stopped")


if __name__ == "__main__":
    main()