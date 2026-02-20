"""Configuration loader for YAML-based system configuration.

Supports:
- YAML file loading
- Environment variable overrides
- Configuration validation
- Default value fallbacks

Configuration Structure:
```yaml
system:
  target_fps: 30.0
  fusion_hz: 1.0

visual:
  person_model: yolov8n.pt
  weapon_model: models/weapon_best.pt
  person_conf: 0.60
  person_min_area: 3500
  weapon_detect_every_n: 3
  face_recognize_every_n: 10

audio:
  sample_rate: 16000
  device: cpu

fusion:
  risk_decay_alpha: 0.95
  confidence_beta: 1.0
```
"""

import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Loads and validates system configuration from YAML files.

    Supports environment variable overrides and default values.
    """

    # Default configuration
    DEFAULT_CONFIG = {
        "system": {
            "target_fps": 30.0,
            "fusion_hz": 1.0,
        },
        "visual": {
            "person_model": "yolov8n.pt",
            "weapon_model": "models/weapon_best.pt",
            "person_conf": 0.60,
            "person_min_area": 3500,
            "weapon_detect_every_n": 3,
            "face_recognize_every_n": 10,
        },
        "audio": {
            "sample_rate": 16000,
            "device": "cpu",
        },
        "fusion": {
            "risk_decay_alpha": 0.95,
            "confidence_beta": 1.0,
        }
    }

    def __init__(self, config_path: Optional[str] = None):
        """Initialize ConfigLoader.

        Args:
            config_path: Path to YAML config file (optional)
        """
        self.config_path = config_path or "config.yaml"
        self.config = {}

    def load(self) -> Dict[str, Any]:
        """Load configuration from file and apply overrides.

        Returns:
            Configuration dictionary
        """
        # Start with defaults
        self.config = self.DEFAULT_CONFIG.copy()

        # Load from file if exists
        if Path(self.config_path).exists():
            file_config = self._load_from_file()
            self._merge_config(self.config, file_config)

        # Apply environment overrides
        self._apply_env_overrides()

        # Validate configuration
        self._validate_config()

        logger.info(f"Configuration loaded from {self.config_path}")
        logger.debug(f"Config: {self.config}")

        return self.config

    def _load_from_file(self) -> Dict[str, Any]:
        """Load configuration from YAML file.

        Returns:
            Configuration dictionary from file
        """
        if yaml is None:
            logger.warning("PyYAML not installed, using default config")
            return {}

        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                if config is None:
                    config = {}
                return config
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            return {}

    def _merge_config(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """Recursively merge override config into base config.

        Args:
            base: Base configuration (modified in place)
            override: Override configuration
        """
        for key, value in override.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides.

        Environment variables should be prefixed with THREAT_ and use double underscores
        for nested keys. Examples:
        - THREAT_SYSTEM__TARGET_FPS=25.0
        - THREAT_VISUAL__PERSON_CONF=0.7
        """
        prefix = "THREAT_"

        for env_key, env_value in os.environ.items():
            if not env_key.startswith(prefix):
                continue

            # Remove prefix and split by double underscore
            config_key = env_key[len(prefix):].lower().split("__")

            try:
                # Navigate to the config location
                config_section = self.config
                for key_part in config_key[:-1]:
                    if key_part not in config_section:
                        config_section[key_part] = {}
                    config_section = config_section[key_part]

                # Set the value (try to convert to appropriate type)
                final_key = config_key[-1]
                config_section[final_key] = self._convert_env_value(env_value)

                logger.info(f"Applied env override: {env_key} = {env_value}")

            except Exception as e:
                logger.warning(f"Failed to apply env override {env_key}: {e}")

    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type.

        Args:
            value: String value from environment

        Returns:
            Converted value (int, float, bool, or str)
        """
        # Try boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'

        # Try int
        try:
            return int(value)
        except ValueError:
            pass

        # Try float
        try:
            return float(value)
        except ValueError:
            pass

        # Return as string
        return value

    def _validate_config(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If configuration is invalid
        """
        # System validation
        system = self.config.get("system", {})
        if system.get("target_fps", 0) <= 0:
            raise ValueError("system.target_fps must be > 0")
        if system.get("fusion_hz", 0) <= 0:
            raise ValueError("system.fusion_hz must be > 0")

        # Visual validation
        visual = self.config.get("visual", {})
        if not (0 < visual.get("person_conf", 0) <= 1):
            raise ValueError("visual.person_conf must be between 0 and 1")
        if visual.get("person_min_area", 0) <= 0:
            raise ValueError("visual.person_min_area must be > 0")

        # Audio validation
        audio = self.config.get("audio", {})
        if audio.get("sample_rate", 0) <= 0:
            raise ValueError("audio.sample_rate must be > 0")

        # Fusion validation
        fusion = self.config.get("fusion", {})
        if not (0 < fusion.get("risk_decay_alpha", 0) <= 1):
            raise ValueError("fusion.risk_decay_alpha must be between 0 and 1")

        logger.info("Configuration validation passed")