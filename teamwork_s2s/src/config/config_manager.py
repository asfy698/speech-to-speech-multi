import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages real-time parameter adjustments, configuration lifecycle, and persistence."""

    def __init__(self, config_path: str = "", initial_config: Optional[Dict[str, Any]] = None) -> None:
        """Initializes ConfigManager with path to a config file."""
        self.config_path = config_path
        self.config: Dict[str, Any] = {
            "vad_threshold": 0.01,
            "stt_model": "mock",
            "llm_model": "mock",
            "tts_voice": "default",
            "fer_enabled": True,
            "alert_phone": "+1234567890",
            "twilio_account_sid": "",
            "twilio_auth_token": "",
            "twilio_from_number": "whatsapp:+14155238886",
        }

        # Override with initial_config if provided (for backward compatibility)
        if initial_config:
            self.config.update(initial_config)

        # Load from file if path is specified
        if self.config_path:
            self.reload_config()

    def reload_config(self) -> bool:
        """Reloads configuration values from the config file path on disk."""
        if not self.config_path or not os.path.exists(self.config_path):
            logger.warning(f"Config file path '{self.config_path}' does not exist. Using defaults.")
            return False
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    self.config.update(data)
                    logger.info(f"Configuration loaded from {self.config_path}")
                    return True
                else:
                    logger.error(f"Invalid JSON content in '{self.config_path}'. Expected dictionary.")
                    return False
        except Exception as e:
            logger.error(f"Error loading configuration from '{self.config_path}': {e}")
            return False

    def update_config(self, config_data: Dict[str, Any]) -> bool:
        """Updates configuration values, persists them to disk, and handles rollbacks on failure."""
        backup_config = self.config.copy()
        try:
            # Perform basic validation: check types
            if "vad_threshold" in config_data:
                if not isinstance(config_data["vad_threshold"], (int, float)):
                    raise TypeError("vad_threshold must be a float or int")
            if "fer_enabled" in config_data:
                if not isinstance(config_data["fer_enabled"], bool):
                    raise TypeError("fer_enabled must be a boolean")

            # Apply changes
            self.config.update(config_data)

            # Persist to disk if path is set
            if self.config_path:
                with open(self.config_path, "w", encoding="utf-8") as f:
                    json.dump(self.config, f, indent=4)
                logger.info(f"Configuration updated and saved to {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to update/persist configuration: {e}. Rolling back.")
            self.config = backup_config
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieves a configuration value by key, returning a default if not found."""
        return self.config.get(key, default)
