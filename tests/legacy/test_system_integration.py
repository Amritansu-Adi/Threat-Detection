"""Integration test for SystemManager - Phase 5 integration.

Tests the SystemManager's ability to:
1. Initialize all components
2. Start/stop threads gracefully
3. Handle configuration loading
4. Provide system status
"""

import time
import logging
from unittest.mock import patch, MagicMock

from core.system_manager import SystemManager
from core.config_loader import ConfigLoader

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def test_system_manager_initialization():
    """Test SystemManager initialization with config loading."""
    logger.info("Testing SystemManager initialization...")

    # Test with default config
    system = SystemManager()

    # Verify config was loaded
    assert "system" in system.config
    assert "visual" in system.config
    assert "audio" in system.config
    assert "fusion" in system.config

    # Verify components are not initialized yet
    assert system.visual_pipeline is None
    assert system.audio_pipeline is None
    assert system.fusion_manager is None

    # Verify system is not running
    assert not system.running

    logger.info("✅ SystemManager initialization test passed")


def test_config_loader():
    """Test ConfigLoader functionality."""
    logger.info("Testing ConfigLoader...")

    loader = ConfigLoader("config.yaml")
    config = loader.load()

    # Verify required sections exist
    assert "system" in config
    assert "visual" in config
    assert "audio" in config
    assert "fusion" in config

    # Verify some key values
    assert config["system"]["target_fps"] == 30.0
    assert config["visual"]["person_conf"] == 0.60
    assert config["audio"]["sample_rate"] == 16000

    logger.info("✅ ConfigLoader test passed")


def test_system_status():
    """Test system status reporting."""
    logger.info("Testing system status...")

    system = SystemManager()
    status = system.get_status()

    # Verify status structure
    assert "running" in status
    assert "uptime_sec" in status
    assert "config" in status
    assert "shared_state" in status

    # Verify initial state
    assert not status["running"]
    assert status["uptime_sec"] == 0

    logger.info("✅ System status test passed")


def test_system_start_stop_mock():
    """Test system start/stop with mocked components (no actual threads)."""
    logger.info("Testing system start/stop with mocks...")

    system = SystemManager()

    # Mock the pipeline components to avoid actual initialization
    with patch.object(system, '_init_pipelines') as mock_init, \
         patch.object(system, '_start_threads') as mock_start, \
         patch.object(system, '_stop_threads') as mock_stop, \
         patch.object(system, '_setup_signal_handlers') as mock_signals:

        # Test start
        system.start()
        assert system.running
        mock_init.assert_called_once()
        mock_start.assert_called_once()
        mock_signals.assert_called_once()

        # Test stop
        system.stop()
        assert not system.running
        mock_stop.assert_called_once()

    logger.info("✅ System start/stop mock test passed")


if __name__ == "__main__":
    # Run tests
    test_system_manager_initialization()
    test_config_loader()
    test_system_status()
    test_system_start_stop_mock()

    print("\n🎉 All SystemManager integration tests passed!")