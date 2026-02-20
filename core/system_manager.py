"""System Manager - Coordinates all threads and system lifecycle.

SystemManager Architecture:
  - Thread coordination (startall/stopall)
  - Lifecycle management (startup/shutdown)
  - Error recovery and monitoring
  - Configuration loading
  - Signal handling

Thread Hierarchy:
  SystemManager (main thread)
    ├── VisualPipeline (background thread)
    ├── AudioPipeline (background thread)
    └── FusionManager (background thread)

Data Flow:
  VisualPipeline → visual_to_fusion_q → FusionManager
  AudioPipeline → audio_to_fusion_q → FusionManager
  FusionManager → fusion_to_visual_q → VisualPipeline

Configuration:
  - Loads from config.yaml
  - Environment variable overrides
  - Validation and logging
"""

import logging
import threading
import time
import signal
import sys
from typing import Optional, Dict, Any
from pathlib import Path

from core.config_loader import ConfigLoader
from core.queue_manager import QueueManager
from core.shared_state import SharedStateManager
from audio.audio_pipeline import AudioPipeline
from visual.visual_pipeline import VisualPipeline
from fusion.fusion_manager import FusionManager
from camera_capture import CaptureManager
from display_manager import DisplayManager

logger = logging.getLogger(__name__)


class SystemManager:
    """Manages all threads and system lifecycle for the threat detection system.

    Coordinates VisualPipeline, AudioPipeline, and FusionManager threads.
    Handles graceful startup/shutdown, error recovery, and configuration.

    Attributes:
        config: System configuration dictionary
        queue_manager: QueueManager for inter-thread communication
        shared_state: SharedStateManager for global state
        visual_pipeline: VisualPipeline instance
        audio_pipeline: AudioPipeline instance
        fusion_manager: FusionManager instance
        running: System running state
    """

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize SystemManager.

        Args:
            config_path: Path to configuration YAML file
        """
        self.config_path = config_path
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.load()

        # Core components
        self.queue_manager = QueueManager()
        self.shared_state = SharedStateManager()

        # Pipeline components (initialized in start())
        self.visual_pipeline: Optional[VisualPipeline] = None
        self.audio_pipeline: Optional[AudioPipeline] = None
        self.fusion_manager: Optional[FusionManager] = None
        self.capture_manager: Optional[CaptureManager] = None
        self.display_manager: Optional[DisplayManager] = None

        # Thread control
        self.running = False
        self._shutdown_event = threading.Event()

        # Metrics
        self.start_time = None
        self.uptime_sec = 0

        logger.info("SystemManager initialized")

    def start(self) -> None:
        """Start all system components and threads.

        Initializes pipelines, starts threads, sets up signal handlers.
        """
        if self.running:
            logger.warning("System already running")
            return

        logger.info("Starting Threat Detection System...")

        try:
            # Initialize pipelines
            logger.info("Phase 1: Initializing pipelines...")
            self._init_pipelines()

            # Start threads
            logger.info("Phase 2: Starting threads...")
            self._start_threads()

            # Start camera and audio capture (async, won't block startup)
            logger.info("Phase 3: Starting camera and audio capture...")
            self._start_capture()

            # Start display manager
            logger.info("Phase 4: Starting display manager...")
            self._start_display()

            # Setup signal handlers
            logger.info("Phase 5: Setting up signal handlers...")
            self._setup_signal_handlers()

            self.running = True
            self.start_time = time.time()
            logger.info("✅ System started successfully")

        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            self._emergency_stop()
            raise

    def stop(self) -> None:
        """Stop all system components gracefully.

        Stops threads, cleans up resources, logs final metrics.
        """
        if not self.running:
            logger.warning("System not running")
            return

        logger.info("Stopping Threat Detection System...")

        try:
            # Signal shutdown
            self.running = False
            self._shutdown_event.set()

            # Stop threads in reverse order
            self._stop_threads()

            # Calculate uptime
            if self.start_time:
                self.uptime_sec = time.time() - self.start_time

            logger.info(f"System uptime: {self.uptime_sec:.1f} seconds")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            self._emergency_stop()

    def _init_pipelines(self) -> None:
        """Initialize all pipeline components."""
        logger.info("Initializing pipelines...")

        # Visual Pipeline
        logger.info("  [Visual] Initializing PersonDetector...")
        visual_config = self.config.get("visual", {})
        logger.info("  [Visual] Creating VisualPipeline...")
        self.visual_pipeline = VisualPipeline(
            person_model_path=visual_config.get("person_model", "yolov8n.pt"),
            weapon_model_path=visual_config.get("weapon_model", "models/weapon_best.pt"),
            person_conf=visual_config.get("person_conf", 0.60),
            person_min_area=visual_config.get("person_min_area", 3500),
            target_fps=self.config["system"].get("target_fps", 30.0),
            weapon_detect_every_n_frames=visual_config.get("weapon_detect_every_n", 3),
            face_recognize_every_n_frames=visual_config.get("face_recognize_every_n", 10),
            visual_to_fusion_queue=self.queue_manager.visual_to_fusion_q,
            fusion_to_visual_queue=self.queue_manager.fusion_to_visual_q,
        )
        logger.info("VisualPipeline initialized")

        # Audio Pipeline
        logger.info("  [Audio] Initializing AudioPipeline...")
        audio_config = self.config.get("audio", {})
        self.audio_pipeline = AudioPipeline(
            queue_manager=self.queue_manager,
            device=audio_config.get("device", "cpu"),
        )
        logger.info("AudioPipeline initialized")

        # Fusion Manager
        logger.info("  [Fusion] Initializing FusionManager...")
        fusion_config = self.config.get("fusion", {})
        self.fusion_manager = FusionManager(
            shared_state=self.shared_state,
            queue_manager=self.queue_manager,
            target_hz=self.config["system"].get("fusion_hz", 1.0),
        )

        logger.info("Pipelines initialized")

    def _start_threads(self) -> None:
        """Start all pipeline threads."""
        logger.info("Starting threads...")

        # Start in dependency order
        self.fusion_manager.start()
        self.visual_pipeline.start()
        self.audio_pipeline.start()

        logger.info("Threads started")

    def _start_capture(self) -> None:
        """Start camera and audio capture."""
        logger.info("Starting camera and audio capture...")

        try:
            visual_config = self.config.get("visual", {})
            camera_id = visual_config.get("camera_id", 0)
            target_fps = self.config["system"].get("target_fps", 30.0)

            audio_config = self.config.get("audio", {})
            sample_rate = audio_config.get("sample_rate", 16000)

            self.capture_manager = CaptureManager(
                camera_id=camera_id,
                target_fps=target_fps,
                sample_rate=sample_rate
            )
            
            # Start capture in a separate thread to avoid blocking
            capture_thread = threading.Thread(
                target=self._start_capture_thread,
                name="CaptureStartup",
                daemon=True
            )
            capture_thread.start()
            logger.info("Camera and audio capture startup initiated")

        except Exception as e:
            logger.error(f"Failed to init capture: {e}")
            logger.warning("System will run without camera/audio input (logic only)")

    def _start_capture_thread(self) -> None:
        """Start capture in background thread."""
        try:
            if self.capture_manager:
                self.capture_manager.start(self.visual_pipeline, self.audio_pipeline)
                logger.info("✅ Camera and audio capture started successfully")
        except Exception as e:
            logger.warning(f"Camera/audio capture unavailable: {e}")
            logger.info("System running in logic-only mode (no hardware input)")

    def _start_display(self) -> None:
        """Start display manager."""
        logger.info("Starting display manager...")

        try:
            self.display_manager = DisplayManager(
                target_fps=self.config["system"].get("target_fps", 30.0)
            )
            self.display_manager.start(self.visual_pipeline, self.shared_state)
            logger.info("Display manager started")

        except Exception as e:
            logger.warning(f"Display unavailable: {e}")
            logger.info("System running without visual display")

    def _stop_threads(self) -> None:
        """Stop all pipeline threads."""
        logger.info("Stopping threads...")

        # Stop capture first
        if self.capture_manager:
            try:
                self.capture_manager.stop()
            except Exception as e:
                logger.error(f"Error stopping capture: {e}")

        # Stop display
        if self.display_manager:
            try:
                self.display_manager.stop()
            except Exception as e:
                logger.error(f"Error stopping display: {e}")

        # Stop in reverse order
        if self.audio_pipeline:
            self.audio_pipeline.stop()
        if self.visual_pipeline:
            self.visual_pipeline.stop()
        if self.fusion_manager:
            self.fusion_manager.stop()

        logger.info("Threads stopped")

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            self.stop()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Windows doesn't have SIGTERM, but SIGINT (Ctrl+C) works
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, signal_handler)

    def _emergency_stop(self) -> None:
        """Emergency stop - force terminate all threads."""
        logger.error("Emergency stop initiated")

        self.running = False
        self._shutdown_event.set()

        # Force stop threads
        threads = [
            getattr(self.audio_pipeline, '_thread', None),
            getattr(self.visual_pipeline, 'pipeline_thread', None),
            getattr(self.fusion_manager, '_thread', None),
        ]

        for thread in threads:
            if thread and thread.is_alive():
                logger.warning(f"Force terminating thread: {thread.name}")
                # Note: In Python, we can't force terminate threads
                # This is just for logging

        logger.error("Emergency stop complete")

    def get_status(self) -> Dict[str, Any]:
        """Get system status and metrics.

        Returns:
            Status dictionary with component states and metrics
        """
        status = {
            "running": self.running,
            "uptime_sec": self.uptime_sec if self.start_time else 0,
            "config": self.config,
        }

        # Component status
        if self.visual_pipeline:
            status["visual"] = {
                "running": self.visual_pipeline.running,
                "frames_processed": getattr(self.visual_pipeline, 'frames_processed', 0),
            }

        if self.audio_pipeline:
            status["audio"] = {
                "running": self.audio_pipeline._running,
                "chunks_processed": self.audio_pipeline.chunks_processed,
                "events_emitted": self.audio_pipeline.events_emitted,
            }

        if self.fusion_manager:
            status["fusion"] = {
                "running": self.fusion_manager._running,
                "loop_count": self.fusion_manager.loop_count,
                "alerts_triggered": self.fusion_manager.alerts_triggered,
            }

        # Shared state
        status["shared_state"] = dict(self.shared_state._state)

        return status

    def wait_for_shutdown(self) -> None:
        """Wait for shutdown signal (blocking)."""
        self._shutdown_event.wait()