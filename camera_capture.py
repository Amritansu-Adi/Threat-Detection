"""Camera and audio capture for real-time threat detection system.

Captures video frames from webcam and audio from microphone,
then submits them to the visual and audio pipelines.
"""

import cv2
import logging
import numpy as np
import threading
import queue
import time
from typing import Optional

# Optional audio capture
try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False

logger = logging.getLogger(__name__)


class CameraCapture:
    """Capture video frames from webcam."""

    def __init__(self, camera_id: int = 0, target_fps: float = 30.0):
        """
        Initialize camera capture.

        Args:
            camera_id: Camera device ID (default 0 = built-in webcam)
            target_fps: Target FPS for capture
        """
        self.camera_id = camera_id
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps if target_fps > 0 else 1.0 / 30.0
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def start(self, visual_pipeline):
        """
        Start camera capture thread..

        Args:
            visual_pipeline: VisualPipeline instance to submit frames to
        """
        if self.running:
            logger.warning("Camera capture already running")
            return

        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera {self.camera_id}")

            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

            self.running = True
            self.thread = threading.Thread(
                target=self._capture_loop,
                args=(visual_pipeline,),
                name="CameraCapture"
            )
            self.thread.daemon = True
            self.thread.start()
            logger.info(f"Camera capture started (camera_id={self.camera_id}, target_fps={self.target_fps})")

        except Exception as e:
            logger.error(f"Failed to start camera capture: {e}")
            if self.cap:
                self.cap.release()
            self.cap = None
            self.running = False
            raise

    def stop(self):
        """Stop camera capture."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        logger.info("Camera capture stopped")

    def _capture_loop(self, visual_pipeline):
        """Main camera capture loop."""
        print("[CAMERA] Camera capture loop started!", flush=True)
        frame_count = 0
        start_time = time.time()

        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    break

                # Submit frame to visual pipeline
                timestamp = time.time()
                print(f"[CAMERA] Submitting frame {frame_count} to visual pipeline", flush=True)
                visual_pipeline.submit_frame(frame, timestamp=timestamp)

                frame_count += 1

                # Calculate FPS
                elapsed = time.time() - start_time
                if elapsed > 1.0:
                    actual_fps = frame_count / elapsed
                    logger.debug(f"Camera capture FPS: {actual_fps:.1f}")
                    frame_count = 0
                    start_time = time.time()

                # Frame rate limiting
                time.sleep(self.frame_interval)

            except Exception as e:
                logger.error(f"Error in camera capture loop: {e}")
                break

        logger.info("Camera capture loop ended")


class AudioCapture:
    """Capture audio from microphone."""

    def __init__(self, sample_rate: int = 16000, chunk_size: int = 1600):
        """
        Initialize audio capture.

        Args:
            sample_rate: Audio sample rate (Hz)
            chunk_size: Samples per chunk
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.audio_queue: queue.Queue = queue.Queue()

    def start(self, audio_pipeline):
        """
        Start audio capture thread.

        Args:
            audio_pipeline: AudioPipeline instance to submit chunks to
        """
        if not HAS_SOUNDDEVICE:
            logger.warning("sounddevice not installed - audio capture unavailable")
            return

        if self.running:
            logger.warning("Audio capture already running")
            return

        try:
            self.running = True
            self.thread = threading.Thread(
                target=self._capture_loop,
                args=(audio_pipeline,),
                name="AudioCapture"
            )
            self.thread.daemon = True
            self.thread.start()
            logger.info(f"Audio capture started (sr={self.sample_rate}Hz, chunk={self.chunk_size})")

        except Exception as e:
            logger.error(f"Failed to start audio capture: {e}")
            self.running = False
            raise

    def stop(self):
        """Stop audio capture."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        logger.info("Audio capture stopped")

    def _capture_loop(self, audio_pipeline):
        """Main audio capture loop using sounddevice."""
        try:
            # Get default audio device for input
            device_info = sd.query_devices(kind='input')
            logger.info(f"Using audio device: {device_info}")

            def audio_callback(indata, frames, time_info, status):
                """Callback for audio stream data."""
                if status:
                    logger.warning(f"Audio callback status: {status}")

                try:
                    # Copy audio data (handle mono or stereo)
                    if indata.ndim == 1:
                        audio_data = indata.copy()
                    else:
                        audio_data = indata[:, 0].copy()  # Take first channel if stereo

                    # Submit to audio pipeline
                    audio_pipeline.submit_audio_chunk(audio_data.astype(np.float32))
                except Exception as e:
                    logger.error(f"Error in audio callback: {e}")

            # Start audio stream
            with sd.InputStream(
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                callback=audio_callback,
                latency='low'
            ):
                logger.info("Audio stream started, listening for input...")
                while self.running:
                    time.sleep(0.1)

        except Exception as e:
            logger.error(f"Error in audio capture loop: {e}")
        finally:
            logger.info("Audio capture loop ended")


class CaptureManager:
    """Manages both camera and audio capture."""

    def __init__(self, camera_id: int = 0, target_fps: float = 30.0, sample_rate: int = 16000):
        """
        Initialize capture manager.

        Args:
            camera_id: Camera device ID
            target_fps: Target capture FPS
            sample_rate: Audio sample rate
        """
        self.camera = CameraCapture(camera_id=camera_id, target_fps=target_fps)
        self.audio = AudioCapture(sample_rate=sample_rate)
        self.running = False

    def start(self, visual_pipeline, audio_pipeline):
        """
        Start both camera and audio capture.

        Args:
            visual_pipeline: VisualPipeline instance
            audio_pipeline: AudioPipeline instance
        """
        try:
            self.camera.start(visual_pipeline)
            self.audio.start(audio_pipeline)
            self.running = True
            logger.info("✅ Capture manager started (camera + audio)")
        except Exception as e:
            logger.error(f"Failed to start captures: {e}")
            self.stop()
            raise

    def stop(self):
        """Stop both camera and audio capture."""
        self.running = False
        try:
            self.camera.stop()
        except Exception as e:
            logger.error(f"Error stopping camera: {e}")

        try:
            self.audio.stop()
        except Exception as e:
            logger.error(f"Error stopping audio: {e}")

        logger.info("Capture manager stopped")
