"""Display module for real-time threat detection visualization.

Shows camera feed with detected persons, weapons, and current threat level.
"""

import cv2
import logging
import threading
import time
import numpy as np
from typing import Optional, Dict, Any
from collections import deque

logger = logging.getLogger(__name__)


class DisplayManager:
    """Manages real-time display of camera feed with threat annotations."""

    def __init__(self, target_fps: float = 30.0):
        """
        Initialize display manager.

        Args:
            target_fps: Target FPS for display
        """
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps if target_fps > 0 else 1.0 / 30.0
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.current_frame: Optional[np.ndarray] = None
        self.current_state: Dict[str, Any] = {
            'risk': 0.0,
            'state': 'IDLE',
            'persons': [],
            'weapons': [],
            'stats': {}
        }

    def start(self, visual_pipeline, shared_state):
        """
        Start display thread.

        Args:
            visual_pipeline: VisualPipeline instance
            shared_state: SharedStateManager for risk info
        """
        if self.running:
            logger.warning("Display already running")
            return

        try:
            self.running = True
            self.thread = threading.Thread(
                target=self._display_loop,
                args=(visual_pipeline, shared_state),
                name="DisplayManager",
                daemon=True
            )
            self.thread.start()
            logger.info("Display manager started")

        except Exception as e:
            logger.error(f"Failed to start display: {e}")
            self.running = False
            raise

    def stop(self):
        """Stop display."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        cv2.destroyAllWindows()
        logger.info("Display manager stopped")

    def _display_loop(self, visual_pipeline, shared_state):
        """Main display loop."""
        window_name = "Threat Detection System"
        fps_deque = deque(maxlen=30)

        while self.running:
            try:
                # Get current frame from visual pipeline (if available)
                if hasattr(visual_pipeline, '_last_frame'):
                    frame = visual_pipeline._last_frame.copy() if visual_pipeline._last_frame is not None else None
                else:
                    frame = None

                # Get threat info from shared state
                risk = shared_state.get_risk()
                state = shared_state.get_state()
                visual_summary = shared_state.get_visual_summary()
                audio_summary = shared_state.get_audio_summary()

                if frame is not None:
                    # Draw annotations
                    annotated_frame = self._annotate_frame(
                        frame,
                        risk,
                        state,
                        visual_summary,
                        audio_summary
                    )

                    # Display
                    cv2.imshow(window_name, annotated_frame)

                    # FPS tracking
                    current_time = time.time()
                    fps_deque.append(current_time)
                    if len(fps_deque) > 1:
                        fps = len(fps_deque) / (fps_deque[-1] - fps_deque[0])
                        logger.debug(f"Display FPS: {fps:.1f}")

                else:
                    # Create a placeholder with text
                    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(
                        placeholder,
                        "Waiting for camera frames...",
                        (150, 240),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2
                    )
                    cv2.putText(
                        placeholder,
                        f"Risk: {risk:.1f} | State: {state}",
                        (150, 280),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 165, 255),
                        2
                    )
                    cv2.imshow(window_name, placeholder)

                # Handle window close
                key = cv2.waitKey(int(self.frame_interval * 1000)) & 0xFF
                if key == ord('q') or key == 27:  # q or ESC
                    logger.info("Display closed by user")
                    self.running = False
                    break

            except Exception as e:
                logger.error(f"Error in display loop: {e}")
                break

        cv2.destroyAllWindows()
        logger.info("Display loop ended")

    def _annotate_frame(self, frame: np.ndarray, risk: float, state: str, 
                       visual_summary: Dict, audio_summary: Dict) -> np.ndarray:
        """
        Add threat annotations to frame.

        Args:
            frame: Input frame
            risk: Current risk score
            state: Current threat state
            visual_summary: Visual pipeline summary
            audio_summary: Audio pipeline summary

        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # Color based on threat level
        if state == 'CRITICAL':
            color = (0, 0, 255)  # Red
            text_color = (0, 0, 255)
        elif state == 'ALERT':
            color = (0, 165, 255)  # Orange
            text_color = (0, 165, 255)
        elif state == 'EVALUATING':
            color = (0, 255, 255)  # Yellow
            text_color = (0, 255, 255)
        elif state == 'CAUTION':
            color = (0, 255, 0)  # Green
            text_color = (0, 255, 0)
        else:
            color = (255, 255, 255)  # White (IDLE)
            text_color = (255, 255, 255)

        # Draw border based on threat level
        thickness = 5 if state in ['CRITICAL', 'ALERT'] else 2
        cv2.rectangle(annotated, (0, 0), (w - 1, h - 1), color, thickness)

        # Draw threat info box (top-right)
        info_y = 30
        cv2.putText(
            annotated,
            f"Risk: {risk:.1f}",
            (w - 250, info_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            text_color,
            2
        )
        cv2.putText(
            annotated,
            f"State: {state}",
            (w - 250, info_y + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            text_color,
            2
        )

        # Draw stats (bottom-left)
        stats_y = h - 120
        persons = visual_summary.get('persons_count', 0)
        unknown = visual_summary.get('unknown_count', 0)
        weapons = visual_summary.get('weapons_count', 0)
        unknown_with_weapon = visual_summary.get('unknown_with_weapon', False)

        cv2.putText(
            annotated,
            f"Persons: {persons} (Unknown: {unknown})",
            (10, stats_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            1
        )
        cv2.putText(
            annotated,
            f"Weapons: {weapons}",
            (10, stats_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255) if weapons > 0 else (255, 255, 255),
            1
        )

        if unknown_with_weapon:
            cv2.putText(
                annotated,
                "⚠ UNKNOWN WITH WEAPON",
                (10, stats_y + 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # Draw audio info if available
        if audio_summary:
            transcript = audio_summary.get('transcript', '')[:40]  # Truncate
            emotion = audio_summary.get('emotion', 'N/A')
            intent = audio_summary.get('intent', 'N/A')

            if transcript:
                cv2.putText(
                    annotated,
                    f"Audio: {transcript}",
                    (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 255),
                    1
                )
            cv2.putText(
                annotated,
                f"Emotion: {emotion} | Intent: {intent}",
                (10, info_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 165, 0),
                1
            )

        return annotated
