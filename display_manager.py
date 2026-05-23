"""Display module for real-time threat detection visualization.

Shows camera feed with detected persons, weapons, and current threat level.
"""

import os
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
        # Keep UI clean by default; optional debug overlay can be enabled via env.
        self.show_object_boxes = os.getenv("THREAT_SHOW_OBJECT_BOXES", "0").lower() in ("1", "true", "yes")

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
                risk_details = (
                    shared_state.get_risk_details()
                    if hasattr(shared_state, "get_risk_details")
                    else {}
                )

                if frame is not None:
                    visual_data = getattr(visual_pipeline, 'latest_visual_data', None)
                    # Draw annotations
                    annotated_frame = self._annotate_frame(
                        frame,
                        risk,
                        state,
                        visual_summary,
                        audio_summary,
                        visual_data=visual_data,
                        risk_details=risk_details,
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
                       visual_summary: Dict, audio_summary: Dict,
                       visual_data: Optional[Any] = None,
                       risk_details: Optional[Dict[str, Any]] = None,
                       title: str = "Threat Detection") -> np.ndarray:
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
        state_value = getattr(state, 'value', str(state))

        # Color based on threat level
        if state_value == 'CRITICAL':
            color = (0, 0, 255)  # Red
            text_color = (0, 0, 255)
        elif state_value == 'ALERT':
            color = (0, 165, 255)  # Orange
            text_color = (0, 165, 255)
        elif state_value == 'EVALUATING':
            color = (0, 255, 255)  # Yellow
            text_color = (0, 255, 255)
        elif state_value == 'CAUTION':
            color = (0, 255, 0)  # Green
            text_color = (0, 255, 0)
        else:
            color = (255, 255, 255)  # White (IDLE)
            text_color = (255, 255, 255)

        # Draw border based on threat level
        thickness = 5 if state_value in ['CRITICAL', 'ALERT'] else 2
        cv2.rectangle(annotated, (0, 0), (w - 1, h - 1), color, thickness)

        self._draw_detections(annotated, visual_data)

        # Draw threat info box (top-right)
        info_y = 30
        cv2.putText(
            annotated,
            title,
            (10, h - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (220, 220, 220),
            2
        )
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
            f"State: {state_value}",
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

        if risk_details:
            visual_risk = float(risk_details.get('visual_risk', 0.0))
            audio_risk = float(risk_details.get('audio_risk', 0.0))
            factors = list(risk_details.get('contributing_factors', []) or [])
            audio_floor_active = any(
                "audio escalation floor active" in str(f).lower()
                for f in factors
            )
            cv2.putText(
                annotated,
                f"V {visual_risk:.1f} | A {audio_risk:.1f}",
                (w - 250, info_y + 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (220, 220, 220),
                1
            )
            cv2.putText(
                annotated,
                f"AUDIO CHECK: {'FLOOR ON' if audio_floor_active else 'NORMAL'}",
                (w - 250, info_y + 102),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 220, 255) if audio_floor_active else (180, 180, 180),
                1
            )
            reason_y = info_y + 124
            for factor in factors[:3]:
                clean = str(factor).lstrip("- ").strip()
                if not clean:
                    continue
                if len(clean) > 32:
                    clean = clean[:29] + "..."
                cv2.putText(
                    annotated,
                    clean,
                    (w - 250, reason_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (220, 220, 220),
                    1
                )
                reason_y += 22

        # Draw audio info if available
        if audio_summary:
            transcript = audio_summary.get('transcript', '')[:40]  # Truncate
            emotion = audio_summary.get('emotion', 'N/A')
            emotion_conf = float(audio_summary.get('emotion_conf', 0.0))
            intent = audio_summary.get('intent', 'N/A')
            intent_conf = float(audio_summary.get('intent_conf', 0.0))

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
                f"Emotion: {emotion} ({emotion_conf:.2f}) | Intent: {intent} ({intent_conf:.2f})",
                (10, info_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 0, 0),
                1
            )

        return annotated

    def annotate_frame(self, frame: np.ndarray, risk: float, state: str,
                       visual_summary: Dict, audio_summary: Dict,
                       visual_data: Optional[Any] = None,
                       risk_details: Optional[Dict[str, Any]] = None,
                       title: str = "Threat Detection") -> np.ndarray:
        """Public helper for offline video/demo rendering."""
        return self._annotate_frame(
            frame,
            risk,
            state,
            visual_summary,
            audio_summary,
            visual_data=visual_data,
            risk_details=risk_details,
            title=title,
        )

    def _draw_detections(self, frame: np.ndarray, visual_data: Optional[Any]) -> None:
        """Draw latest person/object/weapon detections on a frame."""
        if not visual_data:
            return

        persons = getattr(visual_data, 'persons', []) or []
        weapons = getattr(visual_data, 'weapons', []) or []
        objects = getattr(visual_data, 'processing_stats', {}).get('objects_detected', [])

        person_boxes = []
        for person in persons:
            bbox = person.get('bbox') if isinstance(person, dict) else getattr(person, 'bbox', None)
            if not bbox:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            person_boxes.append((x1, y1, x2, y2))
            identity = person.get('identity', person.get('label', 'UNKNOWN')) if isinstance(person, dict) else getattr(person, 'identity', 'UNKNOWN')
            person_id = person.get('id', '?') if isinstance(person, dict) else getattr(person, 'id', '?')
            has_weapon = person.get('has_weapon', False) if isinstance(person, dict) else getattr(person, 'has_weapon', False)
            color = (0, 0, 255) if has_weapon else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if has_weapon else 2)
            self._draw_label(frame, x1, y1, f"person {person_id}: {identity}", color)

        weapon_types = {'knife', 'scissors', 'gun', 'pistol', 'rifle'}
        for weapon in weapons:
            box = getattr(weapon, 'bbox', None)
            if not box:
                continue
            x1, y1, x2, y2 = [int(v) for v in (box.x1, box.y1, box.x2, box.y2)]
            label = getattr(weapon, 'weapon_type', 'weapon')
            conf = getattr(weapon, 'confidence', 0.0)
            det_id = getattr(weapon, 'detection_id', None)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            if det_id:
                self._draw_label(frame, x1, y1, f"{det_id} {label} {conf:.2f}", (0, 0, 255))
            else:
                self._draw_label(frame, x1, y1, f"{label} {conf:.2f}", (0, 0, 255))

        if self.show_object_boxes:
            for obj in objects:
                label = str(obj.get('label', '')).lower()
                if label == 'person':
                    continue
                bbox = obj.get('bbox')
                if not bbox:
                    continue
                x1, y1, x2, y2 = [int(v) for v in bbox]
                if self._inside_any_person((x1, y1, x2, y2), person_boxes) and label not in weapon_types:
                    continue
                color = (0, 0, 255) if label in weapon_types else (255, 180, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                self._draw_label(frame, x1, y1, f"{label} {obj.get('confidence', 0):.2f}", color)

    @staticmethod
    def _draw_label(frame: np.ndarray, x: int, y: int, text: str, color: tuple) -> None:
        y = max(18, y)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x, y - th - 8), (x + tw + 6, y), color, -1)
        cv2.putText(frame, text, (x + 3, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    @staticmethod
    def _inside_any_person(bbox: tuple, person_boxes: list) -> bool:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        for px1, py1, px2, py2 in person_boxes:
            if px1 <= cx <= px2 and py1 <= cy <= py2:
                return True
        return False
