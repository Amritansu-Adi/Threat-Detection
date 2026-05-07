"""Queue management for Late Fusion System.

This module handles creation and management of all inter-thread communication
queues for the Late Fusion State Machine architecture.

Queue Architecture:
    Visual Thread → visual_to_fusion_q → Fusion Manager
    Audio Thread → audio_to_fusion_q → Fusion Manager
    Fusion Manager → fusion_to_visual_q → Visual Thread
"""

from queue import Queue
from typing import Optional, Tuple
import logging

from .data_structures import VisualData, AudioEvent, FusionInterrupt

logger = logging.getLogger(__name__)


class QueueManager:
    """Manager for all queues in Late Fusion system.

    Provides factory methods and lifecycle management for queues connecting
    visual, audio, and fusion components.

    Attributes:
        visual_to_fusion_q: Queue for visual data → fusion manager
        audio_to_fusion_q: Queue for audio events → fusion manager
        fusion_to_visual_q: Queue for control signals → visual pipeline
    """

    def __init__(self, maxsize: int = 100):
        """Initialize all queues.

        Args:
            maxsize: Maximum queue size (0 = unlimited)
                     Recommended: 100 for visual (30 FPS), 500 for audio (VAD chunks)
        """
        self.maxsize = maxsize
        
        # Main communication queues
        # Visual queue: 30 FPS rate, store ~3 seconds
        self.visual_to_fusion_q: Queue = Queue(maxsize=max(100, maxsize))
        logger.info(f"Created visual_to_fusion_q (maxsize={self.visual_to_fusion_q.maxsize})")

        # Audio queue: ~2 second chunks, store ~10 chunks
        self.audio_to_fusion_q: Queue = Queue(maxsize=max(20, maxsize))
        logger.info(f"Created audio_to_fusion_q (maxsize={self.audio_to_fusion_q.maxsize})")

        # Control signal queue: Small, fusion mostly sends on-demand
        self.fusion_to_visual_q: Queue = Queue(maxsize=50)
        logger.info(f"Created fusion_to_visual_q (maxsize={self.fusion_to_visual_q.maxsize})")

    def put_visual_data(self, data: VisualData, block: bool = False) -> bool:
        """Put visual data to fusion queue (non-blocking by default).

        Visual pipeline should not block on queue full - drop frame instead.

        Args:
            data: VisualData to send
            block: Whether to block if queue full

        Returns:
            bool: True if successfully queued, False if queue full
        """
        try:
            self.visual_to_fusion_q.put(data, block=block, timeout=0.001)
            return True
        except:
            # Queue full - drop frame (acceptable for visual at 30 FPS)
            return False

    def put_audio_event(self, event: AudioEvent, block: bool = True) -> bool:
        """Put audio event to fusion queue (blocking by default).

        Audio pipeline should block to ensure events are not dropped.

        Args:
            event: AudioEvent to send
            block: Whether to block if queue full

        Returns:
            bool: True if successfully queued
        """
        try:
            self.audio_to_fusion_q.put(event, block=block, timeout=1.0)
            return True
        except:
            logger.warning("Audio queue full - event dropped")
            return False

    def put_fusion_interrupt(self, interrupt: FusionInterrupt, block: bool = True) -> bool:
        """Put control interrupt to visual queue.

        Args:
            interrupt: FusionInterrupt command
            block: Whether to block if queue full

        Returns:
            bool: True if successfully queued
        """
        try:
            self.fusion_to_visual_q.put(interrupt, block=block, timeout=0.1)
            return True
        except:
            logger.warning("Fusion interrupt queue full")
            return False

    def get_visual_data(self, block: bool = False, timeout: Optional[float] = None) -> Optional[VisualData]:
        """Get latest visual data from queue (non-blocking by default).

        Args:
            block: Whether to block if queue empty
            timeout: Timeout in seconds if blocking

        Returns:
            VisualData or None if queue empty
        """
        try:
            return self.visual_to_fusion_q.get(block=block, timeout=timeout)
        except:
            return None

    def get_audio_event(self, block: bool = False, timeout: Optional[float] = None) -> Optional[AudioEvent]:
        """Get latest audio event from queue (non-blocking by default).

        Args:
            block: Whether to block if queue empty
            timeout: Timeout in seconds if blocking

        Returns:
            AudioEvent or None if queue empty
        """
        try:
            return self.audio_to_fusion_q.get(block=block, timeout=timeout)
        except:
            return None

    def get_fusion_interrupt(self, block: bool = False, timeout: Optional[float] = None) -> Optional[FusionInterrupt]:
        """Get control interrupt from queue (non-blocking by default).

        Args:
            block: Whether to block if queue empty
            timeout: Timeout in seconds if blocking

        Returns:
            FusionInterrupt or None if queue empty
        """
        try:
            return self.fusion_to_visual_q.get(block=block, timeout=timeout)
        except:
            return None

    def drain_visual_queue(self) -> Tuple[int, Optional[VisualData]]:
        """Drain visual queue and return count and latest item.

        Used by fusion manager to get latest visual data and drop old items.

        Returns:
            Tuple[count, latest_data]: Number of items drained, latest VisualData
        """
        count = 0
        latest = None

        while True:
            item = self.get_visual_data(block=False)
            if item is None:
                break
            count += 1
            latest = item

        return count, latest

    def drain_audio_queue(self) -> Tuple[int, Optional[AudioEvent]]:
        """Drain audio queue and return count and latest item.

        Used by fusion manager to get latest audio event and drop old items.

        Returns:
            Tuple[count, latest_event]: Number of items drained, latest AudioEvent
        """
        count = 0
        latest = None

        while True:
            item = self.get_audio_event(block=False)
            if item is None:
                break
            count += 1
            latest = item

        return count, latest

    def qsize(self) -> dict:
        """Get current sizes of all queues.

        Returns:
            dict: {visual_q_size, audio_q_size, control_q_size}
        """
        return {
            "visual_q": self.visual_to_fusion_q.qsize(),
            "audio_q": self.audio_to_fusion_q.qsize(),
            "control_q": self.fusion_to_visual_q.qsize(),
        }

    def clear_all(self) -> None:
        """Clear all queues (for testing/reset).

        WARNING: This will drop any pending messages!
        """
        logger.debug("Clearing all queues - dropping pending messages")
        self.drain_visual_queue()
        self.drain_audio_queue()

        while not self.fusion_to_visual_q.empty():
            try:
                self.fusion_to_visual_q.get(block=False)
            except:
                break

    def close(self) -> None:
        """Close all queues (called at shutdown).

        This ensures proper cleanup of queue resources.
        """
        logger.info("Closing all queues")
        # Note: Queue resources are automatically cleaned up by garbage collection
        # Explicitly closing here is optional but good practice
