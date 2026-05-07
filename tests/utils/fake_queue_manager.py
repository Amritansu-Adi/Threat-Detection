from typing import List
from core.data_structures import AudioEvent


class FakeQueueManager:
    def __init__(self):
        self.audio_events: List[AudioEvent] = []

    def put_audio_event(self, event: AudioEvent, block: bool = True) -> bool:
        self.audio_events.append(event)
        return True
