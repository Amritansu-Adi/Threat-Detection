# Implementation Guide - Late Fusion State Machine

## Overview
This guide provides step-by-step instructions for implementing the Late Fusion State Machine architecture with all three components: VisualPipeline, AudioPipeline, and FusionManager.

## Project Structure

```
yolo/
├── camera_detection/
│   ├── yolo_detect.py           (Refactored as VisualPipeline thread)
│   └── visual_pipeline.py        (New: VisualPipeline class)
├── audio/                        (New directory)
│   ├── __init__.py
│   ├── audio_pipeline.py         (AudioPipeline class)
│   ├── vad_processor.py          (Silero VAD)
│   ├── emotion_analyzer.py       (DistilHuBERT SER)
│   ├── intent_classifier.py      (DistilBERT intent)
│   └── audio_buffer.py           (Sliding window buffer)
├── fusion/                       (New directory)
│   ├── __init__.py
│   ├── fusion_manager.py         (FusionManager class)
│   ├── risk_accumulator.py       (Risk calculation logic)
│   ├── state_manager.py          (State machine)
│   └── interrupts.py             (Interrupt handling)
├── core/                         (New directory)
│   ├── __init__.py
│   ├── system_manager.py         (Thread coordination)
│   ├── data_structures.py        (Type definitions)
│   └── shared_state.py           (Shared memory management)
├── main.py                       (Entry point - orchestrates everything)
└── config.yaml                   (Configuration)
```

## Step-by-Step Implementation

### Step 1: Core Data Structures and Definitions

Create `core/data_structures.py`:

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import time

@dataclass
class TrackedPerson:
    """A person being tracked across frames."""
    track_id: int
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    identity: str = "Unknown"
    confidence: float = 0.0
    has_weapon: bool = False
    frames_in_view: int = 0
    last_detected_frame: int = 0

@dataclass
class WeaponDetection:
    """Detected weapon with location and confidence."""
    bbox: Tuple[float, float, float, float]
    weapon_type: str
    confidence: float
    associated_person_id: Optional[int] = None
    frames_detected: int = 0

@dataclass
class AudioEvent:
    """Complete audio analysis result."""
    timestamp: float
    transcript: str = ""
    emotion: str = "neutral"
    emotion_confidence: float = 0.0
    intent: str = "neutral"
    intent_confidence: float = 0.0
    audio_segment: Optional[bytes] = None

@dataclass
class VisualData:
    """All visual processing results."""
    timestamp: float
    frame_idx: int
    tracked_persons: List[TrackedPerson] = field(default_factory=list)
    weapons: List[WeaponDetection] = field(default_factory=list)
    unknown_persons_count: int = 0
    persons_with_weapons: int = 0

@dataclass
class RiskEvent:
    """Risk score update event."""
    timestamp: float
    score: float
    state: str
    previous_state: str = ""
    visual_contribution: float = 0.0
    audio_contribution: float = 0.0
    decay_component: float = 0.0
    sensory_component: float = 0.0
    contributing_factors: Dict[str, float] = field(default_factory=dict)

@dataclass
class FusionInterrupt:
    """Command from fusion manager to visual thread."""
    command: str  # 'FOCUS_RE_INFERENCE', 'HEIGHTEN_SENSITIVITY', etc.
    zones: List[Tuple[int, int, int, int]] = field(default_factory=list)
    priority: bool = False
    timestamp: float = 0.0
```

### Step 2: Shared State Management

Create `core/shared_state.py`:

```python
from multiprocessing import Manager
from typing import Dict, Any
import time

class SharedState:
    """Thread-safe shared state management."""
    
    def __init__(self):
        self.manager = Manager()
        self._state = self.manager.dict()
        self._initialize_state()
    
    def _initialize_state(self):
        """Set up all shared state variables."""
        self._state['risk_score'] = 0
        self._state['state'] = 'IDLE'
        self._state['last_update'] = time.time()
        self._state['visual_persons'] = {}
        self._state['audio_transcript'] = ""
        self._state['emotion'] = "neutral"
        self._state['intent'] = "neutral"
        self._state['alert_active'] = False
        self._state['priority_zones'] = []
    
    def set(self, key: str, value: Any):
        """Set a value in shared state."""
        self._state[key] = value
        self._state['last_update'] = time.time()
    
    def get(self, key: str, default=None):
        """Get a value from shared state."""
        return self._state.get(key, default)
    
    def get_all(self) -> Dict:
        """Get all state as a dictionary."""
        return dict(self._state)
    
    def update(self, updates: Dict):
        """Update multiple values at once."""
        for key, value in updates.items():
            self._state[key] = value
        self._state['last_update'] = time.time()
```

### Step 3: Risk Accumulation Logic

Create `fusion/risk_accumulator.py`:

```python
from core.data_structures import VisualData, AudioEvent, RiskEvent
import time
import numpy as np

class RiskAccumulator:
    """Implements behavioral narrative with temporal persistence."""
    
    def __init__(self, shared_state):
        self.shared_state = shared_state
        
        # Temporal persistence parameters
        self.alpha = 0.9  # Retention factor (decay)
        self.beta = 1.0   # Confidence scaling factor
        
        # Risk thresholds
        self.THRESHOLD_CAUTION = 25
        self.THRESHOLD_EVALUATING = 50
        self.THRESHOLD_ALERT = 75
        self.THRESHOLD_CRITICAL = 90
        
        # Risk contribution weights
        self.VISUAL_WEIGHTS = {
            'unknown_person': 20,
            'weapon': 40,
            'unknown_with_weapon': 40,  # Additional
            'weapon_unassociated': 35
        }
        
        self.AUDIO_WEIGHTS = {
            'anger_high_conf': 25,    # >85% confidence
            'fear_high_conf': 30,     # >85% confidence
            'threat_intent': 45,      # >80% confidence
            'distress_intent': 35,    # >80% confidence
            'neutral': 0
        }
        
        self.last_risk_score = 0
        self.last_update_time = time.time()
    
    def calculate_risk(self, visual_data: VisualData, 
                      audio_data: AudioEvent) -> RiskEvent:
        """
        Calculate risk using:
        Risk_t = (Risk_{t-1} × α) + (SensoryInput × β)
        """
        current_time = time.time()
        time_delta = current_time - self.last_update_time
        time_delta = max(0.1, min(time_delta, 10.0))  # Clamp to [0.1, 10]
        
        # Decay component
        decay_factor = self.alpha ** time_delta
        decay_component = self.last_risk_score * decay_factor
        
        # Sensory input component
        sensory_component = 0
        visual_contribution = 0
        audio_contribution = 0
        
        if visual_data:
            visual_contribution = self._calculate_visual_risk(visual_data)
            sensory_component += visual_contribution
        
        if audio_data:
            audio_contribution = self._calculate_audio_risk(audio_data)
            sensory_component += audio_contribution
        
        # Apply confidence scaling
        sensory_component *= self.beta
        
        # Calculate new risk score
        new_risk = decay_component + sensory_component
        new_risk = np.clip(new_risk, 0, 100)
        
        # Determine state
        old_state = self._get_state(self.last_risk_score)
        new_state = self._get_state(new_risk)
        
        # Create event
        event = RiskEvent(
            timestamp=current_time,
            score=new_risk,
            state=new_state,
            previous_state=old_state,
            visual_contribution=visual_contribution,
            audio_contribution=audio_contribution,
            decay_component=decay_component,
            sensory_component=sensory_component,
            contributing_factors={
                'decay': decay_component,
                'visual': visual_contribution,
                'audio': audio_contribution,
                'time_delta': time_delta,
                'decay_factor': decay_factor
            }
        )
        
        # Update state
        self.last_risk_score = new_risk
        self.last_update_time = current_time
        
        return event
    
    def _calculate_visual_risk(self, visual_data: VisualData) -> float:
        """Extract risk contribution from visual data."""
        risk = 0
        
        # Unknown persons
        unknown_count = sum(1 for p in visual_data.tracked_persons 
                           if p.identity == "Unknown")
        risk += unknown_count * self.VISUAL_WEIGHTS['unknown_person']
        
        # Weapons
        if visual_data.weapons:
            risk += len(visual_data.weapons) * self.VISUAL_WEIGHTS['weapon']
        
        # Unknown persons with weapons (additional penalty)
        persons_with_weapons = sum(
            1 for p in visual_data.tracked_persons 
            if p.has_weapon and p.identity == "Unknown"
        )
        risk += persons_with_weapons * self.VISUAL_WEIGHTS['unknown_with_weapon']
        
        # Unassociated weapons (weapons not held by tracked persons)
        unassociated = sum(
            1 for w in visual_data.weapons 
            if w.associated_person_id is None
        )
        risk += unassociated * self.VISUAL_WEIGHTS['weapon_unassociated']
        
        return risk
    
    def _calculate_audio_risk(self, audio_data: AudioEvent) -> float:
        """Extract risk contribution from audio data."""
        risk = 0
        
        emotion = audio_data.emotion.lower()
        emotion_conf = audio_data.emotion_confidence
        
        # Emotion-based immediate boost
        if emotion == "anger" and emotion_conf > 0.85:
            risk += self.AUDIO_WEIGHTS['anger_high_conf']
        elif emotion == "fear" and emotion_conf > 0.85:
            risk += self.AUDIO_WEIGHTS['fear_high_conf']
        
        # Intent-based scoring
        intent = audio_data.intent.lower()
        intent_conf = audio_data.intent_confidence
        
        if intent == "threat" and intent_conf > 0.8:
            risk += self.AUDIO_WEIGHTS['threat_intent']
        elif intent == "distress" and intent_conf > 0.8:
            risk += self.AUDIO_WEIGHTS['distress_intent']
        
        return risk
    
    def _get_state(self, risk_score: float) -> str:
        """Determine state name from risk score."""
        if risk_score < self.THRESHOLD_CAUTION:
            return "IDLE"
        elif risk_score < self.THRESHOLD_EVALUATING:
            return "CAUTION"
        elif risk_score < self.THRESHOLD_ALERT:
            return "EVALUATING"
        elif risk_score < self.THRESHOLD_CRITICAL:
            return "ALERT"
        else:
            return "CRITICAL"
```

### Step 4: State Machine Implementation

Create `fusion/state_manager.py`:

```python
import time
from enum import Enum

class SystemState(Enum):
    """System states based on risk score."""
    IDLE = 0
    CAUTION = 1
    EVALUATING = 2
    ALERT = 3
    CRITICAL = 4

class StateManager:
    """Manages state transitions and durations."""
    
    STATE_BOUNDARIES = {
        SystemState.IDLE: (0, 25),
        SystemState.CAUTION: (25, 50),
        SystemState.EVALUATING: (50, 75),
        SystemState.ALERT: (75, 90),
        SystemState.CRITICAL: (90, 100)
    }
    
    def __init__(self, shared_state):
        self.shared_state = shared_state
        self.current_state = SystemState.IDLE
        self.state_entry_time = time.time()
        
        # Duration thresholds for state-based actions
        self.ALERT_DURATION_FOR_EMAIL = 3.0  # seconds
        self.CRITICAL_DURATION_IMMEDIATE = 0.5  # seconds
    
    def update(self, risk_score: float):
        """Update state based on risk score."""
        new_state = self._score_to_state(risk_score)
        
        if new_state != self.current_state:
            self._on_state_change(new_state)
            self.current_state = new_state
            self.state_entry_time = time.time()
        
        return self.current_state
    
    def _score_to_state(self, risk_score: float) -> SystemState:
        """Convert risk score to state."""
        for state, (min_score, max_score) in self.STATE_BOUNDARIES.items():
            if min_score <= risk_score < max_score:
                return state
        return SystemState.CRITICAL
    
    def _on_state_change(self, new_state: SystemState):
        """Handle state change."""
        self.shared_state.set('state', new_state.name)
        
        if new_state == SystemState.ALERT:
            print(f"[STATE CHANGE] → ALERT at {time.time()}")
        elif new_state == SystemState.CRITICAL:
            print(f"[STATE CHANGE] → CRITICAL at {time.time()}")
    
    def should_send_alert(self) -> bool:
        """Determine if alert should be sent."""
        if self.current_state in [SystemState.ALERT, SystemState.CRITICAL]:
            duration = time.time() - self.state_entry_time
            if self.current_state == SystemState.CRITICAL:
                return duration >= self.CRITICAL_DURATION_IMMEDIATE
            else:
                return duration >= self.ALERT_DURATION_FOR_EMAIL
        return False
```

### Step 5: Fusion Manager

Create `fusion/fusion_manager.py`:

```python
from multiprocessing import Queue
from queue import Empty
from fusion.risk_accumulator import RiskAccumulator
from fusion.state_manager import StateManager
from core.data_structures import FusionInterrupt
import time
import logging

logger = logging.getLogger(__name__)

class FusionManager:
    """Central coordination of visual and audio streams."""
    
    def __init__(self, visual_q: Queue, audio_q: Queue, 
                 fusion_to_visual_q: Queue, shared_state):
        self.visual_q = visual_q
        self.audio_q = audio_q
        self.fusion_to_visual_q = fusion_to_visual_q
        self.shared_state = shared_state
        
        self.risk_accumulator = RiskAccumulator(shared_state)
        self.state_manager = StateManager(shared_state)
        
        self.running = False
        self.last_visual = None
        self.last_audio = None
    
    def run(self):
        """Main fusion loop - runs at ~1 Hz."""
        self.running = True
        logger.info("Fusion Manager started")
        
        while self.running:
            try:
                # Collect latest events (non-blocking)
                self._collect_events()
                
                # Calculate new risk score
                risk_event = self.risk_accumulator.calculate_risk(
                    self.last_visual, self.last_audio
                )
                
                # Update state
                self.state_manager.update(risk_event.score)
                
                # Send interrupts if needed
                self._send_interrupts(risk_event)
                
                # Update shared state
                self.shared_state.update({
                    'risk_score': risk_event.score,
                    'state': risk_event.state,
                    'last_risk_event': risk_event
                })
                
                # Log and sleep
                logger.debug(f"Risk: {risk_event.score:.1f} | "
                           f"State: {risk_event.state}")
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Fusion error: {e}", exc_info=True)
                time.sleep(0.1)
    
    def _collect_events(self):
        """Collect latest events from both pipelines."""
        try:
            self.last_visual = self.visual_q.get_nowait()
        except Empty:
            pass
        
        try:
            self.last_audio = self.audio_q.get_nowait()
        except Empty:
            pass
    
    def _send_interrupts(self, risk_event):
        """Send control signals to visual pipeline."""
        # Send high-priority zone focus interrupt
        if risk_event.score > 50:
            interrupt = FusionInterrupt(
                command='FOCUS_RE_INFERENCE',
                zones=self._get_priority_zones(),
                priority=True,
                timestamp=time.time()
            )
            try:
                self.fusion_to_visual_q.put_nowait(interrupt)
            except:
                pass  # Queue full, skip
    
    def _get_priority_zones(self):
        """Determine priority zones from visual/audio data."""
        # TODO: Implement zone calculation based on threats
        return []
    
    def stop(self):
        """Stop the fusion manager."""
        self.running = False
        logger.info("Fusion Manager stopped")
```

### Step 6: System Manager

Create `core/system_manager.py`:

```python
from multiprocessing import Queue
import threading
import logging
from core.shared_state import SharedState

logger = logging.getLogger(__name__)

class SystemManager:
    """Coordinates all threads and system lifecycle."""
    
    def __init__(self, visual_pipeline, audio_pipeline, fusion_manager):
        self.visual_pipeline = visual_pipeline
        self.audio_pipeline = audio_pipeline
        self.fusion_manager = fusion_manager
        
        self.shared_state = SharedState()
        
        self.visual_thread = None
        self.audio_thread = None
        self.fusion_thread = None
        self.running = False
    
    def start(self):
        """Start all system threads."""
        logger.info("Starting system...")
        self.running = True
        
        # Create and start threads
        self.visual_thread = threading.Thread(
            target=self._run_visual,
            name="VisualPipeline",
            daemon=True
        )
        
        self.audio_thread = threading.Thread(
            target=self._run_audio,
            name="AudioPipeline",
            daemon=True
        )
        
        self.fusion_thread = threading.Thread(
            target=self._run_fusion,
            name="FusionManager",
            daemon=True
        )
        
        self.visual_thread.start()
        self.audio_thread.start()
        self.fusion_thread.start()
        
        logger.info("All threads started")
    
    def _run_visual(self):
        """Run visual pipeline in thread."""
        try:
            self.visual_pipeline.run(self.shared_state, self.running)
        except Exception as e:
            logger.error(f"Visual pipeline error: {e}", exc_info=True)
    
    def _run_audio(self):
        """Run audio pipeline in thread."""
        try:
            self.audio_pipeline.run(self.shared_state, self.running)
        except Exception as e:
            logger.error(f"Audio pipeline error: {e}", exc_info=True)
    
    def _run_fusion(self):
        """Run fusion manager in thread."""
        try:
            self.fusion_manager.run()
        except Exception as e:
            logger.error(f"Fusion manager error: {e}", exc_info=True)
    
    def stop(self):
        """Gracefully stop all threads."""
        logger.info("Stopping system...")
        self.running = False
        
        # Stop managers
        self.visual_pipeline.stop()
        self.audio_pipeline.stop()
        self.fusion_manager.stop()
        
        # Wait for threads
        self.visual_thread.join(timeout=5)
        self.audio_thread.join(timeout=5)
        self.fusion_thread.join(timeout=5)
        
        logger.info("System stopped")
```

## Next Steps

1. **Complete VisualPipeline class** - Refactor existing `yolo_detect.py`
2. **Implement AudioPipeline class** - Integrate VAD, SER, ASR, Intent
3. **Create main.py entry point** - Orchestrate all components
4. **Add configuration system** - YAML-based settings
5. **Implement testing framework** - Unit and integration tests
6. **Documentation** - API docs and user guide

This modular structure allows for clean separation of concerns and testability.