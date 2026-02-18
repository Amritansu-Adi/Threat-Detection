# Low-Level Design (LLD) - Late Fusion State Machine Architecture

## Overview

The Late Fusion State Machine is a modular, multi-threaded architecture that separates the Visual Pipeline, Audio Pipeline, and a central Fusion Manager. Instead of a monolithic script, we implement three distinct components that communicate via thread-safe queues, processing at their own speeds and coordinating through a central state machine.

## Core Architectural Philosophy

**Late Fusion**: Don't combine visual and audio data immediately. Instead:
1. Process each modality independently and at its natural frequency
2. Send processed results (not raw data) to a fusion manager
3. Fuse results into a "Behavioral Narrative" that informs global risk
4. Send back control signals (interrupts) to force re-prioritization

**Why This Works**:
- Visual @ ~20 FPS (50ms per frame)
- Audio @ ~2-second chunks (acoustic chunking)
- Fusion @ 1 Hz (decisions every 1 second)
- Each thread operates independently without blocking

## 4.1 Core Class Structure

### Four Primary Components

#### 1. VisualPipeline (Existing, Enhanced)
```python
class VisualPipeline:
    """Real-time person detection, tracking, and weapon identification."""
    
    def __init__(self, visual_to_fusion_q: Queue, fusion_to_visual_q: Queue):
        self.model = YOLO(...)
        self.tracker = PersonTracker()
        self.visual_to_fusion_q = visual_to_fusion_q  # Send results
        self.fusion_to_visual_q = fusion_to_visual_q  # Receive interrupts
        self.priority_zones = None  # Updated by fusion manager
        
    def process_frame(self, frame):
        # Detect and track persons
        # Check for priority zones from fusion manager
        # Send results to fusion queue
        visual_data = {
            'timestamp': time.time(),
            'frame_idx': self.frame_idx,
            'detections': [...],
            'tracked_persons': [...],
            'weapons': [...]
        }
        self.visual_to_fusion_q.put(visual_data)
        
    def receive_interrupt(self):
        # Non-blocking check for interrupt from fusion
        try:
            interrupt = self.fusion_to_visual_q.get_nowait()
            if interrupt['command'] == 'FOCUS_RE_INFERENCE':
                self.priority_zones = interrupt['zones']
        except Empty:
            pass
```

#### 2. AudioPipeline (New, Standalone Thread)
```python
class AudioPipeline:
    """Real-time audio processing with VAD, SER, and intent analysis."""
    
    def __init__(self, audio_to_fusion_q: Queue):
        self.vad = SileroVAD()
        self.ser = DistilHuBERT()  # Emotion recognition
        self.asr = FasterWhisper()
        self.intent_classifier = DistilBERT()
        self.audio_to_fusion_q = audio_to_fusion_q  # Send results
        self.audio_buffer = collections.deque(maxlen=160000)  # 5 seconds @ 16kHz
        
    def process_audio_chunk(self, audio_chunk):
        # Step 1: VAD - Check if speech detected
        if not self.vad.is_speech(audio_chunk):
            return
            
        # Step 2: SER - Analyze emotion from prosody
        emotion, emotion_conf = self.ser.analyze(audio_chunk)
        
        # Step 3: ASR - Transcribe speech
        transcript = self.asr.transcribe(audio_chunk)
        
        # Step 4: Intent - Classify intent
        intent, intent_conf = self.intent_classifier.classify(
            transcript, 
            labels=['distress', 'threat', 'neutral']
        )
        
        # Send to fusion manager
        audio_data = {
            'timestamp': time.time(),
            'transcript': transcript,
            'emotion': emotion,
            'emotion_confidence': emotion_conf,
            'intent': intent,
            'intent_confidence': intent_conf,
            'audio_segment': audio_chunk  # Raw audio for lookback
        }
        self.audio_to_fusion_q.put(audio_data)
```

#### 3. FusionManager (The Brain)
```python
class FusionManager:
    """Central state machine managing late fusion and behavioral narrative."""
    
    def __init__(self, visual_q: Queue, audio_q: Queue, 
                 fusion_to_visual_q: Queue):
        self.visual_q = visual_q
        self.audio_q = audio_q
        self.fusion_to_visual_q = fusion_to_visual_q  # Send interrupts
        
        # Shared state (thread-safe dictionary)
        self.manager = multiprocessing.Manager()
        self.shared_state = self.manager.dict()
        self.shared_state['risk_score'] = 0
        self.shared_state['state'] = 'IDLE'
        self.shared_state['last_update'] = time.time()
        
        # Event buffers
        self.visual_buffer = None
        self.audio_buffer = None
        
        # State machine
        self.state = 'IDLE'
        self.state_start_time = time.time()
        
    def run(self):
        """Main fusion loop - runs at ~1 Hz."""
        while True:
            # Collect latest events (non-blocking)
            self._collect_events()
            
            # Update risk score
            risk_score = self._calculate_risk()
            
            # State machine transition
            self._update_state(risk_score)
            
            # Send interrupts if needed
            self._send_interrupts()
            
            # Sleep to run at ~1 Hz
            time.sleep(1.0)
    
    def _collect_events(self):
        """Collect latest events from both pipelines."""
        try:
            self.visual_buffer = self.visual_q.get_nowait()
        except Empty:
            pass
            
        try:
            self.audio_buffer = self.audio_q.get_nowait()
        except Empty:
            pass
```

#### 4. RiskAccumulator (Shared State)
```python
class RiskAccumulator:
    """Implements the Temporal Persistence Multiplier for behavioral narrative."""
    
    def __init__(self, shared_state: dict):
        self.shared_state = shared_state
        self.alpha = 0.9  # Retention factor (decay)
        self.beta = 1.0   # Confidence factor (scaling)
        
        # Risk thresholds
        self.THRESHOLD_ALERT = 75
        self.THRESHOLD_EVALUATING = 50
        self.THRESHOLD_CAUTION = 25
        
    def calculate_risk(self, visual_data, audio_data, time_delta):
        """
        Risk Decay/Accumulation Formula:
        Risk_t = (Risk_{t-1} × α) + (SensoryInput × β)
        
        α (Retention Factor): 0.9 - allows risk to stay high for ~5 seconds
        β (Confidence Factor): Based on model confidence scores
        """
        previous_risk = self.shared_state.get('risk_score', 0)
        
        # Decay component
        decay_component = previous_risk * (self.alpha ** time_delta)
        
        # Sensory input component
        sensory_component = 0
        
        if visual_data:
            sensory_component += self._visual_risk(visual_data)
            
        if audio_data:
            sensory_component += self._audio_risk(audio_data)
        
        # Apply confidence scaling
        sensory_component *= self.beta
        
        # Calculate new risk
        new_risk = decay_component + sensory_component
        
        # Clamp to 0-100
        new_risk = max(0, min(100, new_risk))
        
        return new_risk
    
    def _visual_risk(self, visual_data):
        """Extract risk from visual detections."""
        risk = 0
        
        # Unknown person = +20 pts
        if any(p['label'] == 'Unknown' for p in visual_data.get('tracked_persons', [])):
            risk += 20
        
        # Weapon detected = +40 pts
        if visual_data.get('weapons'):
            risk += 40
        
        # Unknown person with weapon = +60 pts (not 60, based on combo)
        if any(p['has_weapon'] and p['label'] == 'Unknown' 
               for p in visual_data.get('tracked_persons', [])):
            risk += 40  # Additional to the above
        
        return risk
    
    def _audio_risk(self, audio_data):
        """Extract risk from audio analysis."""
        risk = 0
        
        # Emotion-based immediate boost
        emotion = audio_data.get('emotion', '').lower()
        emotion_conf = audio_data.get('emotion_confidence', 0)
        
        if emotion == 'anger' and emotion_conf > 0.85:
            risk += 25  # Immediate boost
        elif emotion == 'fear' and emotion_conf > 0.85:
            risk += 30  # Immediate boost
        
        # Intent-based scoring
        intent = audio_data.get('intent', '').lower()
        intent_conf = audio_data.get('intent_confidence', 0)
        
        if intent == 'threat' and intent_conf > 0.8:
            risk += 45
        elif intent == 'distress' and intent_conf > 0.8:
            risk += 35
        
        return risk
```

## 4.2 Communication: Thread-Safe Queues

### Queue Specification

```python
from multiprocessing import Queue, Manager

# Main communication channels
audio_to_fusion_q = Queue(maxsize=10)
visual_to_fusion_q = Queue(maxsize=10)
fusion_to_visual_q = Queue(maxsize=10)

# Shared state (for instant reads)
manager = Manager()
shared_state = manager.dict()

# Queue message format specs
VISUAL_MESSAGE = {
    'timestamp': float,           # time.time()
    'frame_idx': int,
    'detections': [],             # [x1, y1, x2, y2, conf, class]
    'tracked_persons': [],        # List of tracked person dicts
    'weapons': [],                # Weapon detections
    'priority_zones': []          # Zones to focus on (if set by fusion)
}

AUDIO_MESSAGE = {
    'timestamp': float,
    'transcript': str,
    'emotion': str,               # 'anger', 'fear', 'happy', 'neutral'
    'emotion_confidence': float,  # 0-1
    'intent': str,                # 'distress', 'threat', 'neutral'
    'intent_confidence': float,   # 0-1
    'audio_segment': np.ndarray   # Raw audio bytes for lookback
}

FUSION_INTERRUPT = {
    'command': str,               # 'FOCUS_RE_INFERENCE', 'HEIGHTEN_SENSITIVITY'
    'zones': [],                  # Regions to focus on
    'priority': bool,             # Force priority processing
    'timestamp': float
}
```

### Queue Flow Diagram

```
Visual Thread          Audio Thread
    |                      |
    |--→ visual_to_fusion_q--→|
    |                         |
    |←--fusion_to_visual_q←---|
    |                         |
    └────→ FusionManager ←───┘
                 |
          shared_state dict
          (risk_score readable
           by all threads)
```

## 4.3 The Behavioral Narrative Math

### Temporal Persistence Multiplier

**Formula**:
$$Risk_t = (Risk_{t-1} \times \alpha) + (SensoryInput \times \beta)$$

Where:
- **Risk_t**: Current risk score (0-100)
- **Risk_{t-1}**: Previous risk score
- **α (Alpha)**: Retention factor = 0.9
  - Allows risk to persist for ~5 seconds
  - Decay rate: 5 pts/sec after 50 pts threshold
  - Formula: Decay = Risk × (0.9^dt) where dt = time delta in seconds
- **β (Beta)**: Confidence factor
  - Scales input by model confidence
  - Example: 0.7 confidence emotion = 0.7× the risk contribution

### Example Scenario Logic: The "Shut Up" Problem

**Scenario**: Someone says "Shut up!" with annoyed tone

**Timeline**:
```
T=0s: "Shut up!"
  - Transcript detected: "Shut up"
  - Emotion: Anger at 0.8 confidence
  - Intent: Threat at 0.7 confidence
  - Aggression Score: 25 (emotion) + 35 (intent*0.7) = 60 points
  - State: EVALUATING (threshold 50-75)

T=1s: Laughter heard in background (VAD detects speech gap)
  - Emotion: Happy at 0.85 confidence
  - Intent: Neutral
  - Sensory Input: 0 points (happy = benign)
  - Risk Calculation:
    - Decay: 60 × 0.9^1.0 = 54 points
    - New Input: 0
    - New Risk: 54 points (still EVALUATING)

T=2s: Continued laughter
  - Decay: 54 × 0.9^1.0 = 48.6 points
  - New Input: 0
  - New Risk: 48.6 points (dropped to CAUTION)

T=3s: Silence
  - Decay: 48.6 × 0.9^1.0 = 43.7 points
  - New Risk: 43.7 points (continues declining)
  
Result: No alert sent because risk never crossed 75-point threshold.
System correctly identified context (joking, not serious threat).
```

## 4.4 State Machine Design

```python
class StateManager:
    """Manages system states based on risk score."""
    
    STATES = {
        'IDLE': {'min': 0, 'max': 25, 'description': 'Normal operation'},
        'CAUTION': {'min': 25, 'max': 50, 'description': 'Minor threat detected'},
        'EVALUATING': {'min': 50, 'max': 75, 'description': 'Context analysis underway'},
        'ALERT': {'min': 75, 'max': 100, 'description': 'High threat confirmed'},
        'CRITICAL': {'min': 90, 'max': 100, 'description': 'Immediate action needed'}
    }
    
    def __init__(self):
        self.current_state = 'IDLE'
        self.state_entry_time = time.time()
        self.state_duration_threshold = 3  # seconds
        
    def transition(self, risk_score):
        """Determine state based on risk score."""
        for state, bounds in self.STATES.items():
            if bounds['min'] <= risk_score <= bounds['max']:
                if state != self.current_state:
                    self._on_state_change(state)
                    self.current_state = state
                    self.state_entry_time = time.time()
                return state
        
        return self.current_state
    
    def _on_state_change(self, new_state):
        """Execute state change callbacks."""
        if new_state == 'ALERT':
            self._send_alert()
        elif new_state == 'IDLE':
            self._clear_alert()

    def should_send_notification(self):
        """Send notification only if state duration exceeded."""
        duration = time.time() - self.state_entry_time
        return self.current_state in ['ALERT', 'CRITICAL'] and \
               duration >= self.state_duration_threshold
```

## 4.5 Implementation Steps and Research Targets

### Step 1: Observer Pattern Implementation
**Research**: "Observer Design Pattern in Python"

The FusionManager will "watch" (observe) queues from both visual and audio threads.

```python
class Observable:
    """Base class for observable objects."""
    
    def __init__(self):
        self._observers = []
        
    def attach(self, observer):
        """Attach an observer."""
        if observer not in self._observers:
            self._observers.append(observer)
            
    def notify(self, event):
        """Notify all observers of an event."""
        for observer in self._observers:
            observer.update(event)

class FusionManagerObserver:
    """Observer for queue events."""
    
    def update(self, event):
        """Called when observable sends update."""
        if event['type'] == 'visual':
            self._handle_visual(event)
        elif event['type'] == 'audio':
            self._handle_audio(event)
```

### Step 2: Shared Memory vs. Queues
**Research**: "multiprocessing.Manager().dict() in Python"

Use shared dictionary for instant state reads:

```python
from multiprocessing import Manager

manager = Manager()
shared_state = manager.dict()

# Accessible from all threads
shared_state['risk_score'] = 65
shared_state['state'] = 'EVALUATING'
shared_state['last_visual_update'] = time.time()

# Thread-safe reads/writes
if shared_state['risk_score'] > 50:
    # Visual thread can read this instantly
    print("In evaluating state")
```

### Step 3: Signal Padding in Audio Pipeline
**Research**: "VAD padding and speech boundary detection"

Ensure complete utterance capture with silence padding:

```python
class AudioProcessor:
    """Process audio with proper VAD padding."""
    
    def __init__(self):
        self.vad = SileroVAD()
        self.padding_frames = 3  # Additional frames after speech ends
        self.speech_buffer = []
        
    def process_stream(self, audio_chunk):
        """Process audio with padding."""
        is_speech = self.vad.is_speech(audio_chunk)
        
        if is_speech:
            self.speech_buffer.append(audio_chunk)
        elif self.speech_buffer:
            # Speech ended, add padding frames
            for _ in range(self.padding_frames):
                self.speech_buffer.append(audio_chunk)
            
            # Process complete utterance
            complete_utterance = np.concatenate(self.speech_buffer)
            self._transcribe_and_analyze(complete_utterance)
            self.speech_buffer = []
```

## 4.6 Thread Management and Lifecycle

```python
class SystemManager:
    """Manages all threads and system lifecycle."""
    
    def __init__(self):
        self.visual_thread = None
        self.audio_thread = None
        self.fusion_thread = None
        self.running = False
        
    def start(self):
        """Start all system threads."""
        # Initialize queues
        self.audio_to_fusion_q = Queue()
        self.visual_to_fusion_q = Queue()
        self.fusion_to_visual_q = Queue()
        
        # Initialize shared state
        manager = multiprocessing.Manager()
        self.shared_state = manager.dict()
        self.shared_state['risk_score'] = 0
        self.shared_state['state'] = 'IDLE'
        
        # Start threads
        self.visual_thread = threading.Thread(
            target=self._visual_worker,
            daemon=True
        )
        self.audio_thread = threading.Thread(
            target=self._audio_worker,
            daemon=True
        )
        self.fusion_thread = threading.Thread(
            target=self._fusion_worker,
            daemon=True
        )
        
        self.running = True
        self.visual_thread.start()
        self.audio_thread.start()
        self.fusion_thread.start()
        
    def stop(self):
        """Gracefully stop all threads."""
        self.running = False
        self.visual_thread.join(timeout=5)
        self.audio_thread.join(timeout=5)
        self.fusion_thread.join(timeout=5)
```

## 4.7 Data Structures and Type Definitions

```python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class TrackedPerson:
    """A person being tracked across frames."""
    track_id: int
    bbox: tuple  # (x1, y1, x2, y2)
    identity: str  # 'Unknown', person name
    confidence: float
    has_weapon: bool = False
    frames_in_view: int = 0

@dataclass
class WeaponDetection:
    """Detected weapon with location and confidence."""
    bbox: tuple  # (x1, y1, x2, y2)
    weapon_type: str  # 'knife', 'gun', etc.
    confidence: float
    associated_person: Optional[int] = None

@dataclass
class AudioEvent:
    """Processed audio event with all modalities."""
    timestamp: float
    transcript: str
    emotion: str  # 'anger', 'fear', 'happy', 'neutral'
    emotion_confidence: float
    intent: str  # 'distress', 'threat', 'neutral'
    intent_confidence: float
    
@dataclass
class RiskScoreEvent:
    """Risk score update event."""
    timestamp: float
    score: float  # 0-100
    state: str  # 'IDLE', 'CAUTION', 'EVALUATING', 'ALERT'
    contributing_factors: Dict[str, float]
```

## Summary of Implementation Path

1. **Phase 1**: Core infrastructure (queues, threading, manager)
2. **Phase 2**: Risk accumulation logic with temporal multiplier
3. **Phase 3**: State machine and interrupt handling
4. **Phase 4**: Integration with visual and audio pipelines
5. **Phase 5**: Testing, optimization, and validation

**Ready for implementation when all three components (Visual, Audio, Fusion) are complete.**