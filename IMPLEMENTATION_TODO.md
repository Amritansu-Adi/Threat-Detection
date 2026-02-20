# Implementation TODO - Late Fusion State Machine

## Status
**Branch**: `feature/late-fusion-state-machine`
**Date**: February 18, 2026
**Phase**: Low-Level Design (LLD) Complete → Ready for Implementation

## Workspace Structure

Already Documented:
- ✅ IMPLEMENTATION_PLAN.md - High-level roadmap
- ✅ VISUAL_PIPELINE_TODO.md - Visual upgrades (MOT, identity caching)
- ✅ AUDIO_PIPELINE_TODO.md - Audio PRD with 3-stage hierarchy
- ✅ AUDIO_RESEARCH_GUIDE.md - Specific model implementations
- ✅ LLD_LATE_FUSION_STATE_MACHINE.md - Architecture & class design
- ✅ IMPLEMENTATION_GUIDE.md - Step-by-step code walkthrough

## Phase 1: Core Infrastructure (Week 1)

### 1.1 Directory Structure
- [ ] Create `audio/` directory with `__init__.py`
- [ ] Create `fusion/` directory with `__init__.py`
- [ ] Create `core/` directory with `__init__.py`
- [ ] Update imports in existing files

### 1.2 Core Data Structures
- [ ] Implement `core/data_structures.py`
  - [ ] TrackedPerson dataclass
  - [ ] WeaponDetection dataclass
  - [ ] AudioEvent dataclass
  - [ ] VisualData dataclass
  - [ ] RiskEvent dataclass
  - [ ] FusionInterrupt dataclass
- [ ] Verify all type hints are correct
- [ ] Add docstrings to all classes

### 1.3 Shared State Management
- [ ] Implement `core/shared_state.py`
  - [ ] SharedState class wrapper
  - [ ] Thread-safe dict initialization
  - [ ] State getter/setter methods
- [ ] Test thread-safe access
- [ ] Add logging for state changes

### 1.4 Queue System Setup
- [ ] Create queue initialization in `core/queue_manager.py`
  - [ ] audio_to_fusion_q
  - [ ] visual_to_fusion_q
  - [ ] fusion_to_visual_q
- [ ] Define message schemas
- [ ] Add queue monitoring utilities

## Phase 2: Fusion Core (Week 2)

### 2.1 Risk Accumulation Logic
- [ ] Implement `fusion/risk_accumulator.py`
  - [ ] Temporal Persistence Multiplier formula
  - [ ] Visual risk calculation
  - [ ] Audio risk calculation
  - [ ] Decay logic (alpha parameter)
  - [ ] Confidence scaling (beta parameter)
- [ ] Validate formula with test scenarios
- [ ] Add logging for risk calculations
- [ ] Test "Shut up" scenario (decay/accumulation)

### 2.2 State Machine
- [ ] Implement `fusion/state_manager.py`
  - [ ] State enumeration
  - [ ] State boundaries definition
  - [ ] State transition logic
  - [ ] Duration-based alerts
  - [ ] State change callbacks
- [ ] Define all state-specific behaviors
- [ ] Test all state transitions

### 2.3 Fusion Manager
- [ ] Implement `fusion/fusion_manager.py`
  - [ ] Queue collection mechanism
  - [ ] Risk calculation loop
  - [ ] State update logic
  - [ ] Interrupt generation
  - [ ] Main run() method
- [ ] Implement non-blocking queue reads
- [ ] Add comprehensive logging
- [ ] Test with mock visual/audio data

### 2.4 Interrupt System
- [ ] Implement `fusion/interrupts.py`
  - [ ] FusionInterrupt dataclass (already in core/data_structures.py)
  - [ ] Interrupt priority system
  - [ ] Zone calculation logic
  - [ ] Priority zone generation

## Phase 3: Audio Pipeline (Week 3)

### 3.1 Audio Buffer & Capture
- [ ] Implement `audio/audio_buffer.py`
  - [ ] collections.deque sliding window (5 seconds)
  - [ ] Audio chunking for processing
  - [ ] VAD padding logic
  - [ ] Buffer management

- [ ] Implement `audio/audio_capture.py`
  - [ ] PyAudio integration
  - [ ] 16kHz mono capture
  - [ ] Error handling
  - [ ] Device enumeration

### 3.2 Stage 1: Silero VAD
- [ ] Implement `audio/vad_processor.py`
  - [ ] ONNX model loading
  - [ ] Speech detection (>300ms threshold)
  - [ ] Frame processing
  - [ ] Gate keeper logic

### 3.3 Stage 2: Emotion Recognition
- [ ] Implement `audio/emotion_analyzer.py`
  - [ ] DistilHuBERT model loading
  - [ ] Prosody analysis
  - [ ] Emotion classification
  - [ ] Confidence thresholding (>85%)
  - [ ] Immediate risk boost triggers

### 3.4 Stage 3: ASR & Intent
- [ ] Implement `audio/intent_classifier.py`
  - [ ] Faster-Whisper integration (INT8, beam_size=1)
  - [ ] Speech transcription
  - [ ] DistilBERT zero-shot classification
  - [ ] Intent categorization [distress, threat, neutral]
  - [ ] Context filtering logic

### 3.5 AudioPipeline Class
- [ ] Implement `audio/audio_pipeline.py`
  - [ ] Main audio processing thread
  - [ ] Queue integration
  - [ ] Shared state updates
  - [ ] Error handling
  - [ ] Graceful shutdown

## Phase 4: Visual Pipeline Refactoring (Week 4)

### 4.1 VisualPipeline Class
- [ ] Refactor existing `camera_detection/yolo_detect.py`
  - [ ] Extract into VisualPipeline class
  - [ ] Queue integration for results
  - [ ] Interrupt listening mechanism
  - [ ] Priority zone processing
  - [ ] Thread-safe operation

- [ ] Create `camera_detection/visual_pipeline.py`
  - [ ] VisualPipeline class definition
  - [ ] Person detection and tracking
  - [ ] Weapon detection integration
  - [ ] Face recognition with caching
  - [ ] Hand tracking for associations

### 4.2 MOT Integration (Pre-existing work)
- [ ] Implement YOLOv8 ByteTrack
- [ ] Update person tracking
- [ ] Add identity caching
- [ ] Integrate weapon detection

### 4.3 Visual Data Publishing
- [ ] VisualData packaging
- [ ] Queue publishing logic
- [ ] Frame skipping for performance
- [ ] Synchronized timestamp

## Phase 5: System Integration (Week 5) ✅ COMPLETED

### 5.1 System Manager
- [x] Implement `core/system_manager.py`
  - [x] Thread coordination
  - [x] Lifecycle management
  - [x] Graceful startup/shutdown
  - [x] Error recovery
  - [x] Thread monitoring

### 5.2 Configuration System
- [x] Create `config.yaml` with all parameters
- [x] Implement `core/config_loader.py`
  - [x] YAML parsing
  - [x] Environment variable overrides
  - [x] Validation
  - [x] Logging of loaded config

### 5.3 Main Entry Point
- [x] Create `main.py`
  - [x] System initialization
  - [x] Thread startup
  - [x] Signal handling (Ctrl+C)
  - [x] Logging configuration
  - [x] Error reporting

### 5.4 Alert System Integration
- [ ] Update `utils/email_alert.py`
  - [ ] Risk score-based alerting
  - [ ] State-aware notifications
  - [ ] Recipient configuration
  - [ ] Alert throttling

## Phase 6: Testing & Validation (Week 6)

### 6.1 Unit Tests
- [ ] Test risk accumulation formula
  - [ ] Visual risk calculation
  - [ ] Audio risk calculation
  - [ ] Decay logic
  - [ ] Confidence scaling
  
- [ ] Test state transitions
  - [ ] Boundary conditions
  - [ ] State durations
  - [ ] State callbacks

- [ ] Test audio processors
  - [ ] VAD detection
  - [ ] Emotion recognition
  - [ ] Intent classification

### 6.2 Integration Tests
- [ ] Test queue communication
- [ ] Test visual → fusion → visual interrupts
- [ ] Test audio → fusion path
- [ ] Test concurrent operation

### 6.3 Scenario Tests
- [ ] Implement "Shut up" scenario test
  - [ ] Risk rises with anger detection
  - [ ] Risk decays when happy tone follows
  - [ ] No alert sent (stays below threshold)
  
- [ ] Implement weapon detection scenario
  - [ ] Unknown person + weapon = high risk
  - [ ] Visual detection triggers audio focus
  
- [ ] Implement distress scenario
  - [ ] Audio distress intent = risk boost
  - [ ] Combined with visual unknowns = alert

### 6.4 Performance Testing
- [ ] Measure CPU usage per thread
- [ ] Measure latency (visual → fusion → interrupt)
- [ ] Measure memory usage
- [ ] Test sustained operation (24-hour)

### 6.5 Edge Case Testing
- [ ] No audio input
- [ ] No video input
- [ ] Network issues
- [ ] Model loading failures
- [ ] Queue overflow scenarios

## Phase 7: Documentation & Deployment (Week 7)

### 7.1 Code Documentation
- [ ] Docstrings for all classes/methods
- [ ] Type hint explanations
- [ ] Architecture diagrams
- [ ] Data flow diagrams

### 7.2 User Documentation
- [ ] Installation guide
- [ ] Configuration guide
- [ ] Usage examples
- [ ] Troubleshooting guide

### 7.3 API Documentation
- [ ] Queue message format specs
- [ ] Shared state keys
- [ ] Configuration parameters
- [ ] Interrupt commands

### 7.4 Deployment Preparation
- [ ] Docker containerization
- [ ] Requirements.txt updates
- [ ] CI/CD pipeline setup
- [ ] Version tagging

## Critical Implementation Notes

### Observer Pattern Research
**What To Learn**:
- Observer/Observable pattern in Python
- Event-driven architecture
- Subject-observer decoupling

**Why It Matters**:
- FusionManager observes queue events
- Clean separation between components
- Easy to add new observers later

### Multiprocessing.Manager() Research
**What To Learn**:
- Manager.dict() for shared state
- Thread-safe operations
- Performance implications
- Memory management

**Why It Matters**:
- All threads can read risk_score instantly
- No need for locks/semaphores
- Enables reactive behavior across threads

### VAD Padding Research
**What To Learn**:
- Voice Activity Detection padding
- Speech boundary detection
- Silence handling
- Utterance completeness

**Why It Matters**:
- Ensures complete utterances are transcribed
- Prevents cutting off important audio
- Allows proper emotion analysis on complete clips

## Success Criteria

### Functional Requirements
- [ ] All three pipelines (visual, audio, fusion) run concurrently
- [ ] Risk score updates every ~1 second
- [ ] Fusion interrupts trigger visual re-processing
- [ ] State machine transitions work correctly
- [ ] Email alerts sent at correct thresholds

### Performance Requirements
- [ ] Visual: 30 FPS sustained
- [ ] Audio: <100ms latency per chunk
- [ ] Fusion: 1 Hz decision rate
- [ ] Total CPU: <80% on 4-core system
- [ ] Total Memory: <1GB

### Quality Requirements
- [ ] 100% code coverage for core logic
- [ ] All edge cases tested
- [ ] Zero crashes/hangs in 24-hour test
- [ ] Clean shutdown without warnings
- [ ] All logs properly formatted

## Dependency Installation

Before implementation:
```bash
pip install -r requirements.txt  # Already updated with audio libs
pip install pytest pytest-cov pytest-mock  # For testing
pip install python-dotenv  # For environment config
```

## Next Step: Start Implementation

Once all TODO items in Phase 1 are complete, begin Phase 2 with focus on:
1. Risk accumulation logic correctness
2. Decay formula validation
3. State machine transitions
4. Comprehensive logging

All components should log their operations for debugging and monitoring.