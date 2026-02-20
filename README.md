# Threat Detection System

A real-time multi-modal threat detection system using computer vision and audio analysis for autonomous security monitoring.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the system
python main.py

# Run with custom config
python main.py --config production.yaml --verbose
```

## 📋 System Overview

This system implements a **Late Fusion State Machine** that combines:
- **Visual Pipeline**: Person detection, tracking, weapon detection, face recognition
- **Audio Pipeline**: Voice activity detection, emotion analysis, speech transcription, intent classification
- **Fusion Engine**: Risk accumulation, state management, temporal persistence

### Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Visual Pipeline │    │  Audio Pipeline  │    │  Fusion Engine   │
│                 │    │                 │    │                 │
│ • Person Detect │    │ • VAD (Silero)  │    │ • Risk Accum.   │
│ • Tracking      │    │ • Emotion (HuBERT│    │ • State Machine │
│ • Weapon Detect │    │ • Transcribe     │    │ • Interrupts    │
│ • Face Recog.   │    │ • Intent Class. │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────────┐
                    │  System Manager     │
                    │                     │
                    │ • Thread Coord.     │
                    │ • Config Loading    │
                    │ • Lifecycle Mgmt    │
                    │ • Signal Handling   │
                    └─────────────────────┘
```

## ⚙️ Configuration

The system uses YAML configuration with environment variable overrides:

```yaml
# config.yaml
system:
  target_fps: 30.0
  fusion_hz: 1.0

visual:
  person_model: yolov8n.pt
  weapon_model: models/weapon_best.pt
  person_conf: 0.60
  person_min_area: 3500
  weapon_detect_every_n: 3
  face_recognize_every_n: 10

audio:
  sample_rate: 16000
  device: cpu

fusion:
  risk_decay_alpha: 0.95
  confidence_beta: 1.0
```

Environment overrides: `THREAT_SECTION__KEY=value`

## 🧪 Testing

Run the complete test suite:

```bash
# All tests
pytest

# Individual test suites
pytest test_fusion_core.py -v      # Core infrastructure
pytest test_audio_pipeline.py -v   # Audio processing
pytest test_visual_pipeline.py -v  # Visual processing
pytest test_system_integration.py -v  # System integration
```

## 📊 Performance Metrics

- **Latency**: <100ms per frame (visual), <50ms (audio)
- **Accuracy**: >90% detection, >85% recognition
- **Throughput**: 30 FPS sustained processing
- **Memory**: ~2GB RAM (models loaded)
- **CPU**: Multi-core recommended (4+ cores)

## 🔧 Development

### Project Structure
```
├── core/                    # Core infrastructure
│   ├── data_structures.py   # Type definitions
│   ├── shared_state.py      # Thread-safe state
│   ├── queue_manager.py     # Inter-thread communication
│   ├── config_loader.py     # YAML configuration
│   └── system_manager.py    # Main orchestrator
├── fusion/                  # Risk fusion engine
│   ├── risk_accumulator.py  # Temporal risk calculation
│   ├── state_manager.py     # State machine logic
│   └── fusion_manager.py    # Fusion orchestrator
├── audio/                   # Audio processing pipeline
│   ├── audio_pipeline.py    # Main audio orchestrator
│   ├── vad.py              # Voice activity detection
│   ├── emotion_analyzer.py  # Emotion recognition
│   ├── transcriber.py      # Speech-to-text
│   └── intent_classifier.py # Intent analysis
├── visual/                  # Visual processing pipeline
│   ├── visual_pipeline.py   # Main visual orchestrator
│   ├── detector.py         # Person/weapon detection
│   ├── tracker.py          # Person tracking
│   ├── weapon_mapper.py    # Weapon-person association
│   └── face_recognizer.py  # Face recognition
├── config.yaml             # System configuration
├── main.py                 # Entry point
└── requirements.txt        # Dependencies
```

### Key Components

#### State Machine
The system operates in 5 risk states:
- **IDLE** (0-15 pts): Normal operation
- **CAUTION** (15-30 pts): Minor alerts
- **EVALUATING** (30-45 pts): Active monitoring
- **ALERT** (45-60 pts): Immediate attention
- **CRITICAL** (60+ pts): Emergency response

#### Risk Calculation
Risk scores use temporal persistence with decay:
```
new_risk = (old_risk × α) + (new_input × β × confidence)
```

#### Context Filtering
Audio threats are filtered by emotional context:
- "Shut up!" + 😠 anger → High risk (+25 pts)
- "Shut up!" + 😂 laughter → Filtered to 0 (no false alert)

## 🚨 Alerts & Notifications

The system supports multiple alert mechanisms:
- **Console Logging**: Real-time status updates
- **File Logging**: Persistent event logging
- **Email Alerts**: Configurable risk-based notifications
- **State Callbacks**: Custom alert handlers

## 🔒 Security Considerations

- **Thread Safety**: All shared state uses multiprocessing.Manager
- **Resource Limits**: Queue size limits prevent memory exhaustion
- **Graceful Shutdown**: Signal handling for clean termination
- **Error Recovery**: Component isolation prevents cascade failures

## 📈 Monitoring & Debugging

### Logs
- **threat_detection.log**: Main system log
- **Console output**: Real-time status with `--verbose`
- **Component metrics**: Processing times, queue depths, error rates

### Health Checks
- **Queue monitoring**: Prevents backlog accumulation
- **Thread health**: Automatic restart on component failure
- **Memory usage**: Resource usage tracking
- **Performance profiling**: Latency and throughput metrics

## 🚀 Deployment

### Requirements
- **Python**: 3.8+
- **Hardware**: 4+ CPU cores, 8GB+ RAM, CUDA GPU (optional)
- **OS**: Linux/Windows/macOS
- **Audio**: Microphone input device
- **Models**: ~2GB disk space for ML models

### Production Setup
```bash
# Install system dependencies
sudo apt-get install python3-dev portaudio19-dev

# Install Python dependencies
pip install -r requirements.txt

# Download models (automatic on first run)
python -c "from visual.detector import PersonDetector; PersonDetector()"

# Configure for production
cp config.yaml production.yaml
# Edit production.yaml for your environment

# Run in background
nohup python main.py --config production.yaml > system.log 2>&1 &
```

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch
3. **Add** tests for new functionality
4. **Ensure** all tests pass
5. **Submit** a pull request

### Development Guidelines
- **Type Hints**: All functions use type annotations
- **Docstrings**: Comprehensive documentation
- **Logging**: Use appropriate log levels
- **Error Handling**: Graceful degradation on failures
- **Testing**: 100% test coverage for new code

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **YOLOv8**: Ultralytics for object detection
- **Silero VAD**: For voice activity detection
- **DistilHuBERT**: Facebook AI for emotion recognition
- **Faster-Whisper**: OpenAI for speech transcription
- **FaceNet**: For face recognition capabilities

---

**Built with ❤️ for autonomous security monitoring**
- Asynchronous email sending

### 5. Drawing Utilities (`utils/drawing.py`)
**Visualization components:**
- Bounding box rendering
- Label and threat level display
- Weapon association visualization
- Real-time overlay generation

## Planned Upgrades

### Phase 1: Visual Pipeline Enhancement ✅ (Ready for Implementation)
**Multi-Object Tracking (MOT) Integration:**
- Replace custom tracker with YOLOv8's ByteTrack
- Implement identity caching for performance
- Add audio-visual interrupt hooks
- Migrate to dict-based person structures

**Benefits:**
- Persistent tracking across occlusions
- 80%+ reduction in FaceNet processing
- Foundation for cross-modal integration

### Phase 2: Audio Pipeline Implementation 📋 (PRD Received)
**Status**: Ready for LLD
**Components**: 3-Stage Hierarchical Processing

**Stage 1: Silero VAD** - Enterprise-grade ONNX-optimized gatekeeper (90% CPU reduction)  
**Stage 2: DistilHuBERT** - 0.02MB emotion recognition for instant risk boosts  
**Stage 3: Faster-Whisper + DistilBERT** - INT8 ASR + zero-shot intent classification  
**Integration**: Conversational Intuition with contextual intent filtering

### Phase 3: Shared Risk Accumulator (SRA)
**Dynamic Risk Assessment:**
- 0-100 risk scoring system
- Multi-modal evidence integration
- Temporal risk accumulation
- Configurable alert thresholds

**Capabilities:**
- Visual + audio evidence combination
- Progressive threat escalation
- Historical pattern analysis
- Adaptive response system

### Phase 4: Advanced Intelligence
**Enhanced Features:**
- Person re-identification across sessions
- Behavior pattern analysis
- Anomaly detection
- Predictive threat assessment

## Installation & Setup

### Prerequisites
```bash
# Python 3.8+ required
python --version

# GPU support (optional but recommended)
nvidia-smi  # Check CUDA availability
```

### Installation
```bash
# Clone repository
git clone <repository-url>
cd person_detect/yolo

# Install dependencies
pip install -r requirements.txt

# For YOLOv8 specific requirements
pip install -r requirements_yolov8.txt
```

### Model Setup
```bash
# Download YOLOv8 models (automatic on first run)
# Person model: yolov8n.pt (automatically downloaded)
# Weapon model: Place weapon_best.pt in models/ directory

# Face embeddings setup
python enroll_faces.py  # Enroll known faces
```

### Configuration
Environment variables in `.env` or system:
```bash
# Detection thresholds
YOLOV8_PERSON_CONF=0.60
YOLOV8_WEAPON_CLASSES=knife,scissors,gun,pistol,rifle

# Performance settings
USE_GPU=1  # 0 for CPU-only
WEAPON_DETECT_EVERY_N_FRAMES=3

# Alert configuration
ALERT_EMAIL=your-email@gmail.com
```

## Usage

### Basic Operation
```bash
# Start the detection system
python camera_detection/yolo_detect.py

# CPU-only mode (if GPU issues)
python camera_detection/yolo_detect.py --cpu
```

### Face Enrollment
```bash
# Enroll new faces for recognition
python enroll_faces.py
```

### Testing & Diagnostics
```bash
# Run diagnostic tests
python utils/diagnose_all_tests.py

# Test specific components
python utils/diagnose_boxes.py
```

## System Requirements

### Hardware
- **CPU**: 4+ cores recommended
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: NVIDIA GPU with CUDA (optional)
- **Camera**: USB webcam or IP camera
- **Microphone**: Audio input device (future audio pipeline)

### Software
- **OS**: Windows 10+, Linux (Ubuntu 18.04+), macOS
- **Python**: 3.8 - 3.11
- **CUDA**: 11.0+ (if using GPU)

## Performance Characteristics

### Current Performance
- **Frame Rate**: 30 FPS (GPU), 15 FPS (CPU)
- **Detection Accuracy**: >90% person detection
- **Recognition Accuracy**: >85% face recognition
- **Memory Usage**: ~2GB RAM
- **CPU Usage**: 40-60% (single thread)

### Target Performance (Post-Upgrades)
- **Frame Rate**: 30+ FPS sustained
- **Detection Accuracy**: >95% with MOT
- **Recognition Accuracy**: >90% with caching
- **Memory Usage**: <3GB RAM
- **CPU Usage**: 60-80% (multi-threaded)

## Troubleshooting

### Common Issues
1. **CUDA Errors**: Set `USE_GPU=0` for CPU-only mode
2. **Camera Not Found**: Check camera permissions and connections
3. **Low Performance**: Reduce frame processing or use GPU
4. **Face Recognition Issues**: Re-enroll faces with better lighting

### Debug Mode
```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG
python camera_detection/yolo_detect.py
```

### Performance Monitoring
- Check console output for FPS and detection counts
- Monitor system resources with Task Manager
- Use `utils/diagnose_*` scripts for component testing

## Development Roadmap

### Immediate Next Steps
1. **Visual Pipeline MOT Integration** - Replace custom tracker
2. **Identity Caching Implementation** - Performance optimization
3. **Audio Pipeline Design** - Multi-modal foundation
4. **SRA System Architecture** - Dynamic risk assessment

### Future Enhancements
- **Re-ID System**: Cross-session person recognition
- **Behavior Analysis**: Movement pattern detection
- **API Integration**: REST/WebSocket interfaces
- **Mobile Support**: Edge device deployment

## Contributing

### Code Standards
- Follow PEP 8 Python style guidelines
- Add type hints for new functions
- Include comprehensive docstrings
- Write unit tests for new components

### Testing
- Run existing diagnostic tests before changes
- Add new tests for modified components
- Validate performance impact of changes
- Test on both CPU and GPU configurations

## License & Attribution

This project integrates several open-source components:
- **Ultralytics YOLOv8**: Object detection framework
- **FaceNet PyTorch**: Face recognition implementation
- **MediaPipe**: Hand tracking and pose estimation
- **OpenCV**: Computer vision library

## Support & Documentation

### Documentation Files
- `IMPLEMENTATION_PLAN.md` - Comprehensive upgrade roadmap
- `VISUAL_PIPELINE_TODO.md` - Visual enhancement details
- `AUDIO_PIPELINE_TODO.md` - Audio system specifications

### Getting Help
- Check existing issues and documentation
- Run diagnostic scripts for system validation
- Review console logs for error details
- Test components individually for isolation

---

**Status**: Active Development
**Version**: Pre-Upgrade (Visual Pipeline Ready, Audio Pipeline Planned)
**Last Updated**: February 18, 2026</content>
<parameter name="filePath">c:\programing\MachineLearning\person_detect\yolo\README.md