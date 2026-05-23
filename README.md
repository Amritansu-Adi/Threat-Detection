# Threat Detection System (Realtime + Demo Replay)

Multimodal realtime threat-detection app that combines:
- Visual signals (person detection, identity, weapon detection, weapon-person association)
- Audio signals (VAD, transcript, emotion/intent, distress/anomaly cues)
- Fusion risk scoring and state machine

This project has two run modes:
- **Live mode**: realtime camera + microphone
- **Demo mode**: prerecorded TEST videos fed through the **same realtime pipeline**

Demo is not a different logic path; it is a replay harness using the same visual/audio/fusion/risk engine.

---

## 1) What the app does

- Detects people in video frames
- Recognizes known identity vs unknown (FaceNet embeddings)
- Detects weapon-like objects
- Associates weapons to persons (hand-aware mapping + spatial heuristics)
- Processes audio segments for transcript, emotion/intent, and acoustic anomalies
- Computes a risk score from multiple factors using weighted sum + temporal decay
- Maps score to threat states:
  - `IDLE`
  - `CAUTION`
  - `EVALUATING`
  - `ALERT`
  - `CRITICAL`
- Renders realtime overlay UI with detections, identities, risk, state, and audio factors

---

## 2) Current design principles

- Realtime-first architecture
- Shared pipeline between live and demo
- Risk depends on **multiple factors**, not one signal:
  - visual evidence
  - emotion
  - intent
  - distress/anomaly audio
  - cross-modal confirmation bonuses
- Known homeowner with tool/weapon alone is lower baseline risk; audio threat/distress evidence increases escalation.

---

## 3) Project structure

- `main.py` - live entrypoint
- `core/system_manager.py` - lifecycle orchestration for visual/audio/fusion/capture/display
- `visual/` - person/weapon detection, tracking, face recognition, association
- `audio/` - VAD, transcription, emotion, intent pipeline
- `fusion/` - risk accumulator, state manager, fusion loop
- `display_manager.py` - UI overlay rendering
- `tools/demo_test_videos.py` - demo replay runner (same pipeline)
- `config.yaml` - runtime defaults
- `TEST/` - prerecorded demo clips

---

## 4) Environment and dependencies (your setup)

Project path:
- `C:\\programing\\MachineLearning\\threat_detection_system`

Conda env used by scripts:
- `threat-detect`

Python executable commonly used in this repo:
- `C:\Users\Amritansu Aditya\.conda\envs\threat-detect\python.exe`

Install requirements:

```powershell
conda activate threat-detect
cd C:\\programing\\MachineLearning\\threat_detection_system
pip install -r requirements.txt
```

Optional dev deps:

```powershell
pip install -r requirements-dev.txt
```

---

## 5) Run commands

### A) Live mode (realtime camera + mic)

From PowerShell:

```powershell
conda activate threat-detect
cd C:\\programing\\MachineLearning\\threat_detection_system
python main.py --verbose
```

Or with absolute python:

```powershell
& 'C:\Users\Amritansu Aditya\.conda\envs\threat-detect\python.exe' main.py --verbose
```

Launcher scripts:
- `run_threat_detect.bat`
- `run_threat_detect.ps1`

---

### B) Demo mode (pre-recorded videos through same realtime pipeline)

Run all supported videos in `TEST/`:

```powershell
& 'C:\Users\Amritansu Aditya\.conda\envs\threat-detect\python.exe' tools\demo_test_videos.py --all
```

Run one video:

```powershell
& 'C:\Users\Amritansu Aditya\.conda\envs\threat-detect\python.exe' tools\demo_test_videos.py --video TEST\test3.mp4
```

Headless:

```powershell
& 'C:\Users\Amritansu Aditya\.conda\envs\threat-detect\python.exe' tools\demo_test_videos.py --video TEST\test3.mp4 --headless
```

Save annotated output:

```powershell
& 'C:\Users\Amritansu Aditya\.conda\envs\threat-detect\python.exe' tools\demo_test_videos.py --all --save-output demo_outputs
```

Batch helper:
- `run_test_video_demo.bat`

Demo controls:
- `q` / `Esc`: quit
- `n`: next video
- `Space`: pause/resume

---

## 6) Configuration (`config.yaml`)

Current important defaults:

- `system.target_fps: 30`
- `system.fusion_hz: 1`
- `visual.person_model: yolov8n.pt`
- `visual.weapon_model: camera_detection/models/weapon_best.pt`
- `visual.person_conf: 0.60`
- `visual.person_min_area: 3500`
- `visual.weapon_detect_every_n: 3`
- `visual.face_recognize_every_n: 30`
- `audio.sample_rate: 16000`
- `audio.device: cpu`
- `audio.vad_model: models/silero_vad.onnx`

Env overrides format:

```powershell
$env:THREAT_SYSTEM__TARGET_FPS="20"
$env:THREAT_VISUAL__PERSON_CONF="0.65"
```

---

## 7) Risk model (current behavior)

Risk uses:
- temporal decay from previous score
- weighted sum of visual and audio components
- cross-modal bonus when audio intent confirms visual concern

High-level factors:
- Unknown person (after grace period)
- Weapon associated with person
- Unknown + associated weapon
- Emotion intensity (anger/fear)
- Intent (threat/distress)
- Sudden anomaly/acoustic distress cues
- Audio-visual confirmation bonuses

State thresholds map score to:
- `<20` IDLE
- `<45` CAUTION
- `<70` EVALUATING
- `<90` ALERT
- `>=90` CRITICAL

---

## 8) Identity and weapon association notes

- Identity from FaceNet embedding comparison (`face_embeddings.npz`)
- Weapon association uses hand-aware logic first, then stricter spatial fallback
- Weapon detections are deduplicated to reduce multiple boxes for same object
- Weapon detections include IDs (e.g., `W1`) for UI traceability

---

## 9) Audio pipeline notes

Pipeline stages:
1. VAD
2. Emotion analysis
3. STT transcription
4. Intent classification
5. Acoustic fallback features for distress/anomaly

If transformer models are unavailable (network/cache restrictions), app falls back to heuristic/audio-feature behavior so UI and scoring can still update dynamically.

Transformer control env vars:

```powershell
$env:THREAT_USE_TRANSFORMER_INTENT="1"
# Optional: force local cache only
$env:THREAT_TRANSFORMER_LOCAL_ONLY="1"
```

---

## 10) Logs and troubleshooting

Primary logs:
- `threat_detection.log`
- `debug_output.log`

Common issues:
- **No camera/mic**: app can still run partially; check device availability.
- **Transformer load errors**: usually network/cache issue; use local cache or fallback mode.
- **Too many weapon boxes**: raise thresholds and keep dedupe enabled (already tuned in current code).
- **Identity flips**: ensure face embeddings quality and lighting; stricter thresholds are already applied.

---

## 11) Testing

General:

```powershell
conda activate threat-detect
cd C:\\programing\\MachineLearning\\threat_detection_system
pytest -q
```

Focused unit tests (example):

```powershell
pytest -p no:cacheprovider tests/legacy/test_unit_core.py tests/legacy/test_unit_fusion.py -q
```

---

## 12) Quick start (recommended)

1. Activate env and install deps.
2. Run demo first to validate pipeline:
   - `tools/demo_test_videos.py --all`
3. Run live mode:
   - `python main.py --verbose`
4. Tune `config.yaml` thresholds if needed.

---

## 13) Important reminder

This app is a **realtime decision-support system** and should be tuned with your real scene data.  
For deployment, validate thresholds and risk behavior clip-by-clip with representative homeowner/unknown/weapon/audio scenarios.

