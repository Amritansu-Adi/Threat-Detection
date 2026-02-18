# Visual Pipeline Upgrades - TODO List

## Overview
This TODO list details the specific implementation steps for upgrading the visual pipeline from frame-by-frame detection to persistent Multi-Object Tracking (MOT) with identity caching and audio-visual hooks.

## Current System State
- ✅ YOLOv8 person/weapon detection working
- ✅ Custom centroid tracker (PersonTracker class)
- ✅ FaceNet recognition on every frame
- ✅ Static threat levels (HIGH/MINOR/NO_THREAT)
- ✅ Single-threaded processing

## TODO Items

### 1. Multi-Object Tracking (MOT) Integration
**Status**: Ready for Implementation
**Priority**: High
**Files**: `camera_detection/yolo_detect.py`, `utils/person_tracker.py`

#### Tasks:
- [ ] Replace `PersonTracker` with YOLOv8's built-in ByteTrack tracker
- [ ] Update `main()` function to use `model.track()` instead of separate detect + track
- [ ] Modify result parsing to handle ByteTrack output format
- [ ] Update person data structure from objects to dictionaries
- [ ] Test MOT persistence across occlusions and re-entries

#### Code Changes:
```python
# Before (custom tracker)
humans = detect_humans(frame)
tracked_persons_dict = tracker.update(humans, frame_idx)
tracked_persons = list(tracked_persons_dict.values())

# After (YOLO ByteTrack)
results = person_model.track(source=frame, persist=True, tracker="bytetrack")
tracked_persons = []
for result in results:
    for box in result.boxes:
        tid = int(box.id) if box.id is not None else -1
        bbox = box.xyxy[0].cpu().numpy()
        tracked_persons.append({
            'id': tid,
            'bbox': bbox,
            'conf': box.conf.item(),
            'class': int(box.cls.item())
        })
```

### 2. Identity Caching System
**Status**: Ready for Implementation
**Priority**: High
**Files**: `camera_detection/yolo_detect.py`

#### Tasks:
- [ ] Add global `identity_cache` dictionary: `Track_ID -> {'label': str, 'last_updated': int}`
- [ ] Implement conditional FaceNet execution logic
- [ ] Add cache initialization and cleanup
- [ ] Update face recognition loop with caching
- [ ] Test cache persistence and accuracy

#### Code Structure:
```python
identity_cache = {}  # Global cache: track_id -> identity info

# In main loop, for each tracked person:
tid = person['id']
cache_entry = identity_cache.get(tid, {'label': 'Unknown', 'last_updated': 0})

should_recognize = (
    cache_entry['label'] == 'Unknown' or
    (frame_idx - cache_entry['last_updated']) > 30  # Re-check every 30 frames
)

if should_recognize and face_img.size > 0:
    name, _ = recognize_face(face_img, mtcnn_face_detector, facenet_resnet, device)
    if name:
        identity_cache[tid] = {'label': name, 'last_updated': frame_idx}
        person['label'] = name
```

### 3. Audio-Visual Interrupt Hooks
**Status**: Ready for Implementation
**Priority**: Medium
**Files**: `camera_detection/yolo_detect.py`

#### Tasks:
- [ ] Add global flags: `audio_trigger_active = False`, `deep_search_mode = False`
- [ ] Implement conditional processing based on flags
- [ ] Add placeholder logic for cross-modal triggers
- [ ] Prepare integration points for audio thread
- [ ] Test flag-based behavior switching

#### Code Changes:
```python
# Global flags for cross-modal integration
audio_trigger_active = False
deep_search_mode = False

# In main processing loop:
if audio_trigger_active:
    # Enhanced processing: lower thresholds, more frequent recognition
    pass

if deep_search_mode:
    # Deep search: extended face recognition, additional analysis
    pass
```

### 4. Dict-Based Person Structures Migration
**Status**: Ready for Implementation
**Priority**: High
**Files**: `camera_detection/yolo_detect.py`, `utils/drawing.py`

#### Tasks:
- [ ] Convert `Person` objects to dictionaries in main loop
- [ ] Update all references from `person.bbox` to `person['bbox']`
- [ ] Update all references from `person.get_label()` to `identity_cache.get(person['id'], {}).get('label', 'Unknown')`
- [ ] Update all references from `person.has_weapon` to `person['has_weapon']`
- [ ] Update `draw_person_box()` to handle both dict and object inputs
- [ ] Test all drawing and processing functions

#### Migration Pattern:
```python
# Before (object-based)
person = Person(id, centroid, bbox)
person.has_weapon = True
display_name = person.get_label()

# After (dict-based)
person = {
    'id': tid,
    'bbox': bbox,
    'has_weapon': False,
    'label': 'Unknown'
}
person['has_weapon'] = True
display_name = identity_cache.get(person['id'], {}).get('label', 'Unknown')
```

### 5. Weapon Detection Optimization
**Status**: Needs Review
**Priority**: Medium
**Files**: `camera_detection/yolo_detect.py`

#### Tasks:
- [ ] Add conditional weapon model loading (graceful fallback)
- [ ] Update weapon association logic for dict-based persons
- [ ] Optimize async weapon detection threading
- [ ] Test weapon detection with MOT integration
- [ ] Handle missing weapon model file gracefully

#### Code Changes:
```python
# Conditional weapon model loading
weapon_model_path = YOLOV8_WEAPON_WEIGHTS
if os.path.exists(weapon_model_path):
    weapon_model = YOLO(weapon_model_path).to(device)
    logging.info(f"Loaded Weapon Model: {weapon_model_path}")
else:
    weapon_model = None
    logging.warning(f"Weapon model not found: {weapon_model_path}. Weapon detection disabled.")
```

### 6. Threat Logic Preparation for SRA
**Status**: Ready for Implementation
**Priority**: Medium
**Files**: `camera_detection/yolo_detect.py`

#### Tasks:
- [ ] Refactor static threat levels to dynamic risk scoring preparation
- [ ] Add evidence collection for SRA integration
- [ ] Update alert system to use risk-based thresholds
- [ ] Prepare data structures for cross-modal evidence
- [ ] Test threat logic with new person structures

#### Future SRA Integration:
```python
# Prepare for SRA (Shared Risk Accumulator)
risk_evidence = {
    'unknown_person': unknown_person_present,
    'known_with_weapon': known_with_weapon,
    'unknown_with_weapon': unknown_with_weapon,
    'audio_distress': audio_trigger_active,
    'unassociated_weapons': len(unassociated_weapons) > 0
}

# Calculate dynamic risk score (0-100)
risk_score = calculate_sra_score(risk_evidence, identity_cache, tracked_persons)
```

### 7. Testing and Validation
**Status**: Pending Implementation
**Priority**: High
**Files**: All modified files

#### Tasks:
- [ ] Create test script for MOT persistence
- [ ] Test identity caching accuracy and performance
- [ ] Validate weapon association with new structures
- [ ] Performance testing: FPS, memory usage, CPU/GPU utilization
- [ ] Integration testing: all components working together
- [ ] Edge case testing: occlusions, lighting changes, multiple persons

### 8. Code Cleanup and Documentation
**Status**: Pending Implementation
**Priority**: Low
**Files**: All modified files

#### Tasks:
- [ ] Remove deprecated PersonTracker usage
- [ ] Update comments and docstrings
- [ ] Add type hints for new dict structures
- [ ] Create migration guide for future changes
- [ ] Update README with new features

## Implementation Order
1. **MOT Integration** (Foundation)
2. **Dict Migration** (Data structure)
3. **Identity Caching** (Performance)
4. **Weapon Detection Updates** (Compatibility)
5. **Audio Hooks** (Extensibility)
6. **Threat Logic Updates** (SRA preparation)
7. **Testing & Validation** (Quality assurance)
8. **Cleanup** (Maintenance)

## Dependencies
- YOLOv8 with ByteTrack support (ultralytics>=8.0.0)
- Existing dependencies: torch, opencv, facenet-pytorch
- Optional: lap (Linear Assignment Problem) for ByteTrack

## Risk Mitigation
- **Incremental Changes**: Each TODO can be implemented independently
- **Backwards Compatibility**: Maintain object-based fallbacks during migration
- **Testing**: Comprehensive testing before/after each change
- **Rollback**: Easy reversion if issues arise

## Success Criteria
- [ ] MOT tracks persons persistently across occlusions
- [ ] Identity caching reduces FaceNet calls by 80%+
- [ ] System maintains 30+ FPS processing
- [ ] All existing functionality preserved
- [ ] Ready for Audio Pipeline integration</content>
<parameter name="filePath">c:\programing\MachineLearning\person_detect\yolo\VISUAL_PIPELINE_TODO.md