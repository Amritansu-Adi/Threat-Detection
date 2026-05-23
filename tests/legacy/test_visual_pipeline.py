"""
Tests for Visual Pipeline Components

Tests the complete visual processing pipeline:
- Person detection
- Weapon detection
- Tracking
- Face recognition
- Weapon association
- Visual pipeline orchestration
"""

import unittest
import numpy as np
import cv2
import tempfile
import os
import logging
from unittest.mock import Mock, patch, MagicMock
from queue import Queue

# Import visual components
from visual.detector import PersonDetector, WeaponDetector, FaceRecognizer
from visual.tracker import PersonTracker, TrackedPerson
from visual.weapon_mapper import WeaponMapper, HandDetector
from visual.visual_pipeline import VisualPipeline
from core.data_structures import VisualData


class TestPersonDetector(unittest.TestCase):
    """Test person detection component."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock YOLO model
        self.mock_model = Mock()
        self.mock_model.names = {0: 'person'}

        # Create detector with mocked model
        with patch('visual.detector.YOLO') as mock_yolo:
            mock_yolo.return_value = self.mock_model
            self.detector = PersonDetector(
                model_path="dummy.pt",
                conf_threshold=0.5,
                min_area=100
            )

    def test_initialization(self):
        """Test detector initialization."""
        self.assertIsNotNone(self.detector.model)
        self.assertEqual(self.detector.conf_threshold, 0.5)
        self.assertEqual(self.detector.min_area, 100)

    def test_detect_persons(self):
        """Test person detection."""
        # Mock detection results - need to mock PyTorch tensor behavior
        mock_tensor_xyxy = Mock()
        mock_tensor_xyxy.cpu.return_value.numpy.return_value = np.array([[10, 10, 110, 110], [200, 200, 300, 300]])
        
        mock_tensor_conf = Mock()
        mock_tensor_conf.cpu.return_value.numpy.return_value = np.array([0.8, 0.6])
        
        mock_tensor_cls = Mock()
        mock_tensor_cls.cpu.return_value.numpy.return_value = np.array([0, 0])
        
        mock_results = Mock()
        mock_results.boxes.xyxy = mock_tensor_xyxy
        mock_results.boxes.conf = mock_tensor_conf
        mock_results.boxes.cls = mock_tensor_cls
        self.mock_model.return_value = [mock_results]

        # Create test frame
        frame = np.zeros((400, 400, 3), dtype=np.uint8)

        # Detect persons
        detections = self.detector.detect(frame)

        # Verify results
        self.assertEqual(len(detections), 2)
        self.assertEqual(detections[0].bbox, (10, 10, 110, 110))
        self.assertEqual(detections[0].confidence, 0.8)
        self.assertEqual(detections[0].class_name, 'person')

    def test_detect_small_objects_filtered(self):
        """Test that small objects are filtered out."""
        # Mock detection results with small bbox
        mock_results = Mock()
        mock_results.boxes.xyxy = np.array([[10, 10, 20, 20]])  # Area = 100, exactly at threshold
        mock_results.boxes.conf = np.array([0.8])
        mock_results.boxes.cls = np.array([0])
        self.mock_model.return_value = [mock_results]

        frame = np.zeros((400, 400, 3), dtype=np.uint8)

        # Should be filtered out (area < min_area)
        detections = self.detector.detect(frame)
        self.assertEqual(len(detections), 0)


class TestWeaponDetector(unittest.TestCase):
    """Test weapon detection component."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_model = Mock()
        self.mock_model.names = {0: 'knife', 1: 'gun'}

        with patch('visual.detector.YOLO') as mock_yolo:
            mock_yolo.return_value = self.mock_model
            self.detector = WeaponDetector(
                model_path="dummy.pt",
                weapon_classes=['knife', 'gun'],
                conf_threshold=0.5
            )
            self.detector.using_fallback_model = False

    def test_initialization(self):
        """Test detector initialization."""
        self.assertEqual(len(self.detector.weapon_class_ids), 2)
        self.assertIn(0, self.detector.weapon_class_ids)  # knife
        self.assertIn(1, self.detector.weapon_class_ids)  # gun

    def test_focused_infer_on_person(self):
        """Test focused weapon detection."""
        # Mock inference results - need to mock PyTorch tensor behavior
        mock_tensor_xyxy = Mock()
        mock_tensor_xyxy.cpu.return_value.numpy.return_value = np.array([[50, 50, 70, 70]])
        
        mock_tensor_conf = Mock()
        mock_tensor_conf.cpu.return_value.numpy.return_value = np.array([0.8])
        
        mock_tensor_cls = Mock()
        mock_tensor_cls.cpu.return_value.numpy.return_value = np.array([0])  # knife
        
        mock_results = Mock()
        mock_results.boxes.xyxy = mock_tensor_xyxy
        mock_results.boxes.conf = mock_tensor_conf
        mock_results.boxes.cls = mock_tensor_cls
        self.mock_model.return_value = [mock_results]

        # Create test frame and person bbox
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        person_bbox = (20, 20, 180, 180)  # Large person bbox

        detections = self.detector.focused_infer_on_person(frame, person_bbox)

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].class_name, 'knife')
        self.assertEqual(detections[0].confidence, 0.8)


class TestFaceRecognizer(unittest.TestCase):
    """Test face recognition component."""

    def setUp(self):
        """Set up test fixtures."""
        with patch('visual.detector.MTCNN'), \
             patch('visual.detector.InceptionResnetV1'), \
             patch('visual.detector.load_known_faces'):
            self.recognizer = FaceRecognizer()

    def test_recognize_face(self):
        """Test face recognition."""
        # Mock face image
        face_img = np.zeros((100, 100, 3), dtype=np.uint8)

        # Mock recognition result
        with patch('visual.detector.recognize_face') as mock_recognize:
            mock_recognize.return_value = ("John Doe", 0.95)

            result = self.recognizer.recognize(face_img)

            self.assertEqual(result.name, "John Doe")
            self.assertEqual(result.confidence, 0.95)

    def test_recognize_no_face(self):
        """Test recognition with no face detected."""
        face_img = np.zeros((10, 10, 3), dtype=np.uint8)  # Too small

        with patch('visual.detector.recognize_face') as mock_recognize:
            mock_recognize.return_value = (None, 0.0)

            result = self.recognizer.recognize(face_img)

            self.assertIsNone(result.name)
            self.assertEqual(result.confidence, 0.0)


class TestPersonTracker(unittest.TestCase):
    """Test person tracking component."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock ByteTrack to force fallback tracker
        with patch('visual.tracker.ByteTrackPersonTracker', None):
            self.tracker = PersonTracker(max_age=5, min_hits=1)

    def test_track_persons(self):
        """Test basic person tracking."""
        # First frame detections
        detections1 = [(10, 10, 50, 50, 0.8, 'person')]
        tracks1 = self.tracker.update(detections1, 0)

        self.assertEqual(len(tracks1), 1)
        person_id = list(tracks1.keys())[0]
        self.assertEqual(tracks1[person_id].bbox, (10, 10, 50, 50))

        # Second frame - same person moved slightly
        detections2 = [(15, 15, 55, 55, 0.8, 'person')]
        tracks2 = self.tracker.update(detections2, 1)

        self.assertEqual(len(tracks2), 1)
        self.assertIn(person_id, tracks2)
        self.assertEqual(tracks2[person_id].bbox, (15, 15, 55, 55))

    def test_track_velocity_calculation(self):
        """Test velocity calculation."""
        # First position
        detections1 = [(10, 10, 50, 50, 0.8, 'person')]
        tracks1 = self.tracker.update(detections1, 0)
        person_id = list(tracks1.keys())[0]

        # Second position (moved 10 pixels right, 10 pixels down)
        detections2 = [(20, 20, 60, 60, 0.8, 'person')]
        tracks2 = self.tracker.update(detections2, 1)

        velocity = tracks2[person_id].velocity
        self.assertEqual(velocity, (10.0, 10.0))  # dx=10, dy=10


class TestWeaponMapper(unittest.TestCase):
    """Test weapon-to-person association."""

    def setUp(self):
        """Set up test fixtures."""
        with patch('visual.weapon_mapper.mp'):
            self.mapper = WeaponMapper(overlap_threshold=0.8)

    def test_map_weapons_by_overlap(self):
        """Test weapon mapping by bbox overlap."""
        # Person and weapon with high overlap
        persons = [{'id': 1, 'bbox': (10, 10, 50, 50), 'name': 'Unknown'}]
        weapons = [(15, 15, 45, 45, 0.9, 'knife')]  # High overlap with person

        # Create mock frame
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        associations = self.mapper.map_weapons_to_persons(frame, weapons, persons)

        self.assertIn(1, associations)
        self.assertEqual(len(associations[1]), 1)
        self.assertEqual(associations[1][0], weapons[0])

    def test_map_weapons_no_association(self):
        """Test weapons with no person association."""
        persons = [{'id': 1, 'bbox': (10, 10, 30, 30), 'name': 'Unknown'}]
        weapons = [(80, 80, 90, 90, 0.9, 'knife')]  # Far from person

        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        associations = self.mapper.map_weapons_to_persons(frame, weapons, persons)

        self.assertEqual(len(associations), 0)

    def test_get_unassociated_weapons(self):
        """Test getting unassociated weapons."""
        persons = [{'id': 1, 'bbox': (10, 10, 50, 50), 'name': 'Unknown'}]
        weapons = [(15, 15, 45, 45, 0.9, 'knife'), (80, 80, 90, 90, 0.8, 'gun')]

        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        associations = self.mapper.map_weapons_to_persons(frame, weapons, persons)
        unassociated = self.mapper.get_unassociated_weapons(weapons, associations)

        self.assertEqual(len(unassociated), 1)
        self.assertEqual(unassociated[0], weapons[1])  # The distant gun


class TestVisualPipeline(unittest.TestCase):
    """Test visual pipeline orchestration."""

    def setUp(self):
        """Set up test fixtures."""
        # Create queues
        self.visual_to_fusion_q = Queue(maxsize=10)
        self.fusion_to_visual_q = Queue(maxsize=10)

        # Mock all the heavy components
        with patch('visual.visual_pipeline.PersonDetector'), \
             patch('visual.visual_pipeline.WeaponDetector'), \
             patch('visual.visual_pipeline.PersonTracker'), \
             patch('visual.visual_pipeline.FaceRecognizer'), \
             patch('visual.visual_pipeline.WeaponMapper'):

            self.pipeline = VisualPipeline(
                visual_to_fusion_queue=self.visual_to_fusion_q,
                fusion_to_visual_queue=self.fusion_to_visual_q,
                target_fps=10.0  # Fast for testing
            )

    def test_initialization(self):
        """Test pipeline initialization."""
        self.assertIsNotNone(self.pipeline.person_detector)
        self.assertIsNotNone(self.pipeline.weapon_detector)
        self.assertIsNotNone(self.pipeline.tracker)
        self.assertIsNotNone(self.pipeline.face_recognizer)
        self.assertIsNotNone(self.pipeline.weapon_mapper)
        self.assertEqual(self.pipeline.target_fps, 10.0)

    def test_submit_frame(self):
        """Test frame submission."""
        self.pipeline.start()  # Start the pipeline first
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Should accept frame
        accepted = self.pipeline.submit_frame(frame)
        self.assertTrue(accepted)

        # Clean up
        self.pipeline.stop()

    def test_get_stats(self):
        """Test statistics retrieval."""
        stats = self.pipeline.get_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn('frames_processed', stats)
        self.assertIn('processing_fps', stats)

    def test_start_stop(self):
        """Test pipeline start/stop."""
        self.pipeline.start()
        self.assertTrue(self.pipeline.running)

        self.pipeline.stop()
        self.assertFalse(self.pipeline.running)


class TestVisualDataStructures(unittest.TestCase):
    """Test visual data structure conversions."""

    def test_visual_data_creation(self):
        """Test VisualData creation."""
        timestamp = 123456.789
        frame_shape = (480, 640, 3)

        persons = [{
            'id': 1,
            'bbox': (10, 10, 50, 50),
            'confidence': 0.9,
            'label': 'John',
            'has_weapon': True,
            'frames_in_view': 10,
            'velocity': (2.0, 1.0)
        }]

        weapons = []  # Empty for this test

        stats = {'frames_processed': 100, 'processing_fps': 25.0}

        visual_data = VisualData(
            timestamp=timestamp,
            frame_shape=frame_shape,
            persons=persons,
            weapons=weapons,
            processing_stats=stats
        )

        self.assertEqual(visual_data.timestamp, timestamp)
        self.assertEqual(visual_data.frame_shape, frame_shape)
        self.assertEqual(len(visual_data.persons), 1)
        self.assertEqual(visual_data.persons[0]['label'], 'John')
        self.assertEqual(visual_data.processing_stats['processing_fps'], 25.0)


if __name__ == '__main__':
    # Set up logging for tests
    logging.basicConfig(level=logging.WARNING)

    # Run tests
    unittest.main(verbosity=2)
