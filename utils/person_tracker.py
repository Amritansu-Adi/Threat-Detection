import logging
from collections import OrderedDict
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
RECOGNITION_BUFFER_SIZE = 15  # Number of frames to buffer for recognition
WEAPON_LOCK_FRAMES = 5  # Number of consecutive frames to confirm weapon association

class Person:
    """Represents a tracked person and their state."""

    def __init__(self, person_id, centroid, bbox):
        self.id = person_id
        self.centroid = centroid
        self.bbox = bbox
        self.label = "Unknown"
        self.recognition_buffer = []
        self.label_lock_frames = 0
        self.last_seen_frame = 0
        self.has_weapon = False
        self.weapon_association_buffer = 0
        self.frames_in_view = 0

    def get_label(self):
        """Return the person's current label."""
        return self.label

    def lock_label(self, label, lock_duration):
        """Lock the current label, preventing it from being changed."""
        if self.label != label:
            logger.info(f"Locked label for person {self.id} as '{label}' for {lock_duration} frames")
        self.label = label
        self.label_lock_frames = lock_duration

    def update_recognition(self, label):
        """Update the person's recognition state with a new label."""
        self.recognition_buffer.append(label)
        if len(self.recognition_buffer) > 20:
            self.recognition_buffer.pop(0)  # Maintain buffer size

class PersonTracker:
    """Tracks people using centroids and manages their state."""

    def __init__(self, max_disappeared=150, recognition_lock_duration=50):
        self.next_object_id = 0
        self.max_disappeared = max_disappeared
        self.recognition_lock_duration = recognition_lock_duration
        self.frame_count = 0

        # State for persistent recognition
        self.name_buffers = OrderedDict()
        self.locked_names = OrderedDict()
        self.frames_since_locked = OrderedDict()

        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

    def _register(self, centroid, bbox):
        obj_id = self.next_object_id

        person = Person(obj_id, centroid, bbox)
        self.objects[obj_id] = person
        self.disappeared[obj_id] = 0

        # Initialize state for the new person
        self.name_buffers[obj_id] = []
        self.locked_names[obj_id] = "Unknown" # Default name
        self.frames_since_locked[obj_id] = 0

        self.next_object_id += 1
        return obj_id

    def _deregister(self, object_id):
        """Deregister a person who has disappeared."""
        person = self.objects.pop(object_id)
        self.disappeared.pop(object_id)
        logger.info(f"Deregistered person {object_id} ({person.get_label()})")

    def update(self, person_boxes, frame_number):
        if len(person_boxes) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self._deregister(object_id)
            return self.objects

        input_centroids = np.zeros((len(person_boxes), 2), dtype="int")
        for (i, (startX, startY, endX, endY, _, _)) in enumerate(person_boxes):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self._register(input_centroids[i], person_boxes[i][:4])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = np.array([p.centroid for p in self.objects.values()])

            # Compute distances between existing and new centroids
            D = np.sqrt(np.sum((object_centroids[:, np.newaxis] - input_centroids) ** 2, axis=2))

            # Find best matches
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[row]
                person = self.objects[object_id]
                person.centroid = input_centroids[col]
                person.bbox = person_boxes[col][:4]
                person.disappeared = 0
                person.frames_in_view += 1
                
                # Decrement lock counter if it's active
                if person.label_lock_frames > 0:
                    person.label_lock_frames -= 1

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(D.shape[0])).difference(used_rows)
            unused_cols = set(range(D.shape[1])).difference(used_cols)

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self._deregister(object_id)

            for col in unused_cols:
                self._register(input_centroids[col], person_boxes[col][:4])

        self.frame_count = frame_number
        return self.objects

    def handle_duplicate_identities(self, updated_person_id, new_label):
        """Check for and resolve duplicate identities."""
        if new_label == "Unknown":
            return

        duplicates = []
        for person_id, person in self.objects.items():
            if person.get_label() == new_label:
                duplicates.append(person)

        if len(duplicates) > 1:
            # Keep the one with the highest ID (the newest)
            newest_person = max(duplicates, key=lambda p: p.id)
            for person in duplicates:
                if person.id != newest_person.id:
                    logger.info(f"Resolving duplicate identity for '{new_label}'. Removing older person ID {person.id} and keeping newer ID {newest_person.id}.")
                    self._deregister(person.id)

    def update_person_recognition(self, person_id, label):
        if person_id not in self.objects:
            return

        person = self.objects[person_id]

        # Only update buffer if the label is not locked
        if person.label_lock_frames <= 0:
            person.update_recognition(label)

            # If buffer is full and consistent, lock the label
            if len(person.recognition_buffer) >= RECOGNITION_BUFFER_SIZE and all(
                l == label for l in person.recognition_buffer
            ):
                person.lock_label(label, self.recognition_lock_duration)
                person.recognition_buffer.clear() # Clear buffer after locking
                # After locking a new label, check for duplicates
                self.handle_duplicate_identities(person_id, label)

    def update_weapon_association(self, person_id, has_weapon):
        """Update the weapon association state for a person with smoothing."""
        if person_id not in self.objects:
            return

        person = self.objects[person_id]

        if has_weapon:
            person.weapon_association_buffer += 1
        else:
            # If the weapon is not associated in this frame, reset the buffer
            person.weapon_association_buffer = 0
            person.has_weapon = False # Immediately lose weapon status

        # If the association has been consistent, lock the weapon status
        if person.weapon_association_buffer >= WEAPON_LOCK_FRAMES:
            if not person.has_weapon:
                logger.info(f"Confirmed and locked weapon association for person {person_id} ({person.get_label()})")
            person.has_weapon = True
        
        # If buffer is positive but not yet locked, they don't have a weapon confirmed
        elif person.weapon_association_buffer > 0:
            logger.debug(f"Weapon association pending for person {person_id} ({person.get_label()})... {person.weapon_association_buffer}/{WEAPON_LOCK_FRAMES}")


    def get_persons(self):
        """Return a list of all currently tracked persons."""
        return list(self.objects.values())
