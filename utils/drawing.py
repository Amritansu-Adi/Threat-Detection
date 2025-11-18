import cv2

def draw_person_box(frame, person):
    """Draws the bounding box and label for a tracked person."""
    x1, y1, x2, y2 = [int(c) for c in person.bbox]
    display_name = person.get_label()
    
    color = (0, 255, 0) if display_name != 'Unknown' else (0, 255, 255)
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"ID: {person.id} - {display_name}"
    
    if person.has_weapon:
        label += " [WEAPON]"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
        
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

def draw_weapon_box(frame, bbox, label, holder_label):
    """Draws a bounding box and a formatted label for a weapon."""
    x1, y1, x2, y2 = [int(c) for c in bbox]
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    holder_text = holder_label if holder_label is not None else 'Unassociated'
    full_label = f'{label} (Holder: {holder_text})'
    
    (tw, th), _ = cv2.getTextSize(full_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    
    # Position label
    label_y = y1 - 10 if y1 - 10 > 10 else y2 + 20
    cv2.rectangle(frame, (x1, label_y - th - 5), (x1 + tw, label_y), (0, 0, 255), -1)
    cv2.putText(frame, full_label, (x1, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

def draw_threat_level(frame, threat_level):
    """Displays the current threat level on the frame."""
    if threat_level and threat_level != 'NO_THREAT':
        cv2.putText(frame, f'THREAT: {threat_level}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
