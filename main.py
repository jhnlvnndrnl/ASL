import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# HAND CONNECTIONS (SKELETON)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index
    (0, 9), (9,10), (10,11), (11,12),    # Middle
    (0,13), (13,14), (14,15), (15,16),   # Ring
    (0,17), (17,18), (18,19), (19,20),   # Pinky
    (5, 9), (9, 13), (13, 17)            # Palm
]

# MEDIAPIPE HAND LANDMARKER
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)

# HELPER FUNCTIONS
def finger_is_up(hand, tip, dip):
    """Finger is up if tip is higher than dip"""
    return hand[tip].y < hand[dip].y

def thumb_is_out(hand):
    """Thumb is out using horizontal distance"""
    return abs(hand[4].x - hand[3].x) > 0.03

def all_up(*fingers):
    return all(fingers)

def all_down(*fingers):
    return not any(fingers)

# ASL A–Z CLASSIFIER (RULE-BASED)
def classify_letter(hand):
    index  = finger_is_up(hand, 8, 6)
    middle = finger_is_up(hand, 12, 10)
    ring   = finger_is_up(hand, 16, 14)
    pinky  = finger_is_up(hand, 20, 18)
    thumb  = thumb_is_out(hand)

    # A
    if all_down(index, middle, ring, pinky) and thumb:
        return "A"

    # B
    if all_up(index, middle, ring, pinky) and not thumb:
        return "B"

    # C (approximate)
    if all_up(index, middle, ring, pinky) and thumb:
        return "C"

    # D
    if index and all_down(middle, ring, pinky) and not thumb:
        return "D"

    # E
    if all_down(index, middle, ring, pinky) and not thumb:
        return "E"

    # F (approximate)
    if middle and ring and pinky and thumb and not index:
        return "F"

    # L
    if index and thumb and all_down(middle, ring, pinky):
        return "L"

    # Y
    if thumb and pinky and all_down(index, middle, ring):
        return "Y"

    # J / Z need motion
    if index and not middle and not ring and not pinky:
        return "J / Z (motion)"

    return ""

# CAMERA LOOP
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        h, w, _ = frame.shape

        # Draw landmarks
        for i, lm in enumerate(hand):
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)
            cv2.putText(frame, str(i), (cx + 4, cy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

        # Draw skeleton
        for start, end in HAND_CONNECTIONS:
            x1, y1 = int(hand[start].x * w), int(hand[start].y * h)
            x2, y2 = int(hand[end].x * w), int(hand[end].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 50, 90), 1)

        # Detect letter
        letter = classify_letter(hand)
        if letter:
            cv2.putText(frame, f"Letter: {letter}", (40, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("ASL A–Z Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
