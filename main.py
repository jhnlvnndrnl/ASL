import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Hand skeleton + palm connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),     # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),     # Index
    (0, 9), (9,10), (10,11), (11,12),   # Middle
    (0,13), (13,14), (14,15), (15,16),  # Ring
    (0,17), (17,18), (18,19), (19,20),  # Pinky
    (5, 9), (9, 13), (13, 17)           # Palm
]

# Load MediaPipe Hand Landmarker
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# Helper functions
def finger_is_up(hand, tip, dip):
    """Check if finger is up based on tip vs dip landmarks (camera flipped)."""
    return hand[tip].y < hand[dip].y

def thumb_is_up(hand):
    """Thumb is special: check x position instead of y for right hand."""
    return hand[4].x < hand[3].x  # thumb extended

# ASL Letter rules (A-D only)
def classify_letter(hand):
    index = finger_is_up(hand, 8, 6)
    middle = finger_is_up(hand, 12, 10)
    ring = finger_is_up(hand, 16, 14)
    pinky = finger_is_up(hand, 20, 18)
    thumb = thumb_is_up(hand)

    # Letter A: all fingers down, thumb out
    if not index and not middle and not ring and not pinky and not thumb:
        return "A"

    # Letter B: all fingers up, thumb across palm
    if index and middle and ring and pinky and thumb:
        return "B"

    # Letter C: all fingers curved, thumb and fingers make a C shape
    if index and middle and ring and pinky and not thumb:
        return "C"

    # Letter D: index up, others down, thumb touching middle finger
    if index and not middle and not ring and not pinky and not thumb:
        return "D"

    return ""  # nothing else

# Open camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    letter = ""
    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        h, w, _ = frame.shape

        # Draw joints
        for i, lm in enumerate(hand):
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)
            cv2.putText(frame, str(i), (cx + 5, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (42, 191, 250), 1)

        # Draw skeleton
        for start, end in HAND_CONNECTIONS:
            x1, y1 = int(hand[start].x * w), int(hand[start].y * h)
            x2, y2 = int(hand[end].x * w), int(hand[end].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 46, 91), 1)

        # Classify letter
        letter = classify_letter(hand)
        if letter:
            cv2.putText(frame, f"Letter: {letter}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Tracking + ASL A-D", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()