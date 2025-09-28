import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    finger_count = 0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get landmark positions
            lm = hand_landmarks.landmark

            # Finger tips landmark indices (Thumb, Index, Middle, Ring, Pinky)
            tip_ids = [4, 8, 12, 16, 20]

            # Check fingers (excluding thumb)
            for id in tip_ids[1:]:
                if lm[id].y < lm[id - 2].y:  # tip above joint
                    finger_count += 1
            
            # Check thumb (different logic - compare x instead of y)
            if lm[tip_ids[0]].x > lm[tip_ids[0] - 1].x:
                finger_count += 1

            # Draw hand landmarks
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show finger count
    cv2.putText(img, f'Fingers: {finger_count}', (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    cv2.imshow("Hand Finger Counter", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
