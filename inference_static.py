import pickle
import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model_enhanced.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# CRITICAL: Allow 2 hands like in training
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, max_num_hands=2)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'G', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: '0', 27: '1', 28: '2', 29: '3',
    30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9'
}

while True:
    data_aux = []  # This will store our 84 features

    ret, frame = cap.read()
    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        # Draw landmarks for all hands
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        # Process exactly 2 hands (with padding) - JUST LIKE IN TRAINING
        for hand_index in range(2):
            if hand_index < len(results.multi_hand_landmarks):
                hand_landmarks = results.multi_hand_landmarks[hand_index]
                hand_x = [lm.x for lm in hand_landmarks.landmark]
                hand_y = [lm.y for lm in hand_landmarks.landmark]
                
                # Normalize this hand's coordinates
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(hand_x))
                    data_aux.append(lm.y - min(hand_y))
            else:
                # Pad with zeros for missing hand
                data_aux.extend([0.0] * 42)

        # Get bounding box from all detected hands
        all_x = []
        all_y = []
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                all_x.append(lm.x)
                all_y.append(lm.y)
        
        x1 = int(min(all_x) * W) - 10
        y1 = int(min(all_y) * H) - 10
        x2 = int(max(all_x) * W) + 10
        y2 = int(max(all_y) * H) + 10

        # Make prediction (should have 84 features now)
        if len(data_aux) == 84:  # Safety check
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()