import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU

import pickle
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

# Load LSTM model and label encoder
model = load_model('lstm_model.h5')
label_dict = pickle.load(open('label_encoder.pickle', 'rb'))
label_encoder = label_dict['label_encoder']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, max_num_hands=2)

# Sequence buffer for real-time detection
sequence_buffer = deque(maxlen=30)  # Store last 30 frames
prediction_buffer = deque(maxlen=10)  # Smooth predictions
current_sign = "Starting..."
confidence_threshold = 0.7

print("🎬 Real-time sequence detection active!")
print("Perform complete sign movements for best results!")

while True:
    data_aux = []
    ret, frame = cap.read()
    
    if not ret:
        continue
        
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

        # Process exactly 2 hands (with padding)
        for hand_index in range(2):
            if hand_index < len(results.multi_hand_landmarks):
                hand_landmarks = results.multi_hand_landmarks[hand_index]
                hand_x = [lm.x for lm in hand_landmarks.landmark]
                hand_y = [lm.y for lm in hand_landmarks.landmark]
                
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(hand_x))
                    data_aux.append(lm.y - min(hand_y))
            else:
                data_aux.extend([0.0] * 42)

        # Add frame to sequence buffer
        if len(data_aux) == 84:
            sequence_buffer.append(data_aux)
            
            # Make prediction when we have enough frames
            if len(sequence_buffer) == 30:
                sequence_array = np.array(sequence_buffer)  # Shape: (30, 84)
                sequence_array = np.expand_dims(sequence_array, axis=0)  # Shape: (1, 30, 84)
                
                # Predict
                prediction = model.predict(sequence_array, verbose=0)
                confidence = np.max(prediction)
                predicted_class = np.argmax(prediction)
                
                # Only accept high-confidence predictions
                if confidence > confidence_threshold:
                    predicted_sign = label_encoder.inverse_transform([predicted_class])[0]
                    prediction_buffer.append(predicted_sign)
                    
                    # Use most frequent prediction in buffer (smoothing)
                    if len(prediction_buffer) == 10:
                        current_sign = max(set(prediction_buffer), key=prediction_buffer.count)
        
        # Get bounding box
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

        # Display current sign
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, f"Sign: {current_sign}", (x1, y1 - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Buffer: {len(sequence_buffer)}/30", (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    else:
        # No hands detected - clear buffer
        if sequence_buffer:
            sequence_buffer.clear()
            current_sign = "No hands detected"
    
    # Display instructions
    cv2.putText(frame, "Perform complete sign movements", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Press 'Q' to quit", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('🎬 SignLink - Dynamic Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
