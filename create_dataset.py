import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './dynamic_data'  # Changed to dynamic data folder
sequences = []  # Changed from data to sequences
labels = []
sequence_length = 30

for sign_class in os.listdir(DATA_DIR):
    sign_path = os.path.join(DATA_DIR, sign_class)
    
    for sequence_num in os.listdir(sign_path):
        sequence_path = os.path.join(sign_path, sequence_num)
        sequence_data = []  # Store landmarks for this sequence
        
        # Process each frame in the sequence
        for frame_num in range(sequence_length):
            frame_path = os.path.join(sequence_path, f'{frame_num}.jpg')
            
            if not os.path.exists(frame_path):
                continue
                
            img = cv2.imread(frame_path)
            if img is None:
                continue
                
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            
            if results.multi_hand_landmarks:
                frame_landmarks = []
                
                # Process exactly 2 hands (pad if necessary)
                for hand_index in range(2):
                    if hand_index < len(results.multi_hand_landmarks):
                        hand_landmarks = results.multi_hand_landmarks[hand_index]
                        hand_x = [lm.x for lm in hand_landmarks.landmark]
                        hand_y = [lm.y for lm in hand_landmarks.landmark]
                        
                        for lm in hand_landmarks.landmark:
                            frame_landmarks.append(lm.x - min(hand_x))
                            frame_landmarks.append(lm.y - min(hand_y))
                    else:
                        # Pad with zeros for missing hand
                        frame_landmarks.extend([0.0] * 42)
                
                # Should always be 84 features per frame
                if len(frame_landmarks) == 84:
                    sequence_data.append(frame_landmarks)
        
        # Only add complete sequences
        if len(sequence_data) == sequence_length:
            sequences.append(sequence_data)
            labels.append(sign_class)

print(f"Final dataset: {len(sequences)} sequences")
print(f"Sequence shape: {np.array(sequences[0]).shape}")  # Should be (30, 84)

# Save as sequences instead of single data points
f = open('sequences.pickle', 'wb')
pickle.dump({'sequences': sequences, 'labels': labels}, f)
f.close()
