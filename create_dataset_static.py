# In create_dataset.py - Use this simpler version:
import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5, max_num_hands=2)

DATA_DIR = './data'
data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        if img is None:
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            data_aux = []
            
            # Process exactly 2 hands (pad if necessary)
            for hand_index in range(2):
                if hand_index < len(results.multi_hand_landmarks):
                    hand_landmarks = results.multi_hand_landmarks[hand_index]
                    hand_x = [lm.x for lm in hand_landmarks.landmark]
                    hand_y = [lm.y for lm in hand_landmarks.landmark]
                    
                    for lm in hand_landmarks.landmark:
                        data_aux.append(lm.x - min(hand_x))
                        data_aux.append(lm.y - min(hand_y))
                else:
                    # Pad with zeros for missing hand
                    data_aux.extend([0.0] * 42)
            
            # Should always be 84 features now
            if len(data_aux) == 84:
                data.append(data_aux)
                labels.append(dir_)

print(f"Final dataset: {len(data)} samples")
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()