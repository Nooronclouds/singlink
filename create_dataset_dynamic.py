import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './dynamic_data'
sequences = []
labels = []
sequence_length = 30
min_required_frames = 25  # Changed from 30 to 25 (83% complete)

print("🚀 Starting dynamic data processing...")
print(f"Looking for data in: {DATA_DIR}")

# Check if directory exists
if not os.path.exists(DATA_DIR):
    print(f"❌ ERROR: Directory {DATA_DIR} not found!")
    exit()

sign_classes = os.listdir(DATA_DIR)
print(f"📁 Found {len(sign_classes)} sign classes: {sign_classes}")

for sign_class in sign_classes:
    sign_path = os.path.join(DATA_DIR, sign_class)
    print(f"\n🔍 Processing sign: {sign_class}")
    
    sequence_nums = os.listdir(sign_path)
    print(f"   Found {len(sequence_nums)} sequences")
    
    for sequence_num in sequence_nums:
        sequence_path = os.path.join(sign_path, sequence_num)
        sequence_data = []
        
        print(f"   📹 Sequence {sequence_num}: ", end="")
        
        valid_frames = 0
        for frame_num in range(sequence_length):
            frame_path = os.path.join(sequence_path, f'{frame_num}.jpg')
            
            if not os.path.exists(frame_path):
                # print(f"❌ Missing frame {frame_num}", end=" ")
                continue
                
            img = cv2.imread(frame_path)
            if img is None:
                # print(f"❌ Corrupted frame {frame_num}", end=" ")
                continue
                
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            
            if results.multi_hand_landmarks:
                frame_landmarks = []
                
                # Process exactly 2 hands
                for hand_index in range(2):
                    if hand_index < len(results.multi_hand_landmarks):
                        hand_landmarks = results.multi_hand_landmarks[hand_index]
                        hand_x = [lm.x for lm in hand_landmarks.landmark]
                        hand_y = [lm.y for lm in hand_landmarks.landmark]
                        
                        for lm in hand_landmarks.landmark:
                            frame_landmarks.append(lm.x - min(hand_x))
                            frame_landmarks.append(lm.y - min(hand_y))
                    else:
                        frame_landmarks.extend([0.0] * 42)
                
                if len(frame_landmarks) == 84:
                    sequence_data.append(frame_landmarks)
                    valid_frames += 1
        
        # LENIENT VERSION: Accept sequences with at least 25 frames
        if len(sequence_data) >= min_required_frames:
            # Pad with zeros to make exactly 30 frames
            while len(sequence_data) < sequence_length:
                sequence_data.append([0.0] * 84)  # Add blank frames
            
            sequences.append(sequence_data)
            labels.append(sign_class)
            print(f"✅ ACCEPTED ({valid_frames}/{sequence_length} frames)")
        else:
            print(f"❌ REJECTED ({valid_frames}/{sequence_length} frames)")

print(f"\n📊 FINAL DATASET SUMMARY:")
print(f"   Total sequences: {len(sequences)}")
print(f"   Total labels: {len(labels)}")

if len(sequences) > 0:
    print(f"   Sequence shape: {np.array(sequences[0]).shape}")
    print(f"   Unique labels: {set(labels)}")
    
    # Show sequence length distribution
    seq_lengths = [len([f for f in seq if any(x != 0 for x in f)]) for seq in sequences]
    print(f"   Actual frames per sequence: {set(seq_lengths)}")
else:
    print("❌ No valid sequences found!")

# Save data
print(f"\n💾 Saving to sequences.pickle...")
f = open('sequences.pickle', 'wb')
pickle.dump({'sequences': sequences, 'labels': labels}, f)
f.close()

print("✅ Processing complete! More data saved! 🎉")