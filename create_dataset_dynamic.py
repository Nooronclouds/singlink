import os
import cv2
import numpy as np

DATA_DIR = './dynamic_data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 20
sequences_per_class = 50  # Reduced from 100 images to 50 video sequences
sequence_length = 30      # 30 frames per sequence

# Find the highest existing class folder
existing_classes = [int(d) for d in os.listdir(DATA_DIR) if d.isdigit()]
start_from = max(existing_classes) + 1 if existing_classes else 0

print(f"Starting collection from class {start_from}")

cap = cv2.VideoCapture(0)

for j in range(start_from, number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting DYNAMIC sequences for class {j}')
    print('Perform the complete sign movement when recording!')

    for sequence in range(sequences_per_class):
        sequence_dir = os.path.join(class_dir, str(sequence))
        if not os.path.exists(sequence_dir):
            os.makedirs(sequence_dir)

        print(f'Sequence {sequence+1}/{sequences_per_class} - Get ready!')
        
        # Countdown
        for countdown in [3, 2, 1]:
            ret, frame = cap.read()
            if ret:
                cv2.putText(frame, f'Starting in {countdown}...', (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('frame', frame)
                cv2.waitKey(1000)  # 1 second delay

        # Record sequence
        frames = []
        for frame_num in range(sequence_length):
            ret, frame = cap.read()
            if ret:
                # Show recording indicator
                cv2.putText(frame, 'RECORDING...', (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, f'Frame: {frame_num+1}/{sequence_length}', (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                frames.append(frame)
                cv2.imshow('frame', frame)
                
            # Small delay between frames
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        # Save all frames in this sequence
        for frame_num, frame in enumerate(frames):
            cv2.imwrite(os.path.join(sequence_dir, f'{frame_num}.jpg'), frame)
        
        print(f'✅ Sequence {sequence+1} saved with {len(frames)} frames')

        # Short pause between sequences
        if sequence < sequences_per_class - 1:
            print("Brief pause... get ready for next sequence")
            cv2.waitKey(2000)  # 2 second pause

cap.release()
cv2.destroyAllWindows()
