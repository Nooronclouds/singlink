from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pickle
import cv2
import mediapipe as mp
import numpy as np
import base64
import json
from datetime import datetime
from ai_transcription import AITranscriber
from tensorflow.keras.models import load_model
from collections import deque

app = FastAPI(title="SignLink API - LSTM Dynamic Detection")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load LSTM model and label encoder
try:
    model = load_model('lstm_model.h5')
    label_dict = pickle.load(open('label_encoder.pickle', 'rb'))
    label_encoder = label_dict['label_encoder']
    print("✅ LSTM model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    label_encoder = None

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, max_num_hands=2)

# AI Transcriber
transcriber = AITranscriber()

# Sequence detection parameters
SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.7
PREDICTION_BUFFER_SIZE = 10

# Word detection buffer to avoid duplicates
class DynamicWordBuffer:
    def __init__(self, stability_threshold=5):
        self.current_word = None
        self.confirmation_count = 0
        self.stability_threshold = stability_threshold
        self.detected_words = []
        self.sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.prediction_buffer = deque(maxlen=PREDICTION_BUFFER_SIZE)
        
    def add_frame(self, landmarks):
        """Add frame landmarks to sequence buffer"""
        self.sequence_buffer.append(landmarks)
        return len(self.sequence_buffer) == SEQUENCE_LENGTH
    
    def predict_sign(self, model, label_encoder):
        """Make prediction from current sequence"""
        if len(self.sequence_buffer) < SEQUENCE_LENGTH:
            return None, 0.0
        
        try:
            sequence_array = np.array(self.sequence_buffer)
            sequence_array = np.expand_dims(sequence_array, axis=0)
            
            prediction = model.predict(sequence_array, verbose=0)
            confidence = np.max(prediction)
            predicted_class = np.argmax(prediction)
            
            if confidence > CONFIDENCE_THRESHOLD:
                predicted_sign = label_encoder.inverse_transform([predicted_class])[0]
                return predicted_sign, confidence
            
            return None, confidence
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0.0
    
    def add_prediction(self, word):
        """Add prediction to buffer for smoothing"""
        if word:
            self.prediction_buffer.append(word)
            
            # Get most frequent prediction
            if len(self.prediction_buffer) >= PREDICTION_BUFFER_SIZE // 2:
                most_common = max(set(self.prediction_buffer), key=self.prediction_buffer.count)
                
                # Check stability
                if most_common == self.current_word:
                    self.confirmation_count += 1
                else:
                    self.current_word = most_common
                    self.confirmation_count = 1
                
                # If stable enough and not already in list (or different from last)
                if self.confirmation_count >= self.stability_threshold:
                    if not self.detected_words or self.detected_words[-1]['word'] != self.current_word:
                        return self.current_word
        
        return None
    
    def add_word(self, word):
        """Add confirmed word to history"""
        word_entry = {
            'word': word,
            'timestamp': datetime.now().strftime('%H:%M:%S')
        }
        self.detected_words.append(word_entry)
        
        # Clear buffers after adding word
        self.prediction_buffer.clear()
        self.confirmation_count = 0
        
        return word_entry
    
    def get_words(self):
        return [w['word'] for w in self.detected_words]
    
    def clear_sequence(self):
        """Clear sequence buffer when hands not detected"""
        self.sequence_buffer.clear()

word_buffer = DynamicWordBuffer()

def process_frame(frame_data):
    """Process a single frame and return detection results"""
    try:
        # Decode base64 image
        img_bytes = base64.b64decode(frame_data.split(',')[1])
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return None
        
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            data_aux = []
            
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
            
            # Should have 84 features now
            if len(data_aux) == 84:
                # Add frame to sequence buffer
                sequence_ready = word_buffer.add_frame(data_aux)
                
                current_word = None
                confidence = 0.0
                
                # Make prediction if sequence is ready
                if sequence_ready and model and label_encoder:
                    current_word, confidence = word_buffer.predict_sign(model, label_encoder)
                    
                    # Add prediction to buffer
                    confirmed_word = word_buffer.add_prediction(current_word)
                    
                    if confirmed_word:
                        return {
                            'detected': True,
                            'word': current_word,
                            'confidence': float(confidence),
                            'bbox': get_bbox(results, W, H),
                            'landmarks': extract_landmarks(results),
                            'newWord': True,
                            'confirmedWord': confirmed_word,
                            'sequenceProgress': len(word_buffer.sequence_buffer)
                        }
                
                # Return current detection (even if not confirmed)
                return {
                    'detected': True,
                    'word': current_word if current_word else "Detecting...",
                    'confidence': float(confidence),
                    'bbox': get_bbox(results, W, H),
                    'landmarks': extract_landmarks(results),
                    'newWord': False,
                    'sequenceProgress': len(word_buffer.sequence_buffer)
                }
        else:
            # No hands detected - clear sequence buffer
            word_buffer.clear_sequence()
        
        return {'detected': False}
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        return None

def get_bbox(results, W, H):
    """Get bounding box from hand landmarks"""
    all_x = []
    all_y = []
    for hand_landmarks in results.multi_hand_landmarks:
        for lm in hand_landmarks.landmark:
            all_x.append(lm.x)
            all_y.append(lm.y)
    
    return {
        'x1': int(min(all_x) * W) - 10,
        'y1': int(min(all_y) * H) - 10,
        'x2': int(max(all_x) * W) + 10,
        'y2': int(max(all_y) * H) + 10
    }

def extract_landmarks(results):
    """Extract hand landmarks for drawing"""
    landmarks = []
    for hand_landmarks in results.multi_hand_landmarks:
        hand_points = []
        for lm in hand_landmarks.landmark:
            hand_points.append({
                'x': lm.x,
                'y': lm.y,
                'z': lm.z
            })
        landmarks.append(hand_points)
    return landmarks

@app.get("/")
async def root():
    return FileResponse('static/index.html')

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection established")
    
    try:
        while True:
            # Receive frame from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message['type'] == 'frame':
                # Process the frame
                result = process_frame(message['data'])
                
                if result and result.get('detected'):
                    # Check if we have a new confirmed word
                    if result.get('newWord'):
                        confirmed_word = result.get('confirmedWord')
                        word_entry = word_buffer.add_word(confirmed_word)
                        
                        # Generate AI transcription
                        words = word_buffer.get_words()
                        ai_interpretation = transcriber.transcribe_signs(words)
                        
                        await websocket.send_json({
                            'type': 'detection',
                            'word': result['word'],
                            'confidence': result['confidence'],
                            'bbox': result['bbox'],
                            'landmarks': result['landmarks'],
                            'newWord': word_entry,
                            'allWords': word_buffer.detected_words,
                            'aiTranscription': ai_interpretation,
                            'wordCount': len(word_buffer.detected_words),
                            'sequenceProgress': result['sequenceProgress']
                        })
                    else:
                        # Still detecting, show progress
                        await websocket.send_json({
                            'type': 'detection',
                            'word': result['word'],
                            'confidence': result.get('confidence', 0),
                            'bbox': result['bbox'],
                            'landmarks': result['landmarks'],
                            'newWord': None,
                            'sequenceProgress': result['sequenceProgress']
                        })
                else:
                    await websocket.send_json({
                        'type': 'no_detection'
                    })
            
            elif message['type'] == 'reset':
                word_buffer.detected_words = []
                word_buffer.sequence_buffer.clear()
                word_buffer.prediction_buffer.clear()
                word_buffer.current_word = None
                word_buffer.confirmation_count = 0
                await websocket.send_json({
                    'type': 'reset_complete'
                })
                
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": "LSTM",
        "sequence_length": SEQUENCE_LENGTH
    }

# Mount static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except:
    print("Warning: static directory not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
