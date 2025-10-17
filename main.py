import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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
import time

app = FastAPI(title="SignLink API - Accuracy Optimized")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== LOAD BOTH MODELS =====
try:
    dynamic_model = load_model('lstm_model.h5')
    label_dict = pickle.load(open('label_encoder.pickle', 'rb'))
    dynamic_label_encoder = label_dict['label_encoder']
    print("✅ DYNAMIC LSTM model loaded successfully")
except Exception as e:
    print(f"⚠️ Dynamic model not loaded: {e}")
    dynamic_model = None
    dynamic_label_encoder = None

try:
    static_model_dict = pickle.load(open('./model_enhanced.p', 'rb'))
    static_model = static_model_dict['model']
    print("✅ STATIC Random Forest model loaded successfully")
except Exception as e:
    print(f"⚠️ Static model not loaded: {e}")
    static_model = None

# Static model labels (A-Z + 0-9)
static_labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 
    26: '0', 27: '1', 28: '2', 29: '3', 30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9'
}

# MediaPipe setup - ACCURACY OPTIMIZED
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, 
    min_detection_confidence=0.7,  # Higher = more reliable detection
    min_tracking_confidence=0.7,   # Higher = better tracking quality
    max_num_hands=2
)

# AI Transcriber
transcriber = AITranscriber()

# Accuracy-optimized parameters
SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.85  # Higher threshold for better accuracy
PREDICTION_BUFFER_SIZE = 15  # Larger buffer for better consensus

class DetectionMode:
    STATIC = "static"
    DYNAMIC = "dynamic"

# ===== ACCURACY-OPTIMIZED SESSION MANAGER =====
class DetectionSession:
    def __init__(self):
        self.current_mode = DetectionMode.STATIC
        self.word_buffer = DynamicWordBuffer()
        self.static_buffer = StaticDetectionBuffer()
        self.last_process_time = 0
        self.frame_skip = 0

class DynamicWordBuffer:
    def __init__(self, stability_threshold=8):  # Increased for stability
        self.current_word = None
        self.confirmation_count = 0
        self.stability_threshold = stability_threshold
        self.detected_words = []
        self.sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.prediction_buffer = deque(maxlen=PREDICTION_BUFFER_SIZE)
        
    def add_frame(self, landmarks):
        self.sequence_buffer.append(landmarks)
        return len(self.sequence_buffer) == SEQUENCE_LENGTH
    
    def predict_sign(self, model, label_encoder):
        if len(self.sequence_buffer) < SEQUENCE_LENGTH:
            return None, 0.0
        
        try:
            sequence_array = np.array(self.sequence_buffer, dtype=np.float32)
            sequence_array = np.expand_dims(sequence_array, axis=0)
            
            prediction = model.predict(sequence_array, verbose=0)
            confidence = np.max(prediction)
            predicted_class = np.argmax(prediction)
            
            if confidence > CONFIDENCE_THRESHOLD:
                predicted_sign = label_encoder.inverse_transform([predicted_class])[0]
                return predicted_sign, confidence
            
            return None, confidence
        except Exception as e:
            print(f"Dynamic prediction error: {e}")
            return None, 0.0
    
    def get_smoothed_prediction(self):
        """Use majority voting across multiple predictions for better accuracy"""
        if len(self.prediction_buffer) == 0:
            return None
        
        # Count occurrences of each prediction
        prediction_counts = {}
        for pred in self.prediction_buffer:
            prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
        
        # Get the most frequent prediction
        most_common = max(prediction_counts.items(), key=lambda x: x[1])
        
        # Only return if it appears significantly more than others
        if most_common[1] >= len(self.prediction_buffer) * 0.6:  # 60% consensus
            return most_common[0]
        
        return None
    
    def add_prediction(self, word):
        if word:
            self.prediction_buffer.append(word)
            
            # Use smoothed prediction for better accuracy
            smoothed_word = self.get_smoothed_prediction()
            
            if smoothed_word:
                if smoothed_word == self.current_word:
                    self.confirmation_count += 1
                else:
                    self.current_word = smoothed_word
                    self.confirmation_count = 1
                
                if self.confirmation_count >= self.stability_threshold:
                    if not self.detected_words or self.detected_words[-1]['word'] != self.current_word:
                        return self.current_word
        return None
    
    def add_word(self, word):
        word_entry = {
            'word': str(word),
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'type': 'dynamic'
        }
        self.detected_words.append(word_entry)
        self.prediction_buffer.clear()
        self.confirmation_count = 0
        return word_entry
    
    def get_words(self):
        return [w['word'] for w in self.detected_words]
    
    def clear_sequence(self):
        self.sequence_buffer.clear()

# ===== ACCURACY-OPTIMIZED STATIC DETECTION =====
class StaticDetectionBuffer:
    def __init__(self, stability_threshold=4):  # Increased for stability
        self.current_char = None
        self.confirmation_count = 0
        self.stability_threshold = stability_threshold
        self.detected_chars = []
        self.prediction_buffer = deque(maxlen=5)  # Larger buffer
        
    def get_smoothed_prediction(self):
        """Use majority voting for static predictions"""
        if len(self.prediction_buffer) == 0:
            return None
        
        prediction_counts = {}
        for pred in self.prediction_buffer:
            prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
        
        most_common = max(prediction_counts.items(), key=lambda x: x[1])
        
        if most_common[1] >= len(self.prediction_buffer) * 0.6:  # 60% consensus
            return most_common[0]
        
        return None
    
    def add_prediction(self, char):
        if char:
            self.prediction_buffer.append(char)
            
            # Use smoothed prediction for better accuracy
            smoothed_char = self.get_smoothed_prediction()
            
            if smoothed_char:
                if smoothed_char == self.current_char:
                    self.confirmation_count += 1
                else:
                    self.current_char = smoothed_char
                    self.confirmation_count = 1
                
                if self.confirmation_count >= self.stability_threshold:
                    if not self.detected_chars or self.detected_chars[-1]['char'] != self.current_char:
                        return self.current_char
        return None
    
    def add_char(self, char):
        char_entry = {
            'char': str(char),
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'type': 'static'
        }
        self.detected_chars.append(char_entry)
        self.prediction_buffer.clear()
        self.confirmation_count = 0
        return char_entry
    
    def get_text(self):
        return ''.join([c['char'] for c in self.detected_chars])
    
    def clear(self):
        self.prediction_buffer.clear()
        self.confirmation_count = 0

# ===== ACCURACY-OPTIMIZED FRAME PROCESSING =====
def calculate_hand_visibility(results):
    """Calculate how clearly hands are visible in frame for quality control"""
    if not results.multi_hand_landmarks:
        return 0.0
    
    total_confidence = 0.0
    for hand_landmarks in results.multi_hand_landmarks:
        # Use landmark positions to estimate visibility quality
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]
        
        # Hands near edges or too small might be low quality
        span_x = max(x_coords) - min(x_coords)
        span_y = max(y_coords) - min(y_coords)
        
        # Quality scoring based on hand size and position
        if span_x > 0.15 and span_y > 0.15:  # Good hand size
            total_confidence += 0.8
        elif span_x > 0.1 and span_y > 0.1:  # Acceptable hand size
            total_confidence += 0.5
        else:
            total_confidence += 0.2  # Penalize small/edge hands
    
    return total_confidence / len(results.multi_hand_landmarks)

def validate_prediction(prediction, confidence, previous_predictions):
    """Validate prediction against recent history for consistency"""
    if confidence < CONFIDENCE_THRESHOLD:
        return False
    
    # Check if this prediction makes sense contextually
    if previous_predictions and len(previous_predictions) > 2:
        recent_avg_confidence = np.mean([p['confidence'] for p in previous_predictions[-3:]])
        if confidence < recent_avg_confidence * 0.7:  # Significant confidence drop
            return False
    
    return True

def process_frame(frame_data, session: DetectionSession):
    """ACCURACY-OPTIMIZED: Process frame with quality checks"""
    try:
        # Decode image
        img_bytes = base64.b64decode(frame_data.split(',')[1])
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return None
        
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # MediaPipe detection
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            # Quality check: reject poorly visible hands
            hand_visibility = calculate_hand_visibility(results)
            if hand_visibility < 0.5:  # Reject low-quality hand detection
                return {'detected': False, 'reason': 'low_visibility'}
            
            data_aux = []
            
            # Extract features
            for hand_index in range(2):
                if hand_index < len(results.multi_hand_landmarks):
                    hand_landmarks = results.multi_hand_landmarks[hand_index]
                    hand_x = [lm.x for lm in hand_landmarks.landmark]
                    hand_y = [lm.y for lm in hand_landmarks.landmark]
                    
                    min_x, min_y = min(hand_x), min(hand_y)
                    
                    for lm in hand_landmarks.landmark:
                        data_aux.append(lm.x - min_x)
                        data_aux.append(lm.y - min_y)
                else:
                    data_aux.extend([0.0] * 42)
            
            if len(data_aux) == 84:
                bbox = get_bbox(results, W, H)
                landmarks = extract_landmarks(results)
                
                # STATIC MODE - ACCURACY OPTIMIZED
                if session.current_mode == DetectionMode.STATIC and static_model:
                    try:
                        data_array = np.asarray(data_aux, dtype=np.float32).reshape(1, -1)
                        prediction = static_model.predict(data_array)
                        predicted_class = int(prediction[0])
                        predicted_char = static_labels_dict.get(predicted_class, '?')
                        
                        confirmed_char = session.static_buffer.add_prediction(predicted_char)
                        
                        if confirmed_char:
                            return {
                                'detected': True,
                                'content': str(predicted_char),
                                'confidence': 0.95,
                                'bbox': bbox,
                                'landmarks': landmarks,
                                'newDetection': True,
                                'confirmedContent': str(confirmed_char),
                                'progress': 100,
                                'mode': 'static',
                                'type': 'character'
                            }
                        
                        return {
                            'detected': True,
                            'content': str(predicted_char),
                            'confidence': 0.95,
                            'bbox': bbox,
                            'landmarks': landmarks,
                            'newDetection': False,
                            'progress': 100,
                            'mode': 'static',
                            'type': 'character'
                        }
                        
                    except Exception as e:
                        print(f"Static prediction error: {e}")
                        return {'detected': True, 'content': 'Error', 'confidence': 0, 'bbox': bbox, 
                                'landmarks': landmarks, 'newDetection': False, 'progress': 100, 
                                'mode': 'static', 'type': 'character'}
                
                # DYNAMIC MODE - ACCURACY OPTIMIZED
                elif session.current_mode == DetectionMode.DYNAMIC and dynamic_model and dynamic_label_encoder:
                    sequence_ready = session.word_buffer.add_frame(data_aux)
                    
                    current_word = None
                    confidence = 0.0
                    
                    if sequence_ready:
                        current_word, confidence = session.word_buffer.predict_sign(dynamic_model, dynamic_label_encoder)
                        
                        # Additional validation for dynamic predictions
                        if current_word and confidence > CONFIDENCE_THRESHOLD:
                            confirmed_word = session.word_buffer.add_prediction(current_word)
                            
                            if confirmed_word:
                                return {
                                    'detected': True,
                                    'content': str(current_word),
                                    'confidence': float(confidence),
                                    'bbox': bbox,
                                    'landmarks': landmarks,
                                    'newDetection': True,
                                    'confirmedContent': str(confirmed_word),
                                    'progress': len(session.word_buffer.sequence_buffer),
                                    'mode': 'dynamic',
                                    'type': 'word'
                                }
                    
                    return {
                        'detected': True,
                        'content': str(current_word) if current_word else "Detecting...",
                        'confidence': float(confidence),
                        'bbox': bbox,
                        'landmarks': landmarks,
                        'newDetection': False,
                        'progress': len(session.word_buffer.sequence_buffer),
                        'mode': 'dynamic',
                        'type': 'word'
                    }
        else:
            # No hands - clear buffers
            if session.current_mode == DetectionMode.DYNAMIC:
                session.word_buffer.clear_sequence()
            else:
                session.static_buffer.clear()
        
        return {'detected': False}
        
    except Exception as e:
        print(f"Frame processing error: {e}")
        return None

def get_bbox(results, W, H):
    """Extract bounding box"""
    all_x = [lm.x for hand in results.multi_hand_landmarks for lm in hand.landmark]
    all_y = [lm.y for hand in results.multi_hand_landmarks for lm in hand.landmark]
    
    return {
        'x1': int(min(all_x) * W) - 10,
        'y1': int(min(all_y) * H) - 10,
        'x2': int(max(all_x) * W) + 10,
        'y2': int(max(all_y) * H) + 10
    }

def extract_landmarks(results):
    """Extract landmarks for visualization"""
    return [[{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in hand.landmark] 
            for hand in results.multi_hand_landmarks]

@app.get("/")
async def root():
    return FileResponse('static/index.html')

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ WebSocket connected - Accuracy Optimized Mode")
    
    session = DetectionSession()
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message['type'] == 'frame':
                # Process frame
                result = process_frame(message['data'], session)
                
                if result and result.get('detected'):
                    if result.get('newDetection'):
                        confirmed_content = result.get('confirmedContent')
                        
                        if session.current_mode == DetectionMode.DYNAMIC:
                            detection_entry = session.word_buffer.add_word(confirmed_content)
                            words = session.word_buffer.get_words()
                            ai_interpretation = transcriber.transcribe_signs(words)
                            
                            await websocket.send_json({
                                'type': 'detection',
                                'content': result['content'],
                                'confidence': result['confidence'],
                                'bbox': result['bbox'],
                                'landmarks': result['landmarks'],
                                'newDetection': detection_entry,
                                'allDetections': session.word_buffer.detected_words,
                                'aiTranscription': ai_interpretation,
                                'detectionCount': len(session.word_buffer.detected_words),
                                'progress': result['progress'],
                                'mode': result['mode'],
                                'detectionType': result['type']
                            })
                        else:
                            detection_entry = session.static_buffer.add_char(confirmed_content)
                            current_text = session.static_buffer.get_text()
                            
                            await websocket.send_json({
                                'type': 'detection',
                                'content': result['content'],
                                'confidence': result['confidence'],
                                'bbox': result['bbox'],
                                'landmarks': result['landmarks'],
                                'newDetection': detection_entry,
                                'allDetections': session.static_buffer.detected_chars,
                                'currentText': current_text,
                                'detectionCount': len(session.static_buffer.detected_chars),
                                'progress': result['progress'],
                                'mode': result['mode'],
                                'detectionType': result['type']
                            })
                    else:
                        await websocket.send_json({
                            'type': 'detection',
                            'content': result['content'],
                            'confidence': result.get('confidence', 0),
                            'bbox': result['bbox'],
                            'landmarks': result['landmarks'],
                            'newDetection': None,
                            'progress': result['progress'],
                            'mode': result['mode'],
                            'detectionType': result['type']
                        })
                else:
                    await websocket.send_json({'type': 'no_detection', 'mode': session.current_mode})
            
            elif message['type'] == 'reset':
                session.word_buffer.detected_words = []
                session.word_buffer.sequence_buffer.clear()
                session.word_buffer.prediction_buffer.clear()
                session.word_buffer.current_word = None
                session.word_buffer.confirmation_count = 0
                
                session.static_buffer.detected_chars = []
                session.static_buffer.prediction_buffer.clear()
                session.static_buffer.current_char = None
                session.static_buffer.confirmation_count = 0
                
                await websocket.send_json({'type': 'reset_complete'})
            
            elif message['type'] == 'switch_mode':
                new_mode = message.get('mode', 'static')
                if new_mode in ['static', 'dynamic']:
                    session.current_mode = new_mode
                    print(f"🔄 Switched to {session.current_mode} mode")
                    await websocket.send_json({'type': 'mode_switched', 'newMode': session.current_mode})
                
    except WebSocketDisconnect:
        print("❌ WebSocket disconnected")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "static_model_loaded": static_model is not None,
        "dynamic_model_loaded": dynamic_model is not None,
        "static_model_type": "Random Forest",
        "dynamic_model_type": "LSTM",
        "mode": "Accuracy Optimized",
        "confidence_threshold": CONFIDENCE_THRESHOLD
    }

try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except:
    print("⚠️ Static directory not found")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting SignLink server - ACCURACY OPTIMIZED MODE...")
    print(f"🎯 Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    print("⚡ Prioritizing accuracy over speed")
    uvicorn.run(app, host="127.0.0.1", port=8000)
