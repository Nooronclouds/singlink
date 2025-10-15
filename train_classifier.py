import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# Load sequences data
sequences_dict = pickle.load(open('./sequences.pickle', 'rb'))

sequences = np.asarray(sequences_dict['sequences'])
labels = np.asarray(sequences_dict['labels'])

print(f"Dataset shape: {sequences.shape}")  # Should be (n_sequences, 30, 84)
print(f"Number of sequences: {len(sequences)}")
print(f"Classes: {set(labels)}")

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)
y_categorical = to_categorical(y_encoded)

print(f"Encoded labels shape: {y_categorical.shape}")

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    sequences, y_categorical, test_size=0.2, shuffle=True, stratify=y_encoded
)

print(f"Training data: {x_train.shape}")
print(f"Testing data: {x_test.shape}")

# Build LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', 
         input_shape=(x_train.shape[1], x_train.shape[2])),
    Dropout(0.2),
    
    LSTM(128, return_sequences=True, activation='relu'),
    Dropout(0.2),
    
    LSTM(64, return_sequences=False, activation='relu'),
    Dropout(0.2),
    
    Dense(64, activation='relu'),
    Dropout(0.2),
    
    Dense(32, activation='relu'),
    Dense(y_categorical.shape[1], activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

# Train with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    x_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(x_test, y_test),
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f'\n🎯 Final Test Accuracy: {test_accuracy * 100:.2f}%')

# Save model and metadata
model.save('lstm_model.h5')

# Save label encoder for inference
with open('label_encoder.pickle', 'wb') as f:
    pickle.dump({'label_encoder': label_encoder}, f)

print("✅ Model saved as 'lstm_model.h5'")
print("✅ Label encoder saved as 'label_encoder.pickle'")
