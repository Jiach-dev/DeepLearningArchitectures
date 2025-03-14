import tensorflow as tf
from tensorflow.keras import layers, models

def build_rnn(input_shape, num_classes):
    """Builds a simple RNN with LSTM layers."""
    model = models.Sequential([
        layers.LSTM(128, return_sequences=True, input_shape=input_shape),
        layers.LSTM(64),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    model = build_rnn((100, 1), 10)
    model.summary()
