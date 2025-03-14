import tensorflow as tf
from tensorflow.keras import layers, models

def build_autoencoder(input_shape):
    """Builds an autoencoder."""
    encoder = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu')
    ])
    
    decoder = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(32,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(input_shape, activation='sigmoid')
    ])
    
    autoencoder = models.Sequential([encoder, decoder])
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

if __name__ == "__main__":
    model = build_autoencoder(784)
    model.summary()
