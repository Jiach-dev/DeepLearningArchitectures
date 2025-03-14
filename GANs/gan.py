import tensorflow as tf
from tensorflow.keras import layers

def build_gan():
    """Builds a simple Generative Adversarial Network (GAN)."""
    generator = tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(100,)),
        layers.Dense(256, activation='relu'),
        layers.Dense(28 * 28, activation='sigmoid'),
        layers.Reshape((28, 28, 1))
    ])
    
    discriminator = tf.keras.Sequential([
        layers.Flatten(input_shape=(28, 28, 1)),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return generator, discriminator

if __name__ == "__main__":
    generator, discriminator = build_gan()
    generator.summary()
    discriminator.summary()
