import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, MultiHeadAttention, Dropout

def transformer_encoder(embed_dim, num_heads, ff_dim):
    """Builds a transformer encoder block."""
    inputs = tf.keras.Input(shape=(None, embed_dim))
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(inputs, inputs)
    attn_output = Dropout(0.1)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)
    
    ffn_output = Dense(ff_dim, activation='relu')(out1)
    ffn_output = Dense(embed_dim)(ffn_output)
    ffn_output = Dropout(0.1)(ffn_output)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
    
    return tf.keras.Model(inputs=inputs, outputs=out2)

if __name__ == "__main__":
    model = transformer_encoder(64, 8, 256)
    model.summary()
