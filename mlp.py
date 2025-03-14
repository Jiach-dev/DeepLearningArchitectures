import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 10 columns = 10 neurons  
# 128 neurons in the hidden layer  
# Fully connected layer  
# Weights = trainable parameters  

### 10 × 128 = 1280 weights  
# Training mechanism = Gradient Descent (updates the weights)  
# z1 ... z128 = w1x1 + w2x2 + ... + w10x10 + bias  

### Input transformation:  
# ℝ⁺ <-- x1 (Input in the positive real domain)  
# z1 <-- ℝ (Mapped to real numbers) → Issue: **Vanishing gradients**  

### Control z1 / Transform into a non-linear activation function  
# Activation functions: Sigmoid / ReLU / Tanh  
# Example: Sigmoid = (exp(x) - exp(-x)) / (exp(x) + exp(-x))  

### Final Layer / Output Layer  
# z_final1 and z_final2 are not probabilities yet  
# Softmax: Converts raw values into probabilities  
# softmax(z_i) = exp(z_i) / sum(exp(z_j)) (Ensures sum of outputs = 1)  

### **ReLU Activation**  
# ReLU(x) = max(0, x)  
# Example: ReLU([-2, 1, 2]) → [0, 1, 2]  

### **Weight Update Rule (Gradient Descent)**  
# W = W - α * ∂L/∂W  
# α = Learning rate  

### **Experimental Setup**  
#neurons = range(0, 1000)  # Number of neurons in a layer  
#layers = 10  # Number of layers  
#activation_f = ["ReLU", "Sigmoid"]  # Activation functions  
#num_classes = len(set(data.target))  # Number of unique classes  


def build_mlp(input_shape, num_classes):
    """Builds a simple Multi-Layer Perceptron (MLP) model."""
    model = keras.Sequential([
        # 10 n
        layers.Dense(128, activation='relu', input_shape=input_shape), # fully connect layer = Dense / all neurone from hi
        # 10X128 - 
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model



if __name__ == "__main__":
    # 1️⃣ Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    ## 28x28 

    #0 255
    #784 = 28X28

    #1000 = training
    # 32 nd = update the weights

    # 2️⃣ Preprocess Data: Flatten images & Normalize to [0, 1]
    x_train = x_train.reshape(-1, 784) / 255.0  # Flatten to (num_samples, 784)
    x_test = x_test.reshape(-1, 784) / 255.0    # Normalize (scale pixel values)

    # 3️⃣ Build Model
    model = build_mlp((784,), 10)
    model.summary()

    # 4️⃣ Train Model
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

    # 5️⃣ Evaluate Model on Test Set
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"\nTest Accuracy: {test_acc:.4f}")
