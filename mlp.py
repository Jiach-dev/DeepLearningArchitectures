import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#input_shape = data.shape[1] = code_contract [ mou, mb , sms, ratios ]

# 10 columns = 10 neurones 
# 128 neurones 
# fully connected 
# weights = parameters 
### 10 X 128 = 1280
# mechanism = gradient descent : update the weights 
# z1 ...z128 = w1x1 ..+++.. x w1010 + bias
### R+ <-- x1
### z1 <-- R == vanishing 
## control z1 / transform into non linear 
## sigmoid / relu / tanh (exp(x)+ exp(-x)/(....)) activation
### final layer / output layer zfinal 1 and 2 but not proba 
### softmax z1/z1+z2+z3 sum() = 1
# ReLu = max(0, x)

#ReLu [-2, 1,2] = [0, 1,2]
# W = W - alpha X DG/Dw
# alpha = learning rate
#Experiment
#neurones = range (0, 1000)
#layers = 10
#ativativation_f = ["RelU, sigm"]
# num_class = data.target.numniqes

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
