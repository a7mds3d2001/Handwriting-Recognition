import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess the training data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Build the CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)


# Use the model to classify new images
predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)

# Plot some examples of the model's predictions
for i in range(10):
    plt.imshow(x_test[i], cmap='gray')
    plt.title('Predicted: {}'.format(predicted_labels[i]))
    plt.show()
