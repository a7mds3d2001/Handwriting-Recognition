# Import the libraries and load the dataset
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten

# The MNIST dataset 
# Contains 60,000 images for training. 
# Contains 10,000 images for testing.
# From Zero to Nine
# The Handwritten Digits Images Are Represented As A 28×28 Matrix Where Each Cell Contains Grayscale Pixel Value.

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)

# Preprocess the data
# The image data cannot be fed directly into the model So we need to perform some operations 
# and process the data to make it ready for our neural network. 
# The dimension of the training data is (60000, 28, 28). 
# The CNN model will require one more dimension so we reshape the matrix to shape (60000, 28, 28, 1).
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

batch_size = 128
num_classes = 10
epochs = 10


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# Create Our CNN Model
# Now we will create our CNN model in Python data science project. 
# A CNN model generally consists of convolutional and pooling layers. 
# It works better for data that are represented as grid structures, 
# this is the reason why CNN works well for image classification problems. 
# The dropout layer is used to deactivate some of the neurons and while training, 
# it reduces offer fitting of the model. 
# We will then compile the model with the Adadelta optimizer.
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Adadelta(), metrics = ['accuracy'])

# Train The Model
# The model.fit() function of Keras will start the training of the model.
# It takes the training data, validation data, epochs, and batch size.
# It takes some time to train the model. 
# After training, we save the weights and model definition in the ‘mnist.h5’ file.
hist = model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, verbose = 1,validation_data = (x_test, y_test))
print("The model has successfully trained")

# Save The Model
model.save('hand_writing_recognition.h5')
print("Saving the model as hand_writing_recognition.h5")

# Evaluate The Model
# We have 10,000 images in our dataset which will be used to evaluate how good our model works.
# The testing data was not involved in the training of the data therefore, 
# it is new data for our model.
# The MNIST dataset is well balanced so we can get around 99% accuracy.
score = model.evaluate(x_test, y_test, verbose = 0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

