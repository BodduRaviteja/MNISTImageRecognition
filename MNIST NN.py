import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
import random
import requests
from PIL import Image
import cv2


url = "https://colah.github.io/posts/2014-10-Visualizing-MNIST/img/mnist_pca/MNIST-p1815-4.png"
response = requests.get(url, stream= True)
print(response)
#in oder to view the raw content of the URL, we need to import the Python Image Lib. (PIL)
img = Image.open(response.raw)
plt.imshow(img)

np.random.seed(0)
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape) # 60K images with each image has size 28x28 Pi
print(X_test.shape) #10k images with each image has size 28x28 Pi
print(y_train.shape[0])

assert (X_train.shape[0] == y_train.shape[0]), 'The number of images is not equal to the no. of lables.'
assert (X_test.shape[0] == y_test.shape[0]), 'The number of images is not equal to the no. of lables.'
assert(X_train.shape[1:] == (28, 28)), 'The dimensions of the image are not 28x28.'
assert(X_test.shape[1:] == (28, 28)), 'The dimensions of the image are not 28x28.'

num_of_samples = []

cols = 5
num_classes = 10
fig, axis = plt.subplots(nrows= num_classes, ncols= cols, figsize = (5,10))
fig.tight_layout() #solves the issue of overlapping of plots
for i in range(cols):
    for j in range(num_classes):
        x_selected = X_train[y_train == j]
        axis[j][i].imshow(x_selected[random.randint(0, len(x_selected-1)), :, :] , cmap = plt.get_cmap(('gray')))
        axis[j][i].axis('off')
        if i == 2:
            axis[j][i].set_title(str(j))
            num_of_samples.append((len(x_selected)))

print(num_of_samples)
#plt.figure(figsize=(12,4))
#plt.bar(range(0, num_classes), num_of_samples)
#plt.title('Distribution of the training dataset')
#plt.xlabel('class number')
#plt.ylabel('Number of images')
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
#Normalization
X_train = X_train/255
X_test = X_test/255
#flattening
num_pixels = 784
X_train = X_train.reshape(X_train.shape[0], num_pixels)
X_test = X_test.reshape(X_test.shape[0], num_pixels)

def create_model():
    model = Sequential()
    model.add(Dense(units= 10, input_shape= (num_pixels,), activation='relu'))
    model.add(Dense(units=10, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))
    model.compile(Adam(lr = 0.01), loss= 'categorical_crossentropy', metrics=['accuracy'])
    return model

model = create_model()
#print(model.summary())
history = model.fit(X_train, y_train, validation_split=0.1, epochs = 10, batch_size=200, verbose=1, shuffle=1)
plt.show()
#print(history)
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.legend(['loss', 'val_loss'])
#plt.xlabel('loss')
#plt.ylabel('epoch')


#To evaluate the score of the test data
score = model.evaluate(X_test, y_test, verbose=0)
print(type(score))
print('Test score is:' , score[0])
print('Test accuracy is:', score[1])

#Prediction
img_array = np.asarray(img) #converts the input data into array
resized = cv2.resize(img_array, (28,28))
gray_scale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
#Actually the NN is traing with the digits with the black background and white foreground
#but the test image is quite opposite
image = cv2.bitwise_not(gray_scale)
plt.imshow(image, cmap= plt.get_cmap('gray'))
#Normalize the test image
image = image/255
image = image.reshape(1, 784)
prediction = np.argmax(model.predict(image))
print('Predicted digit is:', str(prediction))




