from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import optimizers
from PIL import Image
import numpy
import matplotlib.pyplot as plt


def openImage(path):
    return numpy.asarray(Image.open(path))


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

train_images = train_images / 255.0
test_images = test_images / 255.0

model = Sequential()
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))


def compile_fit_print(optimizer, name):
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    H = model.fit(train_images, train_labels, epochs=5, batch_size=128, validation_data=(test_images, test_labels))

    plt.figure(1, figsize=(8, 5))
    plt.title('Training and test accuracy ' + name)
    plt.plot(H.history['accuracy'], 'r', label='train')
    plt.plot(H.history['val_accuracy'], 'b', label='test')
    plt.legend()
    plt.show()
    plt.clf()

    plt.figure(1, figsize=(8, 5))
    plt.title('Training and test loss ' + name)
    plt.plot(H.history['loss'], 'r', label='train')
    plt.plot(H.history['val_loss'], 'b', label='test')
    plt.legend()
    plt.show()
    plt.clf()


compile_fit_print(optimizers.Adam(), 'adam')