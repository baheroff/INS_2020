import pandas
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as p

dataframe = pandas.read_csv("iris.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = to_categorical(encoded_Y)

model = Sequential()
model.add(Dense(16, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X, dummy_y, epochs=350, batch_size=10, validation_split=0.1)

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']

epochs = range(1, len(loss_values)+1)
p.plot(epochs, loss_values, 'bo', label='Training loss')
p.plot(epochs, val_loss_values, 'b', label='Validation loss')
p.title('Training and validation loss')
p.xlabel('Epochs')
p.ylabel('Loss')
p.legend()
p.show()

p.clf()
p.plot(epochs, acc_values, 'bo', label='Training acc')
p.plot(epochs, val_acc_values, 'b', label='Validation acc')
p.title('Training and validation accuracy')
p.xlabel('Epochs')
p.ylabel('Accuracy')
p.legend()
p.show()

