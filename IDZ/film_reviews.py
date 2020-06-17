# Подключение необходимых библиотек
import numpy as np
import pandas
from keras import Sequential
from keras.datasets import imdb
import matplotlib.pyplot as plt
from keras.layers import Embedding, Conv1D, Conv2D, Dropout, MaxPooling1D, LSTM, Dense, Flatten, SpatialDropout1D
from keras_preprocessing import sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import one_hot
from tensorflow.keras.utils import to_categorical
# Определение констант
MAX_REVIEW_LENGTH = 400
VOCAB_SIZE = 61000
NUM = 100
BORDER = 560000

# Личный отзыв с оценкой 1
my_review_1 = ['Product arrived labeled as Jumbo Salted Peanuts...the peanuts were actually small sized unsalted. '
               'Not sure if this was an error or if the vendor intended to represent the product as ""Jumbo"".']

# Получение данных
dataframe = pandas.read_csv("reviews.csv", header=0)
dataset = dataframe.values
reviews = dataset[::NUM][:,9]
#rates = dataset[::NUM][:,6]

# Разделение данных на выборки
train_reviews = dataset[:BORDER:NUM][:,9]
test_reviews = dataset[BORDER::NUM][:,9]
train_rates = dataset[:BORDER:NUM][:,6]
test_rates = dataset[BORDER::NUM][:,6]

# Преобразование выходных данных
train_rates = to_categorical(train_rates)
train_rates = train_rates[:,1:]
test_rates = to_categorical(test_rates)
test_rates = test_rates[:,1:]

#print(reviews)
#print(rates)

# Вывод средней длины отзывов
length = [len(i) for i in reviews]
print("Average Review length:", np.mean(length))

# Функция кодировки и органичения отзывов
def encode_review(rev):
    rev = [one_hot(d, VOCAB_SIZE) for d in rev]
    padded_reviews = sequence.pad_sequences(rev, MAX_REVIEW_LENGTH)
    return padded_reviews

# Преобразование выходных данных
encoded_reviews = encode_review(reviews)
encoded_train_reviews = encode_review(train_reviews)
encoded_test_reviews = encode_review(test_reviews)

# Вывод кол-ва уникальных слов
print("Number of unique words:", len(np.unique(np.hstack(encoded_reviews))))

# Архитектура модели
model = Sequential()
model.add(Embedding(VOCAB_SIZE, 32, input_length=MAX_REVIEW_LENGTH))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))
model.add(LSTM(100))
model.add(Dropout(0.3))
model.add(Dense(5, activation='sigmoid'))

# Компиляция модели
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Обучение модели
H = model.fit(encoded_train_reviews, train_rates, batch_size=16, epochs=7, verbose=1, validation_data=(encoded_test_reviews, test_rates))

# Вывод ошибок и точности
acc = model.evaluate(encoded_test_reviews, test_rates)
print('Test', acc)

# Фукнция построения графика ошибок
def plot_loss(loss, v_loss):
    plt.figure(1, figsize=(8, 5))
    plt.plot(loss, 'b', label='Train')
    plt.plot(v_loss, 'r', label='Validation')
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()
    plt.clf()

# Фукнция построения графика точности
def plot_acc(acc, val_acc):
    plt.plot(acc, 'b', label='Train')
    plt.plot(val_acc, 'r', label='Validation')
    plt.title('Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()
    plt.clf()

# Построение графиков
plot_loss(H.history['loss'], H.history['val_loss'])
plot_acc(H.history['accuracy'], H.history['val_accuracy'])

# Оценка моделью личного отзыва
encoded_my_review_1 = encode_review(my_review_1)
pred_1 = model.predict(encoded_my_review_1)
pred1 = np.array(pred_1)
print('Prediction of my review:', pred1)