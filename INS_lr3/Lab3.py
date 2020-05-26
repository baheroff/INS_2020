import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import boston_housing
import matplotlib.pyplot as plt

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

print(train_data.shape)
print(test_data.shape)
print(test_targets)

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

k = 5
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []
res = []

for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1, verbose=0, validation_data=(val_data, val_targets))
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)
    res.append(np.mean(all_scores))
    history_dict = history.history
    loss = history_dict['loss']
    mae = history_dict['mean_absolute_error']
    vl_loss = history_dict['val_loss']
    vl_mae = history_dict['val_mean_absolute_error']
    epochs = range(1, num_epochs + 1)

    plt.plot(epochs, loss, 'b')
    plt.plot(epochs, vl_loss, 'r')
    plt.title('Model loss, k = ' + str(i+1))
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Train data', 'Test data'], loc='upper right')
    plt.show()

    plt.plot(epochs, mae, 'b')
    plt.plot(epochs, vl_mae, 'r')
    plt.title('Model mean absolute error, k = ' + str(i+1))
    plt.ylabel('Mean absolute error')
    plt.xlabel('Epochs')
    plt.legend(['Train data', 'Test data'], loc='upper right')
    plt.show()

plt.plot(range(k), res)
plt.title('Dependence on k')
plt.ylabel('Mean')
plt.xlabel('k')
plt.show()

print(np.mean(all_scores))
print(history.history.keys())
