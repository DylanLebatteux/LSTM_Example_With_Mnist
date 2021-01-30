# Imports
import numpy
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow.keras.layers import Dense, Dropout, LSTM

# Get mnist dataset
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Display shape of data
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# Displays the first sample of the dataset
print("\nFirst sample of the dataset| Label:", y_train[0])
for x in X_train[0]: print(x)

# Normalize data between 0 and 1
X_train = X_train / 255.0
X_test = X_test / 255.0

# Displays the first normalized sample of the dataset
print("\nFirst normalized sample of the dataset| Label:", y_train[0])
for x in X_train[0]: print(x)

# Build model
model = Sequential()
model.add(LSTM(128, input_shape = (28, 28), activation = "relu", return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(128, activation = "relu", return_sequences = False))
model.add(Dropout(0.2))
model.add(Dense(32, activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(10, activation = "softmax"))

# Display summary of the model
model.summary()

# Compile the model
model.compile(optimizer = Adam(learning_rate = 0.0001), loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

# Fit the model
model.fit(x = X_train, y = y_train, validation_split = 0.1, epochs = 10, shuffle = True, verbose = 1)

# Predict test data
y_pred = model.predict(x = X_test)

# Convert to rounded prediction
y_pred = numpy.argmax(y_pred, axis = -1)

# Compute confusion matrix
cm = confusion_matrix(y_true = y_test, y_pred=y_pred)

# Set the labels
target_names = ['0','1','2','3','4','5','6','7','8','9']

# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = target_names)
disp = disp.plot()
plt.show()

# Display the classification report
print("\nClassification report:\n", classification_report(y_test, y_pred, target_names = target_names))