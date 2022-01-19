# Import necessary libraries
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


# Custom Callback
class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get("accuracy") > 0.99:
            self.model.stop_training = True
            

# https://archive.ics.uci.edu/ml/datasets/banknote+authentication
bank_note = pd.read_csv("Banknote Authentication Dataset.csv")

x = bank_note.drop('Class', axis=1)
y = bank_note['Class']

# Visualize Data Set
fig = plt.figure()
ax = plt.axes(projection='3d')

ax.scatter3D(bank_note["Variance"], bank_note["Skewness"], bank_note["Curtosis"], c=bank_note["Class"], cmap="winter")

ax.set_title("Banknote Authentication Dataset")
ax.set_xlabel("Variance")
ax.set_ylabel("Skewness")
ax.set_zlabel("Curtosis")
plt.show()
plt.clf()

# Split arrays or matrices into random train and test subsets.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Create a model
model = tf.keras.Sequential([tf.keras.layers.Dense(units=16, input_shape=[4], activation="relu"),
                             tf.keras.layers.Dense(units=1),
                             ])


model.compile(optimizer="Adam", loss="binary_crossentropy", metrics="accuracy")

my_callback = MyCallback()
history = model.fit(x_train, y_train, epochs=1000, callbacks=[my_callback], validation_split=0.2)
model.summary()

# Plot what's returned by model.fit()
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["train", "validation"])
plt.show()
plt.clf()

plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='validation')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["train", "validation"])
plt.show()
plt.clf()

# Evaluate Predictions
y_prediction = model.predict(x_test)
y_prediction = np.array(list(map(lambda x: 1 if x > 0.5 else 0, y_prediction)))
print(confusion_matrix(y_test, y_prediction))
print(classification_report(y_test, y_prediction))
