# Import TensorFlow
import tensorflow as tf 

# Import NumPy and Matplotlib
import numpy as np
import matplotlib.pyplot as plt

# Download the IMDB dataset included in Keras
# Set the parameter num_words to 10000
(X_train,y_train),(X_test,y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)

# Print a sample
print(X_train[0])

# Print the number of samples
print(f"X_train: {len(X_train)}")
print(f"X_test: {len(X_test)}")

# Concatenate X_train and X_test and assing it to a variable X
X = np.concatenate((X_train,X_test),axis=0)

# Concatenate y_train and y_test and assing it to a variable y
y = np.concatenate((y_train,y_test),axis=0)

# Pad all reviews in the X dataset to the length maxlen=1024
X = tf.keras.preprocessing.sequence.pad_sequences(X,maxlen=1024)

# Create the training datasets
X_train = X[:40000]
y_train = y[:40000]

# Create the validation datasets
X_val = X[40000:45000]
y_val = y[40000:45000]

# Create the test datasets
X_test = X[45000:50000]
y_test = y[45000:50000]

# Print the number of samples
print(f"X_train: {len(X_train)}")
print(f"y_train: {len(y_train)}")
print(f"X_val: {len(X_val)}")
print(f"y_val: {len(y_val)}")
print(f"X_test: {len(X_test)}")
print(f"y_test: {len(y_test)}")

# Create model
model = tf.keras.Sequential()

# Add an embedding layer and a dropout
model.add(tf.keras.layers.Embedding(input_dim=10000,output_dim=256))
model.add(tf.keras.layers.Dropout(0.7))

# Add a LSTM layer with dropout
model.add(tf.keras.layers.LSTM(256))
model.add(tf.keras.layers.Dropout(0.7))

# Add a Dense layer with dropout
model.add(tf.keras.layers.Dense(128,activation="relu"))
model.add(tf.keras.layers.Dropout(0.7))

# Add the output layer
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))

# Compile model
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

# Train the model for 5 epochs
results = model.fit(X_train,y_train,epochs=5,validation_data=(X_val,y_val))

# Plot the the training loss
plt.plot(results.history["loss"],label="Train")

# Plot the the validation loss
plt.plot(results.history["val_loss"], label="Validation")

# Name the x and y axises
plt.xlabel("Epoch")
plt.ylabel("Loss")

# Put legend table
plt.legend()

# Show the plot
plt.show()

# Plot the the training accuracy
plt.plot(results.history["accuracy"],label="Train")

# Plot the the validation accuracy
plt.plot(results.history["val_accuracy"],label="Validation")

# Name the x and y axises
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

# Put legend table
plt.legend()

# Show the plot
plt.show()

# Evaluate the performance
model.evaluate(X_test,y_test)

# Make prediction on the reshaped sample
prediction_result = model.predict(X_test[789].reshape(1,1024))
print(f"Label: {y_test[789]} | Prediction: {prediction_result}")
