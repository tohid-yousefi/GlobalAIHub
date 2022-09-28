# Import TensorFlow
import tensorflow as tf

# Import Numpy and Matplotlib
import numpy as np
import matplotlib.pyplot as plt

# Download the cifar-10 dataset included in Keras
(X_train,y_train),(X_test,y_test) = tf.keras.datasets.cifar10.load_data()

# Print the number of samples 
print(f"X_train: {len(X_train)}")
print(f"X_test: {len(X_test)}")

# Print a sample from X_test dataset
print(X_test[789])

# Use the .imshow() function and show the plot
plt.imshow(X_test[789])
# Print the shape of the sample image
print(X_test[789].shape)

# Create the validation datasets 
# and assign the last 10000 images of X_val and y_val
X_val = X_train[40000:]
y_val = y_train[40000:]

# Create new train datasets
# and assign the first 40000 images of X_train and y_train
X_train = X_train[:40000]
y_train = y_train[:40000]

# Print the lengths of the each dataset
print(f"X_train: {len(X_train)}")
print(f"X_val: {len(X_val)}")
print(f"X_test: {len(X_test)}")

# Divide each dataset by 255
X_train = X_train / 255
X_val = X_val / 255
X_test = X_test / 255

# Create a model object
model = tf.keras.Sequential()

# Add a convolution and max pooling layer
model.add(tf.keras.layers.Conv2D(32,
                                 kernel_size=(3,3),
                                 strides=(1,1),
                                 padding="same",
                                 activation="relu",
                                 input_shape=(32,32,3)))
model.add(tf.keras.layers.MaxPooling2D((2,2)))

# Add more convolution and max pooling layers
model.add(tf.keras.layers.Conv2D(64,
                                 kernel_size=(3,3),
                                 strides=(1,1),
                                 padding="same",
                                 activation="relu",
                                 input_shape=(32,32,3)))
model.add(tf.keras.layers.MaxPooling2D((2,2)))
model.add(tf.keras.layers.Conv2D(64,
                                 kernel_size=(3,3),
                                 strides=(1,1),
                                 padding="same",
                                 activation="relu",
                                 input_shape=(32,32,3)))

# Flatten the convolution layer
model.add(tf.keras.layers.Flatten())

# Add the dense layer and dropout layer
model.add(tf.keras.layers.Dense(64,activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))

# Add the dense layer and dropout layer
model.add(tf.keras.layers.Dense(64,activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))

# Add the output layer
model.add(tf.keras.layers.Dense(10,activation="softmax"))

# Compile the model
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics="accuracy")

# Train the model for 50 epochs with batch size of 128
results = model.fit(X_train,y_train,
                   batch_size=128,
                   epochs=50,
                   validation_data=(X_val,y_val))

# Plot the the training loss
plt.plot(results.history["loss"],label="loss")

# Plot the the validation loss
plt.plot(results.history["val_loss"],label="val_loss")

# Name the x and y axises
plt.xlabel("Epoch")
plt.ylabel("Loss")

# Put legend table
plt.legend()

# Show the plot
plt.show()

# Plot the the training accuracy
plt.plot(results.history["accuracy"],label="accuracy")

# Plot the the validation accuracy
plt.plot(results.history["val_accuracy"],label="val_accuracy")

# Name the x and y axises
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

# Put legend table
plt.legend()

# Show the plot
plt.show()

# Evaluate the performance
model.evaluate(X_test,y_test)

# Evaluate the performance
model.evaluate(X_test,y_test)

# Find the predicted class
predected_class = prediction_result.argmax()
# Find the prediction probability
prediction_probability = prediction_result.max()

# Print the results
print(f"This image belong to class {predected_class} with {prediction_probability} prediction probability")

