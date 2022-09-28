# RCNN-GlobalAIHub
"RCNN" project of "Introduction to Deep Learning" course of "globalaihub.com"

<hr />

# Building a Recurrent Neural Network

## Sentiment Analysis
In this project, we will build a Long Short-term Memory (LSTM) neural network to solve a binary sentiment analysis problem.

For this, we'll use the “IMDB Movie Review Dataset" available on Keras. It includes 50000 highly polarized movie reviews categorized as positive or negative.

## Importing the required libraries
We'll start with importing required libraries.

📌 Use the keyword "import".

```Python
# Import TensorFlow
import tensorflow as tf 

# Import NumPy and Matplotlib
import numpy as np
import matplotlib.pyplot as plt
```

## Dataset
Let's download the IMDB dataset which is included in Keras, and assign it to the corresponding variables *X_train*, *y_train*, *X_test*, and *y_test*. We want to include the most frequently used 10000 words, so we specify 10000 for the num_words parameter.

📌 Use the datasets.imdb.load_data() function of the Keras.

```Python
# Download the IMDB dataset included in Keras
# Set the parameter num_words to 10000
(X_train,y_train),(X_test,y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)
```

Before we move on, we can print a single sample to see what the data looks like.

📌 Use the print() function for this.

```Python
# Print a sample
print(X_train[0])
```

Then, we print the the number of samples in the X_train and X_test datasets to see how the dataset is distributed.

📌 Use f-strings for this.

```Python
# Print the number of samples
print(f"X_train: {len(X_train)}")
print(f"X_test: {len(X_test)}")
```

# Preprocessing
### Concatenate

To split the dataset with 80-10-10 ratio, we'll first concatenate train and test datasets to create one big dataset.

📌 Use contenate() function of the NumPy library for this.

```Python
# Concatenate X_train and X_test and assing it to a variable X
X = np.concatenate((X_train,X_test),axis=0)

# Concatenate y_train and y_test and assing it to a variable y
y = np.concatenate((y_train,y_test),axis=0)
```

### Padding

Since all reviews are at different lengths, we'll use padding to make all of them same length.

📌 Use preprocessing.sequence.pad_sequences() function for this.

```Python
# Pad all reviews in the X dataset to the length maxlen=1024
X = tf.keras.preprocessing.sequence.pad_sequences(X,maxlen=1024)
```

### Splitting

Now, split X and y into train, validation and test dataset and assign those to corresponding values.

📌 You can use list slicing methods for this.

📌 For this dataset, a 80-10-10 split corresponds to 40000 - 10000 - 10000 number of samples relatively.


```Python
# Create the training datasets
X_train = X[:40000]
y_train = y[:40000]

# Create the validation datasets
X_val = X[40000:45000]
y_val = y[40000:45000]

# Create the test datasets
X_test = X[45000:50000]
y_test = y[45000:50000]
```

To check if that worked out, print the number of samples in each dataset again.

📌 Use f-strings for this.

```Python
# Print the number of samples
print(f"X_train: {len(X_train)}")
print(f"y_train: {len(y_train)}")
print(f"X_val: {len(X_val)}")
print(f"y_val: {len(y_val)}")
print(f"X_test: {len(X_test)}")
print(f"y_test: {len(y_test)}")
```

## Constructing the neural network

That was it for the preprocessing of the data! 

Now we can create our model. First, we start by creating a model object using the Sequential API of Keras.

📌 Use tf.keras.Sequential() to create a model object

```Python
# Create model
model = tf.keras.Sequential()
```

### Embedding Layer

For the first layer, we add an embedding layer.

📌 Use tf.keras.layers.Embedding() for the embedding layer.

📌 Use .add() method of the object to add the layer.

```Python
# Add an embedding layer and a dropout
model.add(tf.keras.layers.Embedding(input_dim=10000,output_dim=256))
model.add(tf.keras.layers.Dropout(0.7))
```

Then, we add a LSTM layer and a dense layer; each with a dropout.

📌 Use tf.keras.layers.LSTM() and tf.keras.layers.Dense() to create the layers.

📌 Use .add() method of the object to add the layer.

```Python
# Add a LSTM layer with dropout
model.add(tf.keras.layers.LSTM(256))
model.add(tf.keras.layers.Dropout(0.7))

# Add a Dense layer with dropout
model.add(tf.keras.layers.Dense(128,activation="relu"))
model.add(tf.keras.layers.Dropout(0.7))
```

### Output layer

As the last part of our neural network, we add the output layer. The number of nodes will be one since we are making binary classification. We'll use the sigmoid activation function in the output layer.

📌 Use tf.keras.layers.Dense() to create the layer.

📌 Use .add() method of the object to add the layer.

```Python
# Add the output layer
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))
```

### Optimizer

Now we have the structure of our model. To configure the model for training, we'll use the *.compile()* method. Inside the compile method, we have to define the following:
*   "Adam" for optimizer
*   "Binary Crossentropy" for the loss function


📌 Construct the model with the .compile() method.

```Python
# Compile model
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
```

## Training the model

It's time to train the model. We'll give the X_train and y_train datasets as the first two arguments. These will be used for training. And with the *validation_data* parameter, we'll give the X_val and y_val as a tuple.

📌 Use .fit() method of the model object for the training.

```Python
# Train the model for 5 epochs
results = model.fit(X_train,y_train,epochs=5,validation_data=(X_val,y_val))
```

### Visualize the results

After the model is trained, we can create a graph to visualize the change of loss over time. Results are held in:
* results.history["loss"]
* results.history["val_loss"]

📌 Use plt.show() to display the graph.

```Python
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
```

Now, do the same thing for accuracy.

📌 Accuracy scores can be found in:
* results.history["accuracy"]
* results.history["val_accuracy"]

```Python
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
```

## Performance evaluation

Let's use the test dataset that we created to evaluate the performance of the model.

📌 Use test_on_batch() method with test dataset as parameter.

```Python
# Evaluate the performance
model.evaluate(X_test,y_test)
```

### Try a prediction

Next, we take a sample and make a prediction on it.

📌 Reshape the review to (1, 1024).

📌 Use the .prediction() method of the model object.

```Python
# Make prediction on the reshaped sample
prediction_result = model.predict(X_test[789].reshape(1,1024))
print(f"Label: {y_test[789]} | Prediction: {prediction_result}")
```
