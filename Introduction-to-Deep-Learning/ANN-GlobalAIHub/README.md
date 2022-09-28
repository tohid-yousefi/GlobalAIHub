# ANN-GlobalAIHub
"ANN" project of "Introduction to Deep Learning" course of "globalaihub.com"

<hr />

# Build an Artificial Neural Network
## Classifying Dates 
In this project, we will build a neural network to classify dates. We'll use the “Date Fruit Dataset” available on Kaggle for this. This dataset includes samples of dates that can be classified into 7 classes according to their types.

## Importing the required libraries
We'll start with importing required libraries.

📌 Use the keywords "import" and "from".

```Python
# Import Pandas and Matplotlib
import pandas as pd
import matplotlib.pyplot as plt

# Import Label Encoder and train_test_split
from sklearn.preprocessing import LabelEncoder,minmax_scale
from sklearn.model_selection import train_test_split
```

## Dataset
Let's load the .xlsx file.

📌 Use the read_excel() function of the Pandas library.

```Python
# Read the "date_fruit.xlsx" file
data = pd.read_excel("/content/drive/MyDrive/Colab Notebooks/date_fruit.xlsx")
```

Next, we take a look at the dataset.

📌 Use the data.head() function.

📌 Use .shape attribute and .unique() methods.

```Python
# Use the head() function to display the first 5 rows of the data
data.head()

# Print the shape of the data and classes
print(data.shape)
print(data["Class"].unique())
```

## Preprocessing

Now that we have a better understanding of our data, let’s split the dataset into features and labels.

📌 Create X and y datasets using .drop() and .loc() methods

```Python
# Create the features dataset
x = data.drop('Class',axis=1)

# Create the labels dataset
y = data.loc[:,'Class']
```

### Feature scaling

Having features in different units or ranges can be problematic in deep learning. We need to scale all of the values between the 0 and 1 range.

📌 Use the minmax_scale() function of the sklearn library.


```Python
# Normalize the features dataset and assign it to a variable
x_scaled = minmax_scale(x)

# Create a DataFrame using the new variable
X = pd.DataFrame(x_scaled)
```

Then, we print the X data again so we can see the difference.

📌 Use the .head() method.

```Python
# Print the newly created DataFrame
X.head()
```

Our features are ready for training. Now, it's time to prepare the labels. 

📌 Print y to take a look at it.

```Python
# Print the y array
y
```

Artificial intelligence algorithms can't use string data when training a model because no mathematical operations can be performed on them. 

📌 Use the LabelEncoder of the sklearn library to converting strings to integers.

```Python
# Create an LabelEncoder object.
encoder = LabelEncoder()

# Convert string classes to integers using fit_transform() method
y = encoder.fit_transform(y)
```

Then, we print y to check the result.

```Python
# Print the y array
y
```

### Splitting

Great, that worked out as we wanted it. Now, we split the dataset into training, validation and test datasets. In general, the ratio for splitting is 80% for training, 10% for validation and 10% for test sets.

📌 Use train_test_split function of the sklearn library.


```Python
# First, create X_train, y_train and X_temporary and y_temporary datasets from X and y.
X_train,X_temporary,y_train,y_temporary = train_test_split(x,y,train_size=0.8)

# Using the X_temporary and y_temporary dataset we just created create validaiton and test datasets.
X_val,X_test,y_val,y_test = train_test_split(X_temporary,y_temporary, train_size=0.5)
```

Let's print the total length of the initial dataset and lengths of the newly created datasets to check our results.

📌 Use the len() function to print the lengths.

```Python
# Print the lengths of the X, X_train, X_val and X_test
print(f"Length of the dataset: {len(X)}")
print(f"Length of the training dataset: {len(X_train)}")
print(f"Length of the validation dataset: {len(X_val)}")
print(f"Length of the test dataset: {len(X_test)}")
```

## Constructing the neural network
And with that, our data is ready to be used in a model. We can move on to the exciting part: constructing a deep learning model. We’ll use TensorFlow for this. To speed up the training time, activate the GPU of Google Colab.

📌 Import TensorFlow

```Python
# Import TensorFlow
import tensorflow as tf
```

Let's start by creating a model object using Sequential API of Keras.

📌 Use tf.keras.Sequential() to create a model object

```Python
# Create a model object
model = tf.keras.Sequential()
```

### Input layer
First, we construct an input layer and assign it to a variable. The first argument is the number of nodes we want in that hidden layer. Only for the input layer, we have to set the input_shape argument which is the number of columns, in this case, 34. For the activation function, we specify “ReLU”.

📌 Use tf.keras.layers.Dense() to create the layer.

📌 Use .add() method of the object to add the layer.

```Python
# Create an input layer
input_layer = tf.keras.layers.Dense(4096,input_shape=(34,),activation='relu')

# Add input layer to model object
model.add(input_layer)
```

### Hidden layers
Next, we need to add the hidden layers. We'll add 4 hidden layers each with 4096 nodes. Again, we specify ReLU as the activation functions and 0.5 dropouts.

📌 Use tf.keras.layers.Dense() to create the layers.

📌 Use .add() method of the object to add the layer.


```Python
# Add the first hidden layer with 4096 nodes and relu activation function
model.add(tf.keras.layers.Dense(4096,activation='relu'))
# Add 0.5 dropout
model.add(tf.keras.layers.Dropout(0.5))

# Add the second hidden layer with 4096 nodes and relu activation function
model.add(tf.keras.layers.Dense(4096,activation='relu'))
# Add 0.5 dropout
model.add(tf.keras.layers.Dropout(0.5))

# Add the third hidden layer with 4096 nodes and relu activation function
model.add(tf.keras.layers.Dense(4096,activation='relu'))
# Add 0.5 dropout
model.add(tf.keras.layers.Dropout(0.5))

# Add the fourth hidden layer with 4096 nodes and relu activation function
model.add(tf.keras.layers.Dense(4096,activation='relu'))
# Add 0.5 dropout
model.add(tf.keras.layers.Dropout(0.5))
```

### Output layer
As the last part of our neural network, we add the output layer. The number of nodes will be equal to the number of target classes which is 7 in our case. We'll use the softmax activation function in the output layer.

```Python
# Add the output layer
model.add(tf.keras.layers.Dense(7,activation="softmax"))
```

### Optimizer
Now we have the structure of our model. To configure the model for training, we'll use the *.compile()* method. Inside the compile method, we have to define the following:
*   "Adam" for optimizer
*   "Sparse Categorical Crossentropy" for the loss function


📌 Construct the model with the .compile() method.

```Python
# Compile the model
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=['accuracy'])
```

## Training the model

It's time to train the model. We'll give the X_train and y_train datasets as the first two arguments. These will be used for training. And with the *validation_data* parameter, we'll give the X_val and y_val as a tuple.

📌 Use .fit() method of the model object for the training.

```Python
# Train the model for 100 epochs 
results = model.fit(X_train,y_train,epochs=100,validation_data=(X_val,y_val))
```

### Visualize the results

After the model is trained, we can create a graph to visualize the change of loss over time. Results are held in:
* results.history["loss"]
* results.history["val_loss"]

📌 Use plt.show() to display the graph.

```Python
# Plot the the training loss
plt.plot(results.history['loss'],label='Train')
# Plot the the validation loss
plt.plot(results.history['val_loss'],label='Test')
# Name the x and y axises
plt.ylabel('Loss')
plt.xlabel('Epoch')

# Put legend table
plt.legend()

# Show the plot
plt.show()
```

## Performance evaluation

Finally, we are going to use the test dataset we created to evaluate the performance of the model.

📌 Use test_on_batch() method with test dataset as parameter

```Python
# Evaluate the performance
test_result = model.test_on_batch(X_test,y_test)
# Print the result
print(test_result)
```
