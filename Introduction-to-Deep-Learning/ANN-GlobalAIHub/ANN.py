# Import Pandas and Matplotlib
import pandas as pd
import matplotlib.pyplot as plt

# Import Label Encoder and train_test_split
from sklearn.preprocessing import LabelEncoder,minmax_scale
from sklearn.model_selection import train_test_split

# Read the "date_fruit.xlsx" file
data = pd.read_excel("/content/drive/MyDrive/Colab Notebooks/date_fruit.xlsx")

# Use the head() function to display the first 5 rows of the data
data.head()

# Print the shape of the data and classes
print(data.shape)
print(data["Class"].unique())

# Create the features dataset
x = data.drop('Class',axis=1)

# Create the labels dataset
y = data.loc[:,'Class']

# Normalize the features dataset and assign it to a variable
x_scaled = minmax_scale(x)

# Create a DataFrame using the new variable
X = pd.DataFrame(x_scaled)

# Print the newly created DataFrame
X.head()

# Print the y array
y

# Create an LabelEncoder object.
encoder = LabelEncoder()

# Convert string classes to integers using fit_transform() method
y = encoder.fit_transform(y)

# Print the y array
y

# First, create X_train, y_train and X_temporary and y_temporary datasets from X and y.
X_train,X_temporary,y_train,y_temporary = train_test_split(x,y,train_size=0.8)

# Using the X_temporary and y_temporary dataset we just created create validaiton and test datasets.
X_val,X_test,y_val,y_test = train_test_split(X_temporary,y_temporary, train_size=0.5)

# Print the lengths of the X, X_train, X_val and X_test
print(f"Length of the dataset: {len(X)}")
print(f"Length of the training dataset: {len(X_train)}")
print(f"Length of the validation dataset: {len(X_val)}")
print(f"Length of the test dataset: {len(X_test)}")

# Import TensorFlow
import tensorflow as tf

# Create a model object
model = tf.keras.Sequential()

# Create an input layer
input_layer = tf.keras.layers.Dense(4096,input_shape=(34,),activation='relu')

# Add input layer to model object
model.add(input_layer)

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

# Add the output layer
model.add(tf.keras.layers.Dense(7,activation="softmax"))

# Compile the model
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=['accuracy'])

# Train the model for 100 epochs 
results = model.fit(X_train,y_train,epochs=100,validation_data=(X_val,y_val))

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

# Evaluate the performance
test_result = model.test_on_batch(X_test,y_test)
# Print the result
print(test_result)
