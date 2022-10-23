# Neural Networks - GlobalAIHub

📌 **Artificial Neural Networks**, ANNs, simply known as neural networks. They are amongst the most popular models in machine learning and deep learning. ANNs are used in a wide range of applications, from image processing and face recognition, speech recognition and generation, stock market predictions, weather forecasting and many more. They are especially useful for solving nonlinear problems and you’ll see why in a bit.

📌 Did you know that there are more than 80 billion neurons in your brain? They communicate with each other using electrochemical signals that they send through a complex network of multi-layered channels. Even though in many ways the human brain is still a scientific mystery, we know that this biological neural network in your brain evolves every time you experience or learn something new. And this is what the artificial neural network is trying to imitate.

📌 Neural networks have an input layer that takes input data and sends it to the hidden layers. There can be one or more **“hidden layers”** communicating with each other. They send the information to the “output layer”. And finally, the output layer calculates the output. As the name suggests the hidden layers are hidden in the process, the user doesn’t interact with them, but only with the input and output layers. Layers are made up of a number of interconnected nodes that are called **“perceptrons”**. And actually, these perceptrons mimic human neurons.

📌 the data is transferred through the network until it reaches the neurons of the output layer. It computes the probability of each input and chooses the highest one as output. We can compute the outputs for all the inputs in our dataset and compare the predicted outputs with the actual ones. This process of passin information forward through each layer is called **“forward propagation”**. During the forward propagation process, lots of advanced mathematical operations are being computed in between multiple interconnected hidden layers. And if we focus on the basic building block of these neural networks, perceptrons, we can see that they are single layer neural networks, so they have the same data processing principle. Let’s also have a look at the math behind it.

📌 When all the perceptrons between the two layers are connected to each other, we call it a **“fully connected layer”**. 

📌 **“Backpropagation”** that means we go backwards and do the crosscheck to adjust the weights, biases, and minimize our error in our classification task.

📌  Now that we understand how neural networks work, it is time to play with it in code and gain some hands-on experience! We will deal with a classification problem. In this exercise we aim to classify the handwritten digits from their 28 by 28 pixels images. We will use the “Mnist Handwritten Digit” dataset, one of the standard datasets used in computer vision. Computer vision is a field that trains computers to enable them to perceive useful information from visual inputs like images, videos, and so on. And neural networks are one of the best methods to do this. The Mnist dataset contains 70000 samples from 28 by 28 images of handwritten digits from zero to nine. For each pixel there is a grayscale value ranging from white to black. And there are 784 pixels for each image.

📌  Let’s start with importing our dataset using Sklearn.datasets. We use **“feth_openml”** to load a dataset from **“OpenML” (open source dataset platform)**. Then we divide our dataset as features X and targets y. After that, we split them as the train and test datasets. Now, let’s check our dataset.

```Python
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
X = X / 255.0

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3)
```

📌 We create a pandas dataframe and print the first 5 elements using head. As you can see, since these images are 28 by 28 pixels, we have 784 features, one for each pixel. We insert target y as the label at the end of our dataframe. But this numerical data is not very easy to understand like this. We can also display the data as images using the imshow and reshape functions and see the handwritten digits in each image. We will call a neural network classification model from the sklearn.neural_network. The model we need is called MLP, “Multi-Layer Perceptron” Classifier.

```Python
import pandas as pd

data = pd.DataFrame(X)
data.insert(784, "label", y)

data.head()

X_train[2]
```

📌 But this numerical data is not very easy to understand like this. We can also display the data as images using the imshow and reshape functions and see the handwritten digits in each image.

```Python
import matplotlib.pyplot as plt
for i in range(5):
  plt.imshow(X[i].reshape((28, 28)), cmap='gray')
  plt.show()
```

📌 We will call a neural network classification model from the sklearn.neural_network. The model we need is called **MLP, “Multi-Layer Perceptron” Classifier**.

```Python
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=1, activation = "logistic")
mlp1 = MLPClassifier(hidden_layer_sizes=100, activation = "logistic")
mlp2 = MLPClassifier(hidden_layer_sizes=1000, activation = "logistic")
```

📌 In neural networks, the way the model works is like a black box. We can’t predict the exact adjustments that will lead to a better performing model. The best solution is to try different options and compare them in order to choose the best one. In our example, we’ll train three different neural networks with 1, 100 and 1000 hidden layers. By doing that we’ll be able to see how each number of hidden layers affects the final results and find out which one works best for our data. For the activation function let’s select the **“logistic sigmoid function”** – referred to as **“logistic”** in sklearn. When training a neural network, the selection of an activation function is random at first. You select one function, if it works, good if not you select another one and train your model again. After we train our model, we will check its performance metrics to decide if we need to try another activation function.

📌 Now we will train our models using the fit function. When we use the fit function the forward propagation process takes place for all the data points in the train dataset and sets the initial weights and biases. Based on this, the outputs are computed. After the forward propagation, the fit function goes backwards to crosscheck and adjust the weights, biases, and minimize our error. This is the **“Backpropagation”**. With those changes on the weights and bias, the training of the model is complete. 

```Python
mlp.fit(X_train, y_train)
mlp1.fit(X_train, y_train)
mlp2.fit(X_train, y_train)
```

📌 Now we can make predictions by using our trained models on our test dataset. We can see the predictions of the three models for each image in the output arrays. As you can see our three models predicted 0 for the first image in the “X_test” dataset. But when we look at the second image, we see that the first model predicted the number 7, while the other two gave the output 4. 

```Python
predictions_NN = mlp.predict(X_test)
predictions_NN

predictions_NN1 = mlp1.predict(X_test)
predictions_NN1

predictions_NN2 = mlp2.predict(X_test)
predictions_NN2
```

📌 Let’s visualize this image and check which model made the correct prediction. We use the imshow function and put 1 as the index for the second image from “X_test”. We can observe that the prediction of the models with more layers is the correct one in this case. Keep in mind that just because a certain number of hidden layers works in this data, it doesn’t mean that it will work best with every other project.

```Python
print(f"Actual Value: {y_test[0]}")
print(f"Predicted Value: {predictions_NN2[0]}")


plt.imshow(X_test[0].reshape((28, 28)), cmap='gray')
plt.show()

print(f"Actual Value: {y_test[1]}")
print(f"Predicted Value For 1 Hidden Layer: {predictions_NN[1]}")
print(f"Predicted Value For 100 Hidden Layer: {predictions_NN1[1]}")
print(f"Predicted Value For 1000 Hidden Layer: {predictions_NN2[1]}")


plt.imshow(X_test[1].reshape((28, 28)), cmap='gray')
plt.show()
```

📌 Lastly, we can evaluate our model using a **confusion matrix** and a **classification report**. In the confusion matrix, we can see a comparison of our model predictions and actual values for all 10 classes, as you can see from the diagonal line in our confusion matrix our correct predictions for each class increase as we increase the number of hidden layers. From the classification report we can observe precision, recall and f1-score as well as the model accuracy for all classes in general. Based on our results, we can say that the logistic sigmoid function works well, but let’s see how the different numbers of hidden layers affect the performance of the model.

```Python
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, predictions_NN)

from sklearn.metrics import classification_report

print(classification_report(y_test,predictions_NN))
```

```Python
confusion_matrix(y_test, predictions_NN1)

print(classification_report(y_test,predictions_NN1))

confusion_matrix(y_test, predictions_NN2)

print(classification_report(y_test,predictions_NN2))
```

📌 The accuracy results are 29% for 1 hidden layer, 96% for 100 hidden layers and 97% for 1000 hidden layers. We can see that with 1 hidden layer, the model wasn’t able to make good classification but when we train the model with 100 hidden layers accuracy increases a lot. This is because with more hidden layers, the model was able to find better patterns in the data. But having more layers doesn’t always mean we’ll have such a large gap between accuracy results.



