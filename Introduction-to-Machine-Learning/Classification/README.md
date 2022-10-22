# Classification - GlobalAIHub

📌 **Classification** is the process of categorizing a given set of data into classes. Here, the classes act as our labels, or ground truth. A classification model uses the features of an object to predict its labels. As we say labels, you can probably guess that here we have another **supervised learning** model at hand. The algorithm used by your email service providers to filter spam from non-spam emails is an example of classification. This model uses the features of the email: subject, sender’s email address, email body, and attachments as inputs; and makes a prediction for one out of the two classes: spam or non-spam. This is an example of *binary classification*, where the output is restricted to two classes. Spam and non-spam, true and false, zeros and ones, yes and no, positive, or negative and so on. If there are more than two classes, we have a **multi-class classification** problem. An example of multi-class classification can be classifying types of fruits based on their color, weight, and size. Or movies into different genres like comedy, romance, drama, and horror.

📌 The question is, how can machine learning solve this problem? Let’s start with our first classification model: **Logistic regression**. The best way to think about logistic regression is that it is a *linear regression* but for classification problems. Logistic regression uses a logistic function, specifically the **Sigmoid function**. This function takes any real input, and outputs a value between zero and one. Unlike linear regression, logistic regression doesn’t need a linear relationship between input and output variables. Once we have the predicted results from our classification model, or classifier, we compare these results with the actual label, the ground truth, and evaluate the performance of our model.

📌 We have a binary classification problem. We need to classify tumors into malignant ones that are cancerous or benign which means non-cancerous. Our dataset contains statistical data from histopathology examinations. We will use this dataset to train our logistic regression model. Let’s import pandas and read the dataset. Using the shape method, we can easily get the number of observations and features. In this dataset, there are unique instances. The features are radius mean, texture mean, radius worst, and so on. There are features in total. We can check the first 5 instances in our dataset using the head function. Here, we can see, the first column represents the target variable. The following columns are the features.

```Python
import pandas as pd
dataset = pd.read_csv("breast-cancer.csv")

dataset.shape

dataset.head()

dataset.tail()
```

📌 This is a real-life dataset and before we can apply machine learning algorithms to it, it has to be cleaned and organized. Since we know that machines operate with numbers, we need to convert our target variable from categorical to numerical type. There are many ways to do this. One of the simpler methods is called **label encoding**. With this, we can convert “M” and “B” to 1 and 0. As a first step, we import **LabelEncoder** from **sklearn** library. Then, to make it easier to use, we assign LabelEncoder to the “labelencoder” variable. Finally, we convert the “diagnosis” column from categoric to numeric with the code in the third line.

```Python
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
dataset["diagnosis"] = labelencoder.fit_transform(dataset["diagnosis"].values) 

dataset.head()

dataset.tail()
```

📌 This time we have only one file, so we need to divide this dataset into a train set and test set ourselves. We can do that using the sklearn library function train test split.

```Python
from sklearn.model_selection import train_test_split

train, test = train_test_split(dataset, test_size=0.3)
```

📌 Also, as we did for the regression problem, we need to define the “target” we want to predict. In this problem, we’re trying to predict if the tumor is malignant (1) or benign (0). Hence, our target variable is the “diagnosis” column. And the rest of the columns are “features”. Let’s assign the x variable as target and the y variable as features. And remember, we need to do this for both the train and test datasets. By the way, you can also do this for the whole dataset and then divide into train and test.

```Python
X_train = train.drop("diagnosis",axis=1)
y_train = train.loc[:,"diagnosis"]

X_test = test.drop("diagnosis",axis=1)
y_test = test.loc[:,"diagnosis"]
```
📌 We’re ready to import our **logistic regression model** from **sklearn library**. And after importing the logistic regression model, we can assign it to the “model” variable. Now, we’re ready to train our model, that means teach the hidden patterns in the train dataset to our model. And finally, we can make the predictions on the test dataset.

```Python
from sklearn.linear_model import LogisticRegression
model_1 = LogisticRegression()
model_1.fit(X_train,y_train)

predictions = model_1.predict(X_test)
predictions
```

📌 Using the **confusion matrix**, we can check the accuracy of our results. First, we import confusion_matrix from sklearn and display the number of each metric. We have 103 true negatives, 0 false positives, 4 false negatives, 64 true positives. That means out of 171 predictions are correct

```Python
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, predictions)
```

📌 We import **classification_report** from **sklearn** and display the *evaluation metrics*. And the ratios we get, are quite high. We’ve done a really good job.

```Python
from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))
```

📌 Well done! We have actually trained and tested a logistic regression classifier. Now, why not try another classification algorithm: **Support Vector Machine**. *SVM* is a **supervised machine learning technique** that can be used to solve classification and regression problems. It is, however, mostly used for classification. In this algorithm, we use an axis to represent each feature and plot all data points in the space. Then, the SVM model finds boundaries to separate these classes. The decision boundary is what separates different data samples into specific classes. Consider a dataset of different animals of two classes: birds and fish. In this dataset there are only three features: body weight, body length, and daily food consumption. We draw a 3-dimensional grid and plot all these points. A SVM model will try to find a 2D plane that differentiates the 2 classes.

📌 If there were more than 3 features, we would have a hyper-space. A hyper-space is a space with higher than 3 dimensions like 4D, 5D, and so on and therefore it is not possible to visualize. We can find a hyper-plane that clearly distinguishes different classes. Hyper-planes are multidimensional planes that exist in four or more dimensions. This hyper-plane is used as a condition to perform classification. If the hyper-planes are linear, the SVM is called Linear Kernel SVM. However, the hyper-plane can be nonlinear as well. 

📌 In that case we use a Polynomial Kernel or other advanced SVMs. Let’s see how this model performs with the same breast cancer dataset we used earlier. We start with importing LinearSVC from sklearn and assigning it to the variable. Now, we’re ready to train our model, that means teach the hidden patterns in the train dataset to our model. Finally, we can make the predictions on the test dataset.

```Python
from sklearn.svm import LinearSVC

model_2 = LinearSVC()

model_2.fit(X_train,y_train)

predictions = model_2.predict(X_test)
predictions
```

📌 Our predictions with the Support Vector Classifier are ready! Now, we can check the accuracy of our model in the same way we did for Logistic Regression. We can start with the confusion matrix. We have one hundred one true negatives, 2 false positives, 4 false negatives, 64 true positives. This means that 165 out of 171 predictions are correct, just a couple less than before. 

```Python
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, predictions)
```

📌 We should also check the classification report. We get quite high metrics here as well. But we got better with Logistic Regression. And here our false positives were higher. This is a key metric for this dataset we want to minimize, because we don’t want healthy patients to be diagnosed with cancer. Therefore, we prefer using the Logistic Regression model for this problem and dataset.

```Python
from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))
```
