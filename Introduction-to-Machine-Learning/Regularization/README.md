# Regularization - GlobalAIHub

📌 We will start with overfitting and underfitting. A machine learning model can easily be **overfitted** or **underfitted** during training. In fact, these are some of the most common problems you can face while training your model. So, what exactly are underfitting and overfitting in ML? Let’s try to understand these concepts with a simple example. Consider a high school student, Clara. She has to prepare for a university entrance exam in biology. She is solving all the easy questions from her book. When she goes to take her exam, however, she will also see more complex questions. She won’t be able to solve them because she only trained on the easy ones and didn’t practice solving the more complex ones. This is underfitting.

📌  So, we could say that underfitting happens when a model is too simple, it is unable to find the patterns in the training data and therefore generates a high error on the training set, but also on unseen data. These models are also described as **“highly biased”**. The bias refers to, the inability of the model to understand complexity of data. 

📌  We usually get an underfit when there is not enough data in the training set, or it lacks complexity, meaning it has too few features to recognize patterns from. Let’s consider Clara once more. Now, she is solving math problems. Instead of studying for all the different types of math questions that will be covered in the exam, she is only focusing on algebra questions. Even though Clara is able to solve both easy and difficult algebra questions, when she sees geometry questions later in the exam, she is unable to solve them because she hasn’t studied them. This is overfitting.

📌 We can think of overfitting as the opposite of underfitting. In this scenario, the model is trained too much on our specific training dataset and it generates high accuracy. But when it is applied to unseen data, the result has low accuracy. This is because it is looking for the patterns it has been trained on, but is unable to generalize in the test data. Generalization refers to the model’s ability to adapt to unseen data. These models are also described as, **“high variance models”**. The variance refers to the sensitivity of a model to specific datasets. More than it learns from the training data, it memorizes it.

📌 We usually get an overfit, when the training data is very specific and has too many features. Both underfitting and overfitting lead to poor predictions. What we want to achieve is optimal fitting, a good balance. The performance of our model is affected by both **variance** and **bias**, which can lead to underfitting and overfitting. By adjusting variance and bias, we aim to generalize our model so that it is neither too complex nor too simple. Because as we have found out, overfitting with high bias or underfitting with high variance are not ideal for our model to make accurate predictions. By the way, we have to mention that there is a trade-off between bias and variance. This means that, as variance increases, bias decreases and vice versa.

📌 Now that we have learned that high bias leads to underfitting, and high variance leads to overfitting, let’s discuss some approaches to solve these problems. We start with underfitting, as it is easier to deal with. The general approach to solving underfitting is to make the data more complex. We can increase the number of observations in the training set. We can also add new features that could impact the predictions. This is easy, because we don’t lose any original data from the training set, as we don’t remove anything. At the end, our model will gain more complexity and will try to find some patterns in the data that are closer to actual values.

📌 Now, we can move on to overfitting. The general idea for solving overfitting and high variance is, to make the data less complex. Making data less complex is hard because by removing complexities, we may lose useful information that helps us to make predictions. One way to address this challenge is through regularization. **Regularization** prevents the learning of more complex patterns. It does this by shrinking coefficients towards zero, so that the impact of less significant features is reduced, and high variance is prevented. Regularization uses **loss functions that are called L1 and L2**. You are already familiar with one of the simplest and most common loss functions *“Mean Squared Error”**, MSE. You can think of the **L1** and **L2** loss functions as a modified version of that.

📌 Generally, the L2 loss function is more common. But when there are outliers in the dataset, using the L2 loss function is not useful because, taking squares of the differences between the actual and predicted values will lead to a much larger error.

📌 Now, let’s see how these concepts play out in practice. In this practical example, we will try to make predictions using L2 regression. And then, we will show the results of regularization on accuracy, using the MSE metric. Let’s start with importing the required libraries. Now, we import the example dataset. Then we divide our dataset into features and targets. Finally, we split them into train and test datasets.

```Python
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = pd.read_csv("train.csv")

X = data.drop('SalePrice',axis=1)
y = data.loc[:,'SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
```

📌 Let’s use ridge regression and compare the result to linear regression! We assign each of the two models to a variable in order to use it easily. You may notice that we used some hyperparameters, one is our lambda, which as explained before is our tuning parameter. The **“normalize”** parameter converts all of the data points into the range of 0 and 1 which decreases the variety in our dataset. By setting different hyperparameters, we can improve our model.

```Python
linear_reg = LinearRegression()
ridge_reg = Ridge(alpha=0.05, normalize=True)
```

📌 Then, we can train both of our models using the training dataset. Finally, we are ready to make predictions using the test dataset.

```Python
linear_reg.fit(X_train, y_train)
ridge_reg.fit(X_train, y_train)

linear_pred = linear_reg.predict(X_test)
ridge_pred = ridge_reg.predict(X_test)
```

📌 Now, how can we compare the two models? Yes, we can use metrics! By calculating the mean squared error, **MSE**, we will be able to make a comparison between the performances of linear and ridge regression. Let’s print the results to see the difference. You can notice that, MSE for ridge regularization is lower, which means its predictions are better. Even though we only see a slight improvement in results, it won’t be the same for real-life datasets, since their size is larger and, they are more complex. The variety of data is greater and therefore, easier to overfit. Using regularization in such cases, will have a bigger impact on its accuracy.

```Python
linear_mse = mean_squared_error(y_test, linear_pred)
ridge_mse = mean_squared_error(y_test, ridge_pred)

print(f"MSE without Ridge: {linear_mse}")
print(f"MSE with Ridge : {ridge_mse}")
```



