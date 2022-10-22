# Regression - GlobalAIHub

📌 **Regression** is a type of *supervised learning* that uses an algorithm to understand the relationship between a dependent variable, that is the input, and an independent variable, which is the output. Regression models are helpful for predicting numerical values based on different features’ values. For example, temperature forecast based on wind, humidity and pressure, or price estimations of a car based on its model year, brand, and transmission type. In regression, we want to build a relationship between each feature and the output so that we can predict for example, the price of the house when we know the features but not the price. If this relationship is linear, this algorithm is called linear regression.

📌 **Linear regression** is perhaps the most well-known and well-understood algorithm in statistics and machine learning. A simple linear regression model tries to explain the relationship between the two variables using a best fitting straight line. We call this a regression line.

📌The first step is reading the data. To do that, we need to import Python’s handy data science library, **Pandas**. After importing the pandas library we can easily load our train and test datasets using *read_csv*. We will use the train dataset to help our regression model to learn some important patterns in the data. Then we’ll use the test dataset to check how well the model learned the patterns or how well it predicts. Let’s start with this simple operation.

```Python
import pandas as pd

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
```

📌 Let’s observe features in our train dataset. We can see that some of the features we have include the house’s general quality, the year it was built, the size of the garage, and so on.

```Python
train.columns
```


📌 Now that we have our datasets loaded, we can import linear regression models from the most important machine learning library of Python; **sklearn**. It is an open-source library for machine learning. There are many models constructed in this library, we just need to import the one that we will use.

```Python
from sklearn.linear_model import LinearRegression
```

📌 After importing linear regression model, we can assign it to the **“model”** variable to use it easily. 

```Python
model = LinearRegression()
```

📌 In this example, we’re trying to predict the house prices. The column we want to predict also is called “ground truth”, “target” or “labels” and other columns are called “features” or “attributes”. For the model to predict the house prices, first, we need to define which column has the house prices, or the ground truth. Then we remove it from the features of the data using the drop function and assign it as ‘labels’ using the *loc* function. Inside the drop function, we write the column name followed by the argument axis which we set to 1. This indicates that the specified column needs to be deleted.

```Python
X_train = train.drop('SalePrice', axis=1)
y_train = train.loc[:,'SalePrice']
```

📌 Basically, we want to predict y with the help of x. And generally, we assign target to the y variable, features to the X variable. Then we can **fit** our model, which means teaching the hidden patterns in the training dataset into it.

```Python
model.fit(X_train,y_train)
```

📌 After the fitting process, our model is almost ready to make predictions. But before that, we also need to divide our test dataset into **“target”** and **“features”**.

```Python
X_test = test.drop('SalePrice', axis=1)
y_test = test.loc[:,'SalePrice']
```

📌 Here the target dataset contains the actual values which our model will compare its predictions.

```Python
predictions = model.predict(X_test)
```

📌 comparing some data points with your eyes won’t tell you how well your model predicts. This is exactly where we use evaluation metrics! Let’s import **mean squared error** from the *sklearn library*. We also need to import **square root** from the *NumPy library*, because we want to observe our root mean squared error as it’s in the same unit with our data. After importing, we can use these functions to average the calculated error.

```Python
from sklearn.metrics import mean_squared_error
from numpy import sqrt

rmse = sqrt(mean_squared_error(y_test, predictions))
rmse

comparison = pd.DataFrame({"Actual Values": y_test,"Predictions": predictions})
```

📌 Now, we are completely ready to make predictions using the features of the test dataset. Let’s observe our predictions and actual test values. For that, we can simply put them into the same data frame and observe some of the rows using head and tail functions. We see that the actual values and the predictions by our model are more or less close. Of course, as in every machine learning model, there are some inaccuracies.

```Python
comparison.head()

comparison.tail()
```

📌 Our **RMSE** is approximately 33 000! If we consider our average price is 185 000, and maximum price as big as 500 000, then 33 000 may be considered normal for your first model. Keep going! Also, we can check which features have the most impact on our predictions. Basically, we can check for **correlations** on our train dataset. But since we need correlations between target and features, we can simply take the “SalePrice” column from this data frame. From the data frame, we can’t decide which ones have the most impact. Let’s sort and see the top 10 using sort_values, then head functions.

```Python
train.corr()["SalePrice"].sort_values(ascending=False).head(10)

correlations = train.corr()
correlations

saleprice_correlations = correlations["SalePrice"]
saleprice_correlations
```

📌 Don’t forget that we need to set ascending as false because we want to see 10 highest values!

```Python
saleprice_correlations.sort_values(ascending=False).head(10)
```


