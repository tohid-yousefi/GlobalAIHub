# Naive Bayes - GlobalAIHub

📌 The main aim of classification is, to find the class corresponding to a given feature set. The **Naive Bayes** classifier algorithm is based on a famous theorem called **“Bayes theorem”** which is centered on conditional probability. Conditional probability is the probability of an event ‘A’ happening given that another event ‘B’ has already happened. For example, consider event ‘A’ to be “having a fever”, and event ‘B’ to be “infected with Covid-19”. With conditional probability, we can ask the question: what is the chance of having a fever given that you have been infected with Covid-19.

📌 The Bayes’ theorem is an extension of conditional probability. In a sense, it allows us to use reverse reasoning. The Naive Bayes algorithm does the same for the class and its features. Instead of calculating the probability of a feature belonging to a class, it approaches the issue from another angle. 

📌 There are three types of Naive Bayes Classifiers in sklearn; **Bernoulli Naive Bayes**, **Multinomial Naive Bayes** and **Gaussian Naive Bayes**. We use Bernoulli Naive Bayes when our data is binary like true or false, yes or no and so on. We use Multinomial Naive Bayes when we have discrete values such as number of family members or, pages in a book. We use Gaussian Naive Bayes when all of our features are continuous variables, like temperature or height. Let’s take the dataset on tumors from our classification session, which only has continuous variables. We use the Gaussian Naive Bayes algorithm. As always, we start with importing the Pandas library for reading our data file. Then we can read and display the first few rows of the dataset.

```Python
import pandas as pd

dataset = pd.read_csv("breast-cancer.csv")
dataset.head()
```

📌 we need to convert our target variable from categorical to numerical type using label encoding. Then, we can continue with defining our features and a target.kalın metin

```Python
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
dataset["diagnosis"] = labelencoder.fit_transform(dataset["diagnosis"].values)

X = dataset.drop("diagnosis", axis =1)
y = dataset["diagnosis"]
```

📌 After we define the features and a target, we can split them into train and test.

```Python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
```

📌 Now, the most exciting part! Let’s create our model, Gaussian Naive Bayes, teach some hidden patterns to it with training data, and finally use it to make predictions.

```Python
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train, y_train)
```

📌 Now, we can check the strength of our predictions.

```Python
predictions = model.predict(X_test)
predictions

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, predictions)
```

📌 Looking at the results, we have accuracy and precision above 90% which can be considered as a good result.

```Python
from sklearn.metrics import classification_report

print(classification_report(y_test,predictions))
```
