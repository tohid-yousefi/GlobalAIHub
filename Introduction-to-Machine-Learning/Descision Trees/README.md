# Descision Trees - GlobalAIHub

๐ **Decision Tree** algorithms are a series of if-else statements, that can be used to predict a result based on data. Simply put, it randomly asks questions, checks if the dataset is divided correctly and changes the questions accordingly.

๐ A decision tree algorithm creates a decision tree with the right series of questions or comparisons. These will lead to the right result, which could be a prediction or category. A decision tree is also described as a series of so-called **โnodesโ**. The starting point of the tree is called a **root** or **top node**, then we have a set of decision nodes until we reach the **leaf node**. The leaf is our result, our final category or prediction.

๐ Generally, decision tree algorithms are more suitable for classification problems, however, regression trees are very useful if there are strong distinctions between the different cases. This is because the logic of decision based algorithms is categorical, thatโs why itโs closer to classification problems. Predicting the output of continuous values, which is the case in regression problems, is harder based on decisions. And there are other things that we should consider when using decision trees. We already mentioned that decision trees ask questions and split data according to these. Based on the questions, we will calculate the information gain.

๐ **Information gain** is the measure of how much information a question gives about a class. If a question splits the dataset better than the other one, that question will be used.

๐ For decision trees, we can perform an operation called **โpruningโ**. *Pruning* is basically removing some unnecessary features based on the information gain, in order to get a less complex decision tree. Pruning helps to prevent overfitting, and high variance so that our model generalizes well to unseen data.

๐ The performance of the decision tree is measured just like any other ML algorithm.We use the **confusion matrix**, **accuracy score**, **f1-score** for *classification trees*. And **RMSE**, **MSE** for *regression trees*. We can compare the performance of other models, by comparing the results of their metrics with results we get from decision trees, and choose the optimal model that fits the problem we are trying to solve.

๐ So, what are the main advantages of decision trees? Compared to other algorithms, decision trees take less effort to prepare data and there is no need to normalize and scale it. Missing values in the data donโt significantly affect the process of creating a decision tree. But there are of course some disadvantages. Even though decision trees have many advantages, they require longer time to train. Another problem is that decision trees are unstable. Slight changes in the data can result in an entirely different tree being constructed. Also, decision trees tend to overfit for complicated data, which is why they may not generalize the input well.

๐ Indeed, **overfitting is a huge problem**. This is where another algorithm, **random forest**, based on decision trees comes into play. The approach with random forest is that, instead of using only one tree, it uses many trees. It runs a test dataset on each tree, gets predictions from each and based on the majority votes, that means the category that appears the most, we get a final output. But with many trees, we need many datasets. Since we already have a dataset, we can easily divide it into smaller ones, and use each of these sets for random forest trees. Since each tree focuses on different parts of the dataset, this prevents overfitting.

๐ Now we import the example dataset, then we divide our dataset into features, X and targets, y. Then, we split them into train and test datasets.

```Python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

dataset = pd.read_csv("titanic.csv")
dataset.head()

X = dataset.drop("Survived", axis=1)
y = dataset.loc[:,"Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
```

๐ Letโs use a decision tree and random forest classifier on the training dataset!

```Python
decisiontree = DecisionTreeClassifier()
randomforest = RandomForestClassifier()

decisiontree.fit(X_train, y_train)
randomforest.fit(X_train, y_train)
```

๐ You can see the features and splitting values used on each of the nodes of the decision tree. This process goes on until the decision tree converges on a best performing solution. The nodes at the bottom tell us that this tree goes deeper, as we only wanted to see the first three layers. Finally, we are ready to make predictions using the test dataset.

```Python
decisiontree_pred = decisiontree.predict(X_test)
randomforest_pred = randomforest.predict(X_test)
```

๐ Now, how can we compare the two models? Yes, we can use performance metrics! And since we are handling a classification problem, metrics weโll use are precision score, recall score, f1-score and accuracy. By showing the classification report, weโll be able to see these scores and make a comparison between the performances of these two algorithms. Letโs print out the classification reports and see the difference. By observing these classification reports, we can see the difference between the performances of these two models. We have higher precisions, recalls and f1-scores for the predictions of the random forest classifiers, therefore it is performing better.

```Python
print(classification_report(y_test, decisiontree_pred))

print(classification_report(y_test, randomforest_pred))
```

๐ We import plot_tree from sklearn and matplotlib. As the name suggests, plot_tree will allow us to plot the decision tree. You can also see that we adjust many arguments inside the function. These help us to get a better visualization and filter the information that is shown. For example, the argument โmax_depthโ is set to 2, this specifies that only the first three layers of the tree should be shown. You can check out the documentation of **plot_tree** to learn more about these arguments. And finally we can display the plot.

```Python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10,10))
plot_tree(decisiontree,
          max_depth=2,
          feature_names=X.columns,
          filled=True,
          impurity=False,
          rounded=True,
          precision=1)

plt.show()
```
