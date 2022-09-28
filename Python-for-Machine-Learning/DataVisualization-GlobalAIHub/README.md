# DataVisualization-GlobalAIHub
"Data Visualization" project of "Python for Machine Learning" course of "globalaihub.com"

<hr />

# Chapter 6
## Data Visualization

Your manager wants you to provide data visualizations for the sales team to help them gain better, useful insights.

### Importing Libraries

First, you need to import the required libraries.

📌 Import the Matplotlib and Pandas libraries.

```Python
#Import Matplotlib and Pandas
import matplotlib.pyplot as plt
import pandas as pd
```

### Matplotlib Basics

Before focusing on your task, first practice the Matplotlib basics.

📌 Create example data:
  1. A list for the x-axis values
  2. A list for the y-axis values

```Python
#Create a list for the x-axis values
x = [0,2,4,6,8,10,12,14,16]
#Create a list for the y-axis values
y = [0,4,16,36,64,100,144,196,256]
```

#### Line plot
Use this simple data to explore some of the different graph types.

📌 Use the .plot() function of Matplotlib to create a line plot and the .title() method to add the title "Example data - Line plot". 

📌 Use *plt.show()* to display the plot.

```Python
#Use the .plot() function to create a line plot
plt.plot(x,y)
#Use the .title() method to add the title
plt.title("Example data - Line plot")
#Display the plot
plt.show()
```

#### Scatter plot

Next, use the same data to create a scatter plot.

📌 Use the .scatter() function of Matplotlib to create a scatter plot and use the .title() method to add the title "Example data - Scatter plot".

📌 Use *plt.show()* to display the plot.

```Python
#Use the .scatter() function to create a scatter plot
plt.scatter(x,y)
#Use the .title() method to add the title
plt.title("Example data - Scatter plot")
#Display the plot
plt.show()
```

#### Bar chart

Lastly, use the same data to create a bar chart.

📌 Use the .bar() function of Matplotlib to create a bar chart and use the .title() method to add the title "Example data - Bar chart".

📌 Use *plt.show()* to display the chart.

```Python
#Use the .bar() function to create a bar chart
plt.bar(x,y)
#Use the .title() method to add the title
plt.title("Example data - Bar plot")
#Display the chart
plt.show()
```

### Display multiple graphs in on figure

You decide that you would like to see the graphs side by side. Each graph should have a different color to make the figure more readable.

📌 Use the .figure() function of Matplotlib and its argument "figsize" to create a figure object of the size 18x5.

📌 Use the .add_subplot() method to add the three graphs you just created to the figure and add a title to each. 

  * There should be 1 row with the 3 graphs. 

  * Use the "color" argument to change the color of each plot:
    1. Line plot: red
    2. Scatter plot: green
    3. Bar chart: orange
  
 ```Python
 #Create a figure object of the size 18x5
fig = plt.figure(figsize=(18,5))

#Use the .add_subplot() method to add the line plot
first_plot = fig.add_subplot(1,3,1)
#Change the color to red
first_plot.plot(x,y,color="red")
#Add the title
first_plot.set_title("Example data - Line plot")

#Use the .add_subplot() method to add the scatter plot
second_plot = fig.add_subplot(1,3,2)
#Change the color to green
second_plot.scatter(x,y,color="green")
#Add the title
second_plot.set_title("Example data - Scatter plot")

#Use the .add_subplot() method to add the bar chart
third_plot = fig.add_subplot(1,3,3)
#Change the color to orange
third_plot.bar(x,y,color="orange")
#Add the title
third_plot.set_title("Example data - Bar Chart")
#Display the figure
plt.show()
 ```
 
 ### Visualization of the sales report

Now, you are prepared to take on your task. 

The sales team is creating a report and they need a visualization of the results. Your manager Rachel asks you to create some graphs so people reviewing the report will have a better understanding of the data. They deliver you a .csv file that contains the sales data.

#### Data Preparation

First you need to get the data from the .csv file.

📌 Use the .read_csv() function to read "employee_performance.csv" and assign it to the variable "data".

📌 Use the .head() function to check what the dataset contains.

```Python
#Read the data from "employee_performance.csv""
data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/GlobalAIHub-Projects/employee_performance.csv")
#Use the .head() function to check the dataset
data.head()
```

#### Plot the education level

First you create some insights about the education level of sales team and you decide that a pie chart will be useful. You need to determine the number of people in the different categories.

📌 Use the .value_counts() function to determine the number of people in the different categories.

```Python
#Use the .value_counts() function to determine the number of people in the different categories
educational_level = data["Education"].value_counts()
#Display the result
educational_level
```

Then, create a pie chart with the labels "College", "High School", "University"

📌 Use the .pie() function to create a pie chart and the "labels" argument, to give the indexes of the Pandas Series as labels.


```Python
#Create a pie chart with the labels "College", "High School", "University"
plt.pie(educational_level,labels=educational_level.index)
#Display the chart
plt.show()
```

#### Plot the revenue

Next, you need to create a bar chart of the revenue generated by the employees.
The names should be on the x-axis and revenues on the y-axis.

📌 Use the .bar() function to create a bar chart of the revenue values.

```Python
#Create a bar chart with the names on the x-axis and the revenue values on the y-axis
plt.bar(data["Name"],data["Revenue"])
#Display the chart
plt.show()
```

The graph, in this state, just shows a comparison between employees. 

You decide to add the data "Number of calls" in the graph to increase the understandability. 

📌 Add the data for "Revenue" as well as for "Number of calls" to the bar chart. Use the argument "label" to label the data.

To differentiate between the data, you need to add a legend. Also adding grid lines will be useful.

📌 Add a legend by using the .legend() function and the .grid() function to add grid lines.


```Python
#Create a bar chart with the data "Revenue" and "Number of calls"
plt.bar(data["Name"],data["Revenue"],label="Revenues")
plt.bar(data["Name"],data["Number of Calls"],label="Number of calss")
#Add a legend
plt.legend()
#Add grid lines
plt.grid()
#Display the chart
plt.show()
```
