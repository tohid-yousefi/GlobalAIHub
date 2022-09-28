#Import Matplotlib and Pandas
import matplotlib.pyplot as plt
import pandas as pd

#Create a list for the x-axis values
x = [0,2,4,6,8,10,12,14,16]
#Create a list for the y-axis values
y = [0,4,16,36,64,100,144,196,256]

#Use the .plot() function to create a line plot
plt.plot(x,y)
#Use the .title() method to add the title
plt.title("Example data - Line plot")
#Display the plot
plt.show()

#Use the .scatter() function to create a scatter plot
plt.scatter(x,y)
#Use the .title() method to add the title
plt.title("Example data - Scatter plot")
#Display the plot
plt.show()

#Use the .bar() function to create a bar chart
plt.bar(x,y)
#Use the .title() method to add the title
plt.title("Example data - Bar plot")
#Display the chart
plt.show()

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

#Read the data from "employee_performance.csv""
data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/GlobalAIHub-Projects/employee_performance.csv")
#Use the .head() function to check the dataset
data.head()

#Use the .value_counts() function to determine the number of people in the different categories
educational_level = data["Education"].value_counts()
#Display the result
educational_level

#Create a pie chart with the labels "College", "High School", "University"
plt.pie(educational_level,labels=educational_level.index)
#Display the chart
plt.show()

#Create a bar chart with the names on the x-axis and the revenue values on the y-axis
plt.bar(data["Name"],data["Revenue"])
#Display the chart
plt.show()

#Create a bar chart with the data "Revenue" and "Number of calls"
plt.bar(data["Name"],data["Revenue"],label="Revenues")
plt.bar(data["Name"],data["Number of Calls"],label="Number of calss")
#Add a legend
plt.legend()
#Add grid lines
plt.grid()
#Display the chart
plt.show()
