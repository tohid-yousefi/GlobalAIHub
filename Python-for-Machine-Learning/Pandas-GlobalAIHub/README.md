# Pandas-GlobalAIHub
"Pandas" project of "Python for Machine Learning" course of "globalaihub.com"

<hr />

# Chapter 5
## Pandas
The sales department wants to compare the performances of this and last year. Firstly, they delivered your team the performance review from last year. The data is in a comma-separated format, a .csv file called “employee_revenue_lastyear.csv”. It is in tabular form and includes information about 11 employees. 

### Importing libraries
First, you need to import the required libraries. 

📌 Import the Pandas and NumPy libraries.

```Python
#Import Pandas and NumPy
import pandas as pd
import numpy as np
```

### Get the data from last year
Next, you need to get the data from the .csv file that the sales team provided. 

📌 Use the .read_csv() function of the Pandas library to import the data from "employee_revenue_lastyear.csv" and assign it to the variable "last_year".

```Python
#Read the data from the "employee_revenue_lastyear.csv" file using the .read_csv() method
last_year = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/GlobalAIHub-Projects/employee_revenue_lastyear.csv")
```

### Check the imported data
It is important to check if everything is included. But you don't need to see the whole DataFrame, it will be enough to see the first and last rows.

📌 Use the .head() function to get the first n-rows of the DataFrame:
  1. Without specifying any number to get the first 5 rows by default.
  2. Give a number n as an argument to print the first n-rows, for example 8.

```Phython
#Use the .head() function without specifying any number
last_year.head()

#Use the .head() function with a number n as an argument
last_year.head(8)
```
📌 Use the .tail() function to get the last n-rows of the DataFrame:
  1. Without specifying any number to get the last 5 rows by default.
  2. Give a number n as an argument to print the last n-rows, for example 6.

```Python
#Use the .tail() function without specifying any number
last_year.tail()

#Use the .tail() function with a number n as an argument
last_year.tail(6)
```

Another way to quickly get the number of rows and columns is using the shape attribute.

📌 Use the *shape* attribute to display the number of rows and columns.

```Python
#Use the shape attribute to display the number of rows and columns 
last_year.shape
```

If you want to get more details about the DataFrame, you can use the .info() function.

📌 Use the .info() function to get more details about the DataFrame.

```Python
#Use the .info() function to get more details
last_year.info()
```

### Add information to the DataFrame
Now you should have enough insights about the DataFrame. :)

You know that the data in the DataFrame is from 2021. This information should be included, so you decide to add the column "Year".

📌 Add the column "Year" to the DataFrame and pass the value "2021" to its rows. Then display the DataFrame to see the result.

```Python
#Add the column "Year" and assign the value "2021" to its rows
last_year["Year"] = 2021
#Display the DataFrame
last_year
```

### Get the data from this year

You also need to add the data from this year to be able to compare both years.
You already prepared this data in chapter 4.

📌 Copy and paste the NumPy arrays "names", "call_numbers", "average_deal_sizes", and "revenues" from <a href="https://github.com/tohid-yousefi/Numpy-GlobalAIHub" target="_blank">chapter 4</a>.

```Python
#Copy and paste the NumPy arrays "names", "call_numbers", "average_deal_sizes", and "revenues"
names = np.array(['Ben', 'Omer', 'Karen', 'Celine', 'Sue', 'Bora', 'Rose', 'Ellen', 'Bob', 'Taylor,', 'Jude'])
call_numbers = np.array([300, 10, 500, 70, 100, 100, 600, 800, 200, 450, 80])
average_deal_size = np.array([8, 6, 24, 32, 5, 25, 25, 40, 15, 10, 12])
revenues = np.array([2400, 60, 12000, 2275, 500, 770, 4000, 6000, 800, 1200, 500])
```

Now create a DataFrame using these arrays. 

📌 Create a dictionary with the arrays. Specify the column names in the keys.

📌 Convert the dictionary to a Pandas DataFrame "current_year" and use the .head() function to check it.

```Python
#Create a dictionary with the column names as keys and the arrays as values
dictionary = {"names":names,
              "call_numbers":call_numbers,
              "average_deal_size":average_deal_size,
              "revenues": revenues}
#Convert the dictionary to a Pandas Dataframe
current_year = pd.DataFrame(dictionary)
#Use the .head() function to check it
current_year.head()
```

Similar to what we did with last year's data, add the year information.

📌 Add the column "Year" to the DataFrame and pass the value "2022" to its rows. Then use the .head() function to see the result.

```Python
#Add the column "Year" and assign the value "2022" to its rows
current_year["Year"] = 2022
#Use the .head() function to see the result
current_year.head()
```

### Compare the two DataFrames

Now that you printed the DataFrame "current_year", print "last_year" as well to compare them.

📌 Use the .head() function to print the DataFrame "last_year".

```Python
#Use the .head() function to print the DataFrame "last_year"
last_year.head()
```

You notice that the column names of the two DataFrames are different. You need to fix this problem.

📌 Assign the column names of "last_year" to "current_year" by using the *columns* attribute.

```Python
#Use the columns attribute to assign the column names of "last_year" to "current_year"
current_year.columns = last_year.columns
```

### Concatenate two DataFrames
Now that the two DataFrames have the same column names, You can merge - or concatenate - them into a single DataFrame "all_data".

📌 Use the .concat() function with the argument "axis" set to 0. Then display the DataFrame.

```Python
#Use the .concat() function to concatenate the two DataFrames
all_data = pd.concat([last_year,current_year],axis=0)
#Display the DataFrame "all_data"
all_data
```

### Check the data

This worked out well, but you noticed that the indexes are incorrect. You need to reset them.

📌 Use the .reset_index() function. Set the arguments "drop" and "inplace" to "True". Then display the DataFrame.

```Python
#Use the .reset_index() function to reset the indexes
all_data.reset_index(drop=True,inplace=True)
#Display the DataFrame "all_data"
all_data
```

Next, you need to check the entries for missing values.

📌 Use the isna.() and .any() function to see if there are missing values in the DataFrame.

```Python
#Check the DataFrame for missing values
all_data.isna().any()
```

From the output, you see that there are missing values in the columns "Average deal size" and "Revenue". You decide to fix this problem by filling the missing values using with the mean of the respective column.

📌 Use the .fillna() function to fill the missing values. Use the .mean() function to set the argument "value" to the mean of "all_data". Again, set "inplace" to "True".

```Python
#Replace the missing values with the mean of the respective column
all_data.fillna(value=np.mean(all_data),inplace=True)
#Display the DataFrame
all_data
```

Also, there may be some duplicated rows, you need to drop them.

📌 Use the .drop_duplicates() method to remove any duplicated rows.

📌 Reset the indexes again using the reset_index() function.

```Python
#Drop the duplicates
all_data.drop_duplicates(inplace=True)
#Reset the indexes
all_data.reset_index(drop=True,inplace=True)
#Display the DataFrame 
all_data
```

### Data analysis
#### Statistical analysis
The DataFrame is ready, great!

Now you can use it to analyse the overall performance of the employees over the last two years. You prepare a summary of the statistics.

📌 Use the .describe() method.

```Python
#Use the .describe() method to get a summary of the statistics
all_data.describe()
```

You can do the same for each year separately. 

📌 Prepare a summary of the statistics for the data from 2021.

📌 Prepare a summary of the statistics for the data from 2022.

```Python
#Use the .describe() method to get a summary of the statistics for 2021
all_data[all_data["Year"] == 2021].describe()
#Use the .describe() method to get a summary of the statistics for 2022
all_data[all_data["Year"] == 2022].describe()
```

#### Ranking of the employees by revenue
You also want to rank the employees by the generated revenue. 

📌 Use the .sort_values() method to sort the values by the column “Revenue”.

```Python
#Sort the DataFrame by the column "Revenue"
all_data.sort_values(by="Revenue")
```

As you did with the .describe() function, you can also use conditions to filter information. 

📌 Sort the revenue values of 2022.

```Python
#Sort the revenue values od 2022
all_data[all_data["Year"] == 2022].sort_values(by="Revenue")
```

#### How many years of employment?

Finally, you can count how many times an employee appears in the DataFrame to determine which employee has worked for the company for two years.

📌 Use the value_counts() function for the column "Name".

```Python
#Count how often the names of the employees appear in the DataFrame
all_data["Name"].value_counts()
```
