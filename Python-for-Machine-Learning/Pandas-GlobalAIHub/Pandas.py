#Import Pandas and NumPy
import pandas as pd
import numpy as np

#Read the data from the "employee_revenue_lastyear.csv" file using the .read_csv() method
last_year = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/GlobalAIHub-Projects/employee_revenue_lastyear.csv")

#Use the .head() function without specifying any number
last_year.head()

#Use the .head() function with a number n as an argument
last_year.head(8)

#Use the .tail() function without specifying any number
last_year.tail()

#Use the .tail() function with a number n as an argument
last_year.tail(6)

#Use the shape attribute to display the number of rows and columns 
last_year.shape

#Use the .info() function to get more details
last_year.info()

#Add the column "Year" and assign the value "2021" to its rows
last_year["Year"] = 2021
#Display the DataFrame
last_year

#Copy and paste the NumPy arrays "names", "call_numbers", "average_deal_sizes", and "revenues"
names = np.array(['Ben', 'Omer', 'Karen', 'Celine', 'Sue', 'Bora', 'Rose', 'Ellen', 'Bob', 'Taylor,', 'Jude'])
call_numbers = np.array([300, 10, 500, 70, 100, 100, 600, 800, 200, 450, 80])
average_deal_size = np.array([8, 6, 24, 32, 5, 25, 25, 40, 15, 10, 12])
revenues = np.array([2400, 60, 12000, 2275, 500, 770, 4000, 6000, 800, 1200, 500])

#Create a dictionary with the column names as keys and the arrays as values
dictionary = {"names":names,
              "call_numbers":call_numbers,
              "average_deal_size":average_deal_size,
              "revenues": revenues}
#Convert the dictionary to a Pandas Dataframe
current_year = pd.DataFrame(dictionary)
#Use the .head() function to check it
current_year.head()

#Add the column "Year" and assign the value "2022" to its rows
current_year["Year"] = 2022
#Use the .head() function to see the result
current_year.head()

#Use the .head() function to print the DataFrame "last_year"
last_year.head()

#Use the columns attribute to assign the column names of "last_year" to "current_year"
current_year.columns = last_year.columns

#Use the .concat() function to concatenate the two DataFrames
all_data = pd.concat([last_year,current_year],axis=0)
#Display the DataFrame "all_data"
all_data

#Use the .reset_index() function to reset the indexes
all_data.reset_index(drop=True,inplace=True)
#Display the DataFrame "all_data"
all_data

#Check the DataFrame for missing values
all_data.isna().any()

#Replace the missing values with the mean of the respective column
all_data.fillna(value=np.mean(all_data),inplace=True)
#Display the DataFrame
all_data

#Drop the duplicates
all_data.drop_duplicates(inplace=True)
#Reset the indexes
all_data.reset_index(drop=True,inplace=True)
#Display the DataFrame 
all_data

#Use the .describe() method to get a summary of the statistics
all_data.describe()

#Use the .describe() method to get a summary of the statistics for 2021
all_data[all_data["Year"] == 2021].describe()
#Use the .describe() method to get a summary of the statistics for 2022
all_data[all_data["Year"] == 2022].describe()

#Sort the DataFrame by the column "Revenue"
all_data.sort_values(by="Revenue")

#Sort the revenue values od 2022
all_data[all_data["Year"] == 2022].sort_values(by="Revenue")

#Count how often the names of the employees appear in the DataFrame
all_data["Name"].value_counts()
