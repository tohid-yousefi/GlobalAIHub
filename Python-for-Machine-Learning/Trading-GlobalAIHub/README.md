# Trading-GlobalAIHub
"Trading" project of "Python for Machine Learning" course of "globalaihub.com"

<hr />

### Trading
The last company you need to digitalize is the trading one.

You are expected to create a method to update the sales and expenses and calculate the revenue.

As with the previous companies, you need to create a class.

📌 Use the keyword "class" to create the class "Trading" 

📌 Use the "__init__" method to initialze the object's attributes: self, company_name, foundation_year, founder_name,  company_slogan, sales, expenses, and revenue.

#### Define the methods

Next, define the methods of the class:

1. print_report method: A method to print out information about the company. 

  📌 The report should have this structure: 

  "The company *name*, was founded in *year*. The founder of the company is *founder_name*. 

  Company slogan: *company_slogan*

  Total sales: *sales*

  Total expenses: *expenses*

  Total revenue: *revenue*


2. update_sales_and_expenses method: A method to update the sales and expenses.
It should consider the parameters "new_sales" and "new_expenses".

3. calculate_revenue method: A method  which computes the difference between sales and expenses and prints out the revenue.

```Python
#Define the "Trading" class
class Trading:
    #Initialize the object's attributes 
    def __init__(self,company_name,foundation_year,founder_name,company_slogan,sales,expenses,revenue):
      self.company_name = company_name
      self.foundation_year = foundation_year
      self.founder_name = founder_name
      self.company_slogan = company_slogan
      self.sales = sales
      self.expenses = expenses
      self.revenue = revenue

    #Define the print_report method
    def print_report(self):
      print(f"""
      The company name is {self.company_name} and was founded in {self.foundation_year}.
      The founder of the company is {self.founder_name}
      Company slogan: {self.company_slogan}
      Total sales: {self.sales}
      Total expenses: {self.expenses}
      Total revenue: {self.revenue}""")

    #Define the update_sales_and_expenses method
    def update_sales_and_expenses(self,new_sales,new_expenses):
      self.sales += new_sales
      self.expenses += new_expenses
      print("self and expenses are updated!")

    #Define the calculate_revenue method
    def calculate_revenue(self):
      self.revenue = self.sales - self.expenses
      print(f"The revenue of the company is: {self.revenue}")
```

#### Create the object 
Now that we created our class and initialized its attributes and methods, we can create the object “trading_company1” with the attributes: 

* Company_name “TraCom”

* foundation_year “2005”

* founder_name “Chong Wu”

* company_slogan “We revolutionize trading.” 

* sales "5000"

* expenses "2000"

* revenue "3000"

```Python
#Create the object "trading_company1" with it's attributes
trading_company1 = Trading("TraCom",2005,"Chong Wu","We revolutionize trading.",5000,2000,3000)
```

#### Let’s check if the methods work.
Update the sales and expenses. The sales went up by 100 and the expenses increased by 50.

📌 Use the update_sales_and_expenses method with this information.

```Python
#Update the sales and expenses
trading_company1.update_sales_and_expenses(100,50)
```

After updating sale and expense informations, it is time to calculate the company revenue.

📌 Use the calculate revenue method.

```Python
#Calculate the revenue
trading_company1.calculate_revenue()
```

Finally, let’s check our company’s information.

```Python
#Call the print_report method for trading_company1
trading_company1.print_report()
```
