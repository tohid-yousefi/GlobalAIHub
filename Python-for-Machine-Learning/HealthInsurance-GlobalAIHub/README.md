# HealthInsurance-GlobalAIHub
"Health Insurance" project of "Python for Machine Learning" course of "globalaihub.com"

<hr />

# Chapter 2
## Object-oriented programming

Your department receives a request to prepare a test software to digitalize the health insurance, logistics, and trading companies that can be expanded in the future.

You come up with an idea: object-oriented programming. Your approach is to create classes for each company that will have different attributes and methods to digitalize some of their functions.   

### Health insurance
#### Define the class
You start with the health insurance company. 

You need to create a system for the supplemental health insurance registration process by taking different variables like age, chronic disease, and income into account. 

📌 Use the keyword "class" to create the class "HealthInsurance" 

📌 Use the "__init__" method to initialze the object's attributes: self, company_name, foundation_year, founder_name,  company_slogan, num_of_employees and num_of_clients.

#### Define the methods

Next, define the methods of the class:

1. print_report method: A method to print out information about the company. 

  📌 The report should have this structure: 

  "The company *name*, was founded in *year*. The founder of the company is *founder_name*. 

  Company slogan: *company_slogan*

  Number of employees: *number of employees*

  Number of clients: *number of clients*"

2. sup_health_insurance method: A method to check the eligibility for supplemental health insurance. It should consider the parameters "age", "chronic_disease", and "income".

  📌 Use if-else statements and logical operators.

3. update_num_clients method: A method to update the attribute number of clients.

```Python
#Define the "HealthInsurance" class
class HealthInsurance:
    #Initialize the object's attributes
    def __init__(self,company_name,foundation_year,founder_name,company_slogan,num_of_employees,num_of_clients):
      self.company_name = company_name
      self.foundation_year = foundation_year
      self.founder_name = founder_name
      self.company_slogan = company_slogan
      self.num_of_employees = num_of_employees
      self.num_of_clients = num_of_clients

    #Define the print_report method
    def print_report(self):
      print(f"""The company {self.company_name} was founded in {self.foundation_year}. 
      The founder of company is {self.founder_name}.
      Company slogan: {self.company_slogan}
      Number of Employees: {self.num_of_employees}
      Number of Clients: {self.num_of_clients}""")

    #Define the sup_health_insurance method
    def sup_health_insurance(self,age,chronic_disease,income):
      #if-else statements to check whether person can get supplemental insurance or not
      if age > 60 and chronic_disease == True and income < 6000:
        print("We ar sorry! You are not eligable for supplemental health insurance.")
      elif age < 60 and income >=6000 or chronic_disease == False:
        print("Congratulations! You can get supplemental health insurance.")

    #Define the update_num_clients method
    def update_num_clients(self,new_numnber):
      self.num_of_clients = new_numnber
      print(f"Number of clients has been changed to {self.num_of_clients} !")
```

#### Create the object 
Now that we created our class and initialized its attributes and methods, we can create the object “HI_company1” with the attributes: 

* Company_name “Healthy”

* foundation_year “2012”

* founder_name “Bob Mayer”

* company_slogan “We care for you.” 

* num_of_employees “3500”

* num_of_clients “13230”

```Python
#Create the object "HI_company1" with it's attributes
HI_company1 = HealthInsurance("Healthy",2012,"Bob Mayer","We care for you",3500,13230)
```
#### Let’s check if the methods work.

There is a new customer that wants to register for supplemental health insurance. He is 45 years old, does not have a chronic disease, and has an income of 5000 dollars per month.

📌 Use the sup_health_insurance method with this information. 

```Python
#Use the sup_health_insurance for the new customer
HI_company1.sup_health_insurance(45,False,5000)
```

Because the number of clients has increased, you should use the update_num_clients method. 

📌 The new_number will be 13231.

```Python
#Update the number of clients
HI_company1.update_num_clients(13231)
```

To see the output of the latest update and check whether it worked or not, call the print_report method.

```Python
#Call the print_report method for HI_company1
HI_company1.print_report()
```
