# Logistic-GlobalAIHub
"Logistic" project of "Python for Machine Learning" course of "globalaihub.com"

<hr />

### Logistic
Let's continue with the next company from your brief.

Storage plays a crucial part in logistics, and the company wants to track the status of inventory space with software and be able to make changes in the system whenever it is needed.

Again, you need to create a class.

📌 Use the keyword "class" to create the class "Logistics" 

📌 Use the "__init__" method to initialze the object's attributes: self, company_name, foundation_year, founder_name,  company_slogan, and inventory_space.

#### Define the methods

Next, define the methods of the class:

1. print_report method: A method to print out information about the company. 

  📌 The report should have this structure: 

  "The company *name*, was founded in *year*. The founder of the company is *founder_name*. 

  Company slogan: *company_slogan*

  Inventory space of the company: *inventory_space*


2. update_inventory_space method: A method to update its inventory_space, and print a statement to describe the change.
It should consider the parameter "new_storage_space".

```Python
#Define the "Logistics" class
class logistic:
    #Initialize the object's attributes 
    def __init__(self,company_name,foundation_year,founder_name,company_slogan,inventory_space):
      self.company_name = company_name
      self.foundation_year = foundation_year
      self.founder_name = founder_name
      self.company_slogan = company_slogan
      self.inventory_space = inventory_space
        
    #Define the print_report method
    def print_report(self):
      print(f"""
      The Company name is {self.company_name} and was founded in {self.foundation_year}
      The founder of the company is {self.founder_name}.
      Company slogan: {self.company_slogan}
      Inventory space of the company: {self.inventory_space}""")

    #Define the update_inventory_space method
    def update_inventory_space(self,new_storage_space):
      self.inventory_space = new_storage_space
      print(f"Inventory space has been changed to {self.inventory_space} !")
```
#### Create the object 
Now that we created our class and initialized its attributes and methods, we can create the object “logistic_company1” with the attributes: 

* Company_name “LogCom”

* foundation_year “1990”

* founder_name “Laura McCartey”

* company_slogan “There is no place we cannot reach.” 

* inventory_space “2500”

```Python
#Create the object "logistic_company1" with it's attributes
logistic_company1 = logistic("LogCom",1990,"Laura McCartey","There is no place we cannot reach.",2500)
```

#### Let’s check if the methods work.
The logistics company plans to buy a new warehouse which will increase the inventory_space to 3000. 

📌 Call the update_inventory_space method to reflect this change in the system and then use the “print_report” method.

```Python
#Update the inventory space
logistic_company1.update_inventory_space(3000)
#Call the print_report method for logistic_company1
logistic_company1.print_report()
```
