# Shopping-GlobalAIHub
"Shopping" project of "Python for Machine Learning" course of "globalaihub.com"
<hr />

# Chapter 1
### Your first day at your new job 👩‍💻👨‍💻

You are starting a new job as a junior software developer in an IT company. 

The company’s HR department asks you to fill out a form, so you start by assigning your personal information to corresponding variables.

📌 Create a variable for your name, surname, age, ID number, place of residence, to specify if you have active health insurance or not, and lastly one for specifying your nationality.


```Python
#Please assign your personal information to variables
my_name = "Tohid"
my_surname = "Yousefi"
my_age = 29
ID_number = 123456789
where_i_live = "Turkiye"
health_insurance = True
#my_nationality = "Iran"
```
### Meet And Greet
Introduce yourself to your new co-workers.

📌 Use a f-string to print "My name is Joey Tribbiani I am 25 years old and I live in London”.

```Python
#Write a sentence using the print function to describe yourself using the variables above in the correct data type
print(f"My name is {my_name} {my_surname} I am {str(my_age)} years old and I live in {where_i_live}")
```

### Equipment starter pack
The HR department asks you to list the items you would need to improve your work efficiency

Mandatory:
* Laptop
* Headset
* Second monitor

Optional:
* Mousepad
* USB drive
* External drive


📌 Create a shopping list that contains items above and print it.

```Python
#Create the item_list
item_list = ["Laptop","Headset","Second monitor","Mousepad","USB drive","External drive"]

#Print the list
print(item_list)
```

#### What is mandatory and what is optional?

📌 Use list slicing to devide your list in two list: 'mandatory_item_list' and 'optional_item_list' and print both to the screen.

```Python
#Use list slicing to divide the mandatory items
mandatory_item_list = item_list[0:3]

#Use list slicing to divide the optional items
optional_item_list = item_list[3:]

['Laptop', 'Headset', 'Second monitor']
['Mousepad', 'USB drive', 'External drive']
```

#### Go Shopping
Next, you will have to go and purchase these items, the finance department confirmed a budget of $5000.

📌 Assign 5000 to a variable called limit, so you know how much you can spend.

```Python
#Assign the spending limit value to a variable called limit
limit = 5000
```

#### Price dictionary

Before you start shopping yo need to find the best items that you can buy within the company budget. 

📌 Prepare a dictionary called “price_sheet” that includes the items as keys and the prices as values.  
 
 ```Python
 #Create a dictionary that contains each item and its price
price_sheet = {'Laptop':1500,
               'Headset': 100,
               'Second monitor': 200,
               'Mousepad':50,
               'USB drive': 70,
               'External drive': 250}
 ```
 
 #### Shopping functions

You need to define three functions that will help you during shopping.

📌 First, create an empty list that  will be your shopping cart. Here you will add the items you need to purchase.

1. Define a function for both adding items to the cart and removing them from the item_list.

📌 The "add_to_cart" function should take the item name and the quantity to buy as an argument. 

2. Define a function that will create an invoice. 

📌 The "create_invoice" function should calculate the taxes of each item (18%) and add it to the total amount.

3. Define a function for the checkout. 

📌 The "checkout" function should subtract the total amount from the budget and print a statement to inform if the payment was successful. 

```Python
#Initialize the cart list
cart = []

#Define the "add_to_cart" function
def add_to_cart(item,quantity):
  cart.append((item,quantity))
  item_list.remove(item)
  
  #Define the "create_invoice" function
def create_invoice():
  total_amount_inc_tax = 0
  for item,quantity in cart:
    price = price_sheet[item]
    tax = 0.18 * price
    total = (tax + price) * quantity
    total_amount_inc_tax += total
    print('Item: ', item, '\t', 'Price: ', price, '\t', 'Quantity', quantity, '\t', 'Tax: ', tax, 'Total: ', total, '\t', '\n')
  
  print('After the taxes are applied the total amount is: ', total_amount_inc_tax)

  return total_amount_inc_tax
  
  #Define the "checkout" function
def checkout():
  global limit
  total_amount = create_invoice()
  if limit == 0:
    print("You done have any budget!")
  elif total_amount > limit:
    print("The amount you have to pay is above the spending limit. You have to drop some items.")
  else:
    limit -= total_amount
  print(f"The total amount you have paid is {total_amount}. You have {limit} dolars left")
  
  #Call the "add_to_cart" function for each item
 
#Add first item to cart
add_to_cart('Laptop',1)
 
#Add second item to cart
add_to_cart('Headset',8)
 
#Add third item to cart
add_to_cart('Second monitor',1)
 
#Add fourth item to cart
add_to_cart('Mousepad',1)
 
#Add fifth item to cart
add_to_cart('USB drive',2)
 
#Add last item to cart
add_to_cart('External drive',4)
 
#Call the create "checkout" function to pay for all your items 
checkout()

```
