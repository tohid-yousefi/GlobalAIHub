#Please assign your personal information to variables
my_name = "Tohid"
my_surname = "Yousefi"
my_age = 29
ID_number = 123456789
where_i_live = "Turkiye"
health_insurance = True
#my_nationality = "Iran"

#Write a sentence using the print function to describe yourself using the variables above in the correct data type
print(f"My name is {my_name} {my_surname} I am {str(my_age)} years old and I live in {where_i_live}")

#Create the item_list
item_list = ["Laptop","Headset","Second monitor","Mousepad","USB drive","External drive"]

#Print the list
print(item_list)

#Use list slicing to divide the mandatory items
mandatory_item_list = item_list[0:3]

#Use list slicing to divide the optional items
optional_item_list = item_list[3:]

['Laptop', 'Headset', 'Second monitor']
['Mousepad', 'USB drive', 'External drive']

#Assign the spending limit value to a variable called limit
limit = 5000

#Create a dictionary that contains each item and its price
price_sheet = {'Laptop':1500,
              'Headset': 100,
              'Second monitor': 200,
              'Mousepad':50,
              'USB drive': 70,
              'External drive': 250}

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
