#Please create a list containing the three actions of the game.
action_list = ['rock','paper','scissors']

#Import the random library
import random

#Select a random action for each player
player1_choice = random.choice(action_list)
player2_choice = action_list[random.randint(0,2)]

#Use the print function to print the players choices
print("Player1: ", player1_choice)
print("Player2: ", player2_choice)

#1 - Tie Condition
#Please write an if statement to check if the players chose the same action
if player1_choice == player2_choice:
  print("Tie! Both player chose same action.")

#Copy your code of the tie-conditions here
#1 - Tie Condition
if player1_choice == player2_choice:
  print("Tie! Both player chose same action.")

#2-Winning Conditions
#Please add the conditional statements for the remaining combinations 
if player1_choice == player2_choice:
  print("Tie! Both player choice same action.")

#Method 2 Nested if statements

elif player1_choice == 'paper':
  if player2_choice == 'rock':
    print("Winner is: Player 1")
  else:
    print("Winner is: Player 2")

elif player1_choice == 'rock':
  if player2_choice == 'paper':
    print("Winner is: Player 2")
  else:
    print("Winner is: Player 1")

elif player1_choice == 'scissors':
  if player2_choice == 'paper':
    print("Winner is: Player 1")
  else:
    print("Winner is: Player 2")
  
#Please ask the user how many rounds they want to play
total_round = input("How many rounds do you want to play? ")

#Scores of players
player1_score = 0
player2_score = 0

#Collect all the components of your program to run it in a for loop
#Import the random library
import random

#Add the code to create a list containing the three actions of the game.
action_list = ['rock','paper','scissors']

#Add the code to set the scores of players to 0
player1_score = 0
player2_score = 0

#Add the code to ask the user how many rounds they want to play
total_score = int(input("How many rounds do you want to play? "))

#Write a for loop and put the game inside
for i in range(total_score):
  
  #Add the code to select a random action for each player
  player1_choice = random.choice(action_list)
  player2_choice = action_list[random.randint(0,2)]

  #Add the code to print the players choices
  print("Player1: ",player1_choice)
  print("Player2: ",player2_choice)

  #Add the tie condition
  if player1_choice == player2_choice:
    print("Tie! Both player chose same action.")
  
  #Add the remaining condition
  elif player1_choice == 'paper':
    if player2_choice == 'rock':
      print("Winner is: Player 1")
      player1_score +=1
    else:
      print("Winner is: Player 2")
      player2_score +=1

  elif player1_choice == 'rock':
    if player2_choice == 'paper':
      print("Winner is: Player 2")
      player2_score +=1
    else:
      print("Winner is: Player 1")
      player1_score +=1

  elif player1_choice == 'scissors':
    if player2_choice == 'paper':
      print("Winner is: Player 1")
      player1_score +=1
    else:
      print("Winner is: Player 2")
      player2_score +=1

  #print the score
  print("Score:", "Player1 =",player1_score, "Player2 =",player2_score)
  
  #Collect all the components of your program to run it in a while loop
#Import the random library
import random

#Add the code to create a list containing the three actions of the game.
action_list = ['rock','paper','scissors']

#Add the code to set the scores of players to 0
player1_score = 0
player2_score = 0

#Write a while loop and put the game inside
while True:
  #Add the code to select a random action for each player
  player1_choice = random.choice(action_list)
  player2_chice = action_list[random.randint(0,2)]

  #Add the code to print the players choices
  print("Player1: ",player1_choice)
  print("Player2: ", player2_choice)

  #Add the tie condition
  if player1_choice == player2_choice:
    print("Tie! Both player chose same action.")
    
  #Add the remaining condition
  elif player1_choice == 'paper':
    if player2_choice == 'rock':
      print("Winner is: Player 1")
      player1_score +=1
    else:
      print("Winner is: Player 2")
      player2_score +=1

  elif player1_choice == 'rock':
    if player2_choice == 'paper':
      print("Winner is: Player 2")
      player2_score +=1
    else:
      print("Winner is: Player 1")
      player1_score +=1

  elif player1_choice == 'scissors':
    if player2_choice == 'paper':
      print("Winner is: Player 1")
      player1_score +=1
    else:
      print("Winner is: Player 2")
      player2_score +=1

  #print the score
  print("Score:", "Player1 =",player1_score, "Player2 =",player2_score)
  
  #Collect all the components of your program to run it in a while loop
#Import the random library
import random

#Add the code to create a list containing the three actions of the game.
action_list = ['rock', 'paper', 'scissors']

#Add the code to set the scores of players to 0
player1_score = 0
player2_score = 0
#Add a round_counter that is 0 at the beginning
round_counter = 0

#Add the code to ask the user how many rounds they want to play
total_round = input("How many rounds do you want to play? ")

#Write a while loop and put the game inside
while True:

  #increase round_counter by 1 and print it
  round_counter +=1
  print("Round number:", round_counter)

  #Add the code to select a random action for each player
  player1_choice = random.choice(action_list)
  player2_choice = action_list[random.randint(0,2)]

  #Add the code to print the players choices
  print("Player1:", player1_choice)
  print("Player2:", player2_choice)

  #Add the tie condition
  if player1_choice == player2_choice:
    print("Tie! Both player chose same action.")

  #Add the remaining condition
  elif player1_choice == 'paper':
    if player2_choice == 'rock':
      print("Winner is: Player 1")
      player1_score +=1
    else:
      print("Winner is: Player 2")
      player2_score +=1

  elif player1_choice == 'rock':
    if player2_choice == 'paper':
      print("Winner is: Player 2")
      player2_score +=1
    else:
      print("Winner is: Player 1")
      player1_score +=1

  elif player1_choice == 'scissors':
    if player2_choice == 'paper':
      print("Winner is: Player 1")
      player1_score +=1
    else:
      print("Winner is: Player 2")
      player2_score +=1

  #print the score
  print("Score:", "Player1 =",player1_score, "Player2 =",player2_score)

  #stop the while loop if the round_counter equals the number of total rounds
  if round_counter == int(total_round):
    break
#Print the outcome of the game by using conditional statements
if player1_score == player2_score:
  print("There is no winner, tie.")
elif player1_score > player2_score:
  print("Player 1 won with score", player1_score, ":", player2_score)
elif player1_score < player2_score:
  print("Player 2 won with score", player1_score, ":", player2_score) 
