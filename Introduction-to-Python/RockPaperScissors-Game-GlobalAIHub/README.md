# RockPaperScissors-Game-GlobalAIHub
"Rock Paper Scissors" project of "Introduction to Python" course of "globalaihub.com"

<hr />

# You are a game developer :)

In this practical exercise you are a game developer. You will program your first game – Rock, Paper, Scissors.

If you are not familiar with the game, you can read about it here: https://en.wikipedia.org/wiki/Rock_paper_scissors.

You will program the game step by step. Please read the instructions carefully. There are hints and important explanations that will help you with this exercise.

## Warm up

Each player can choose between the three actions "rock", "paper" and "scissors". First, you need to create a list that contains all possible actions. 

```Python
#Please create a list containing the three actions of the game.
action_list = ['rock','paper','scissors']
```

# Choose an action

Now you have the list with the actions ready. The next step is to let Python randomly choose an action from this list. Each player has to make a choice, so Python has to make a choice for player 1 and another one for player 2.

📌The built-in module "random" can help you with this. As you might remember from previous lessons you can use the .randint() method to generate random numbers.

📌📌There is another method in the random module called .choice(). It takes a list as input and selects an item randomly from that list.


```Python
#Import the random library
import random

#Select a random action for each player
player1_choice = random.choice(action_list)
player2_choice = action_list[random.randint(0,2)]

#Use the print function to print the players choices
print("Player1: ", player1_choice)
print("Player2: ", player2_choice)
```

# Determine the winner

So, who won? To determine the winner, you first have to specify all possible outcomes of the game. Then you can use conditional statements to specify which outcome will lead to a win for player 1 or player 2, or a tie.

First, let's start with the outcome that both users chose the same action: a tie!
In case of a tie, the game should print the message "Tie! Both players chose the same action." to the screen.

📌Use conditional statements to check whether the players actions are equal to each other and *if* this condition is true the message should be printed.

```Python
#1 - Tie Condition
#Please write an if statement to check if the players chose the same action
if player1_choice == player2_choice:
  print("Tie! Both player chose same action.")
```

Now, let's focus on the remaining outcomes. We have three different actions: rock, paper and scissors.

This is a list of all the possible combinations (without those that result in a tie):

1.   (paper, rock)
2.   (paper, scissors)
3.   (scissors, rock)
4.   (scissors, paper)
5.   (rock, paper)
6.   (rock, scissors)

You need to check all these conditions in order to determine the winner.

📌You can use the "and" operator to combine two actions in one conditional statement.

📌📌Another approach is using nested if statements to check whether the action of player 2 beats the action of player 1.

🧨💣In the cell, first copy the code you wrote for "Tie" conditions. Then continue to add the other outcomes using "elif".


```Python
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
```

# Best of 3 or 5 or single round ? :)

Now that you have your game set, let's also ask the users how many rounds they want to play.

📌 You can use the input function for this.

And because there are multiple rounds, you need to keep track of the score. For this, you need to create a variable for each player that is 0 in the beginning.

```Python
#Please ask the user how many rounds they want to play
total_round = input("How many rounds do you want to play? ")

#Scores of players
player1_score = 0
player2_score = 0
```

# Use a for loop to specify the number of rounds 
The user specified how many rounds he or she wants to play. 

One way to use this information is to put the game inside of a for loop.

📌 You can use a for loop and the range() function to repeat the loop for a specified number of times.

📌📌 Don't forget to convert the user input to the data type integer. You can use the int() function for this.

📌📌📌 Don't forget to increment or add 1 to the score of the players that win, and print the current score after each round. You can use the assignment operator +=1 for this. 

```Python
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
```

# Use a while loop to specify the number of rounds - Part 1

In this module we also learned about while loops. Another way to play the game over and over again is using a while loop. 

This time, let's put the game inside a while loop.

📌 You can directly use the value *True* to specify that the while loop should be executed. 

💣 By the way: Another very common name for True/False values is "**Boolean values**" or "**Bools**" in short.

🧨🧨 Caution! Since we run the game in an while loop, you have to stop the game manually. Otherwise it will run infinitely.


```Python
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
```

# Use a while loop to specify the number of rounds - Part 2

Now the game runs infinitely and the user can end it manually. But there is a more efficient way to do this: Let's ask the user again how many rounds they want to play, just as you did for the game inside the for loop.

To prevent an infinite loop, you need to add a round_counter that is 0 at the beginning. In each round, this counter needs to increase by 1. 

At the end of your while loop, you need to add an "if" condition to check whether the round_counter is equal to the number of rounds that we got from user as an input. If this is the case, we need to use "**break**" to end the game.

Once we end the game, it would be nice to see a message that prints the winner to the screen. In case of a tie, it should print a message that the game ended in a tie.

📌 You can use "if" and "elif" statements to determine the winner at the end.

```Python
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
```
