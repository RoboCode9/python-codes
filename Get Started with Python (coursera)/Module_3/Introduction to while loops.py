import random

number = random.randint(1, 25)
number_of_guesses = 0

while number_of_guesses < 5:
    print('Guess a number between 1 and 25: ')
    guess = input()
    guess = int(guess)
    number_of_guesses += 1

    if guess == number:
        break
    elif number_of_guesses == 5:
        break
    else:
        print('Nope try again.')

if guess == number:
    print('Correct! You found the number in ' + str(number_of_guesses) + ' tries!')
else:
    print('You couldnt find the number. The number was ' + str(number) + '.')


x = 0

while x < 5:
    print('Not there yet, x = ' + str(x))
    x += 1

print('x = ' + str(x))