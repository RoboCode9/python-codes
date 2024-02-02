def hint_username(username):
    if len(username) < 8:
        print('invalid username. Must be at least 8 characters long.')
        return True
    elif len(username) > 15:
        print('invalid username. Cannot exceed 15 characters.')
        return True
    else:
        print('Valid username.')
        return False

x = True

while x:
    username = str(input("Enter a username, must be greater than 8 charatcers long, but less than 15: "))
    x = hint_username(username)

print(5%2) #show remainder
print(11%3)
print(10%2)

def is_even(number):
    if number % 2 == 0:
        return True
    return False

print(is_even(19))