print([1, 2, 3] + [2, 4, 6]) # it will combine both these lists

#a simple function in python
def to_celsius(x):
    '''Convert to Celsius'''
    celsius = (x - 32) * 5/9 
    return print(f"{celsius:.2f}")

to_celsius(75) #"calling the function"

#conditional statements
number = -4

if number > 0:
   print('Number is positive.')
elif number == 0:
   print('Number is zero.')
else:
   print('Number is negative.')