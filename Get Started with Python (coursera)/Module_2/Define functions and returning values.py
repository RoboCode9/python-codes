print('Black dove')
number = 15
print(type(number))
number = str(number)
print(type(number))

#defining a function
def greeting(name):
    print('Welcome, ' + name + '!')
    print('You are now part of the team!')

#calling the greeting function
greeting('Rebecca')

#defining a function with a return statement
def area_triangle(base, height):
    return base * height / 2 # if no return is used lines 19 and 20 would have no value to store on area a and area b

area_a = area_triangle(5, 4)
area_b = area_triangle(7, 3)

total_area = area_a + area_b

print(total_area)

#defining a function that calculates the amount of seconds based on the hours, minutes, and seconds
def get_seconds(hours, minutes, seconds):
    total_seconds = 3600 * hours + 60 * minutes + seconds
    return total_seconds

print(get_seconds(16, 45, 20))