#this is not efficiant instead try to make a function
name = "John"
number = len(name)*9
print("Hello " + name + "!" " Your lucky number is " + str(number) + ".")

name = "Marisol"
number = len(name)*9
print("Hello " + name + "!" " Your lucky number is " + str(number) + ".")

#this is more efficiant with the function
def lucky_number(name):
    number = len(name)*9
    print("Hello " + name + "!" " Your lucky number is " + str(number) + ".")

#collect user input, convert it to string, then call the lucky_number function.
lucky_number(str(input("What is your name to find your lucky number: ")))

def factorial(n):
    #exculde 0 as product, start with 1
    y = 1
    for i in range(int(n)):
        print("this is i: " + str(i))
        y = y * (i + 1)
        print("this is y: " + str(y))
        print("")
    return y

input_num = input("input a number to find its factorial: ")
print("the factorial is: " + str(factorial(input_num)) + ", for the number: " + str(input_num))