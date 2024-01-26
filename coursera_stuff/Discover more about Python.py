print("hello world")
print(22)
print((5 + 4) / 3)
country = 'Brazil'
age = 30
print(country)
print(age)
print(10**3 == 1000)
print(10 * 3 == 40)
print(10 * 3 == age)
if age >= 18:
    print("adult")
else:
    print("minor")
for number in [1, 2, 3, 4, 5]:
    print(number)
my_list = [3, 6, 9]

for x in my_list:
    print(x/3)

def is_adult(age):
    if age >= 18:
        print("adult")
    else:
        print("minor")

#call function
is_adult(14)

raw_list = [20, 25, 10, 5]
sorted(raw_list)

for age in raw_list:
    is_adult(age)