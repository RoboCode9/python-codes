import random

days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"] #create list of days
activity = [] #for activities

for i in range(7): #iterate over a range of 7
    task = input(f'Enter activity #{i + 1}: ') #gather user input then add it to task variable
    activity.append(task) #append it to activity list

random.shuffle(activity) #shuffle the activity list
print("") #print blank line for better view

for day, task in zip(days_of_week, activity): #iterate over both day_of_week and activity lists, if i only used "task" it would look like this: ('sunday', 'jump') etc.
    print(f'This {day} you will: {task}')