import random

days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",] #create list of days
activity = [] #for activities
color = ['Red', 'Red', 'Red', 'Blue'] # create a list of colors.
random.shuffle(color)

for i in range(5): #iterate over a range of 7
    task = input(f'Enter activity #{i + 1}: ') #gather user input then add it to task variable
    activity.append(task) #append it to activity list

random_color = random.choice(color)#randomly select from the color list

if random_color == 'Blue': #if the selected color is 'Blue'
    random_index = random.choice(range(len(activity))) #randomly select from the activity list and add that index value to random_index variable
    activity[random_index] = random_color #replace selected activity index caused by the random_index with the random color, which should be blue

random.shuffle(activity) #shuffle the activity list
print("") #print blank line for better view

for day, task in zip(days_of_week, activity): #iterate over both day_of_week and activity lists, if i only used "task" it would look like this: ('sunday', 'jump') etc.
    print(f'This {day} you will: {task}')