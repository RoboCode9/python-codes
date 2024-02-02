def seed_calculator(fountain_side, grass_width):
    '''
    Calculate number of kg of grass seed needed for a border around a sq. fountain.

        Parameters:
            fountain_side (num): length of 1 side of fountain in meters
            grass_width (num): width of grass border in meters

        Returns:
            seed (float): amount of seed (kg) needed for grass border
    '''
    #area of the fountain
    fountain_area = fountain_side**2
    #total area
    total_area = (fountain_side + 2 * grass_width) ** 2
    #area of the grass border
    grass_area = total_area - fountain_area
    #amount of seed needed (35 g/sq.a)
    seed = grass_area * 35
    #convert to kg
    seed = seed / 1000

    return seed

x = True
while x == True:
    try:
        fountain_input = input("What is the length of the fountain side: ")
        fountain_side = int(fountain_input)
        x = False
    except ValueError:
        print("invalid input. please enter a number!")
        continue

y = True
while y == True:
    try:
        grass_input = input("What is the width of the grass square: ")
        grass_width = int(grass_input)
        y = False
    except ValueError:
        print("invalid input. please enter a number!")
        continue

print(f"The amount of grass in kilograms needed for your border around the square fountain is: " + str(seed_calculator(fountain_side, grass_width)))