# Pet Program

from random import randrange

class Pet (object):
    """A VPet"""
    excitement_reduce = 3
    excitement_max = 10
    excitement_warning = 3
    food_reduce = 2
    food_max = 10
    food_warning = 3
    vocab = ['"Arr..."', '"Hello"']

    def __init__(self, name, animal_type):
        self.name = name
        self.animal_type = animal_type
        self.food = randrange(self.food_max)
        self.excitement = randrange(self.excitement_max)
        self.vocab = self.vocab[:]

    def __clock__tick(self):
        self.excitement -= 1
        self.food -= 1

    @property
    def mood(self):
        if self.food > self.food_warning and self.excitement > self.excitement_warning:
            return "happy"
        elif self.food < self.food_warning:
            return "hungry"
        else:
            return "bored"
        
    def __str__(self):
        return "\nI'm" + self.name + "\nI feel " + self.mood() + "."