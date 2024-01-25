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

    def __clock_tick(self):
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
    
    def teach(self, word):
        self.vocab.append(word)
        self.__clock_tick()

    def talk(self):
        print(
            "I am a ",
            self.animal_type,
            " named "
            self.name,
            ".",
            "I feel ",
            self.mood(),
            " now.\n"
        )

        self.__clock_tick()

    def feed(self):
        print("***munch*** \n yum!")
        meal = randrange(self.food, self.full_max)
        self.food += meal

        if self.food < 0:
            self.food = 0
            print("I still need food!")
        elif self.food > self.food_max:
            self.food = self.food_max
            print("I'm full.")
        self.__clock_tick()

    def play(self):
        print("Yes!")
        fun = randrange(self.excitement, self.excitement_max)
        self.excitement += fun
        if self.excitement < 0:
            self.excitement = 0
            print("Bored...")
        elif self.excitement > self.excitement_max:
            self.excitement = self.excitement_max
            print("I am happy")
        self.__clock_tick