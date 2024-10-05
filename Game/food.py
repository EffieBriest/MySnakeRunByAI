import random

SIZE = 10

class Food:

    def __init__(self):
        self.x = random.randint(1,119)*SIZE
        self.y = random.randint(1,79)*SIZE


    def moveToNewLocation(self):
        self.x = random.randint(1,119)*SIZE
        self.y = random.randint(1,79)*SIZE

