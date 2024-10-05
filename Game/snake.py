from pygame.locals import *

SIZE = 10
class Snake:

    def __init__(self):
        self.direction = 4 #0= UP, 1=DOWN, 2=LEFT, 3=RIGHT 
        self.length = 1
        self.x = [2000]
        self.y = [2000]
        self.x[0]=600
        self.y[0]=400

    def move(self):
        #update body
        for i in range(self.length-1,0,-1):
            self.x[i]=self.x[i-1]
            self.y[i]=self.y[i-1]
        #Move the head body            
        if self.direction == 0:
            self.y[0] -= SIZE
        if self.direction == 1:
            self.y[0] += SIZE
        if self.direction == 2:
            self.x[0] -= SIZE
        if self.direction == 3:
            self.x[0] += SIZE


    def increaseLength(self):
         self.length += 1
         self.x.append(-1)
         self.y.append(-1)

    

        