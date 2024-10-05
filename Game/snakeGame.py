import Game.snake as snake
import Game.food as food
import pygame
from pygame.locals import *
import Game.snakeVisuals as snakeVisuals

class Game:

    def __init__(self):
        pygame.init()         
        self.screen_width = 600
        self.screen_height = 400
        self.SIZE = 10
        self.snake = snake.Snake()
        self.food = food.Food()
        self.snakeSpeed = 15
        self.episode=1

        self.visuals = snakeVisuals.snakeVisuals()



    def collisionCheck(self):
        #snake bites itself
        for i in range(1, self.snake.length-1):
            if(self.snake.x[0]==self.snake.x[i] and self.snake.y[0]==self.snake.y[i]):
                print('bit itself')
                return True 
        #food spawned on snake
            if(self.food.x==self.snake.x[i] and self.food.y==self.snake.y[i]):
                    self.food.moveToNewLocation()
        #snake colliding with the boundaries of the window
        if not(0<= self.snake.x[0]<=1200 and 0<=self.snake.y[0]<=800):
            print('wall')
            return True
        #snake catches food
        if(self.snake.x[0]==self.food.x and self.snake.y[0]==self.food.y):
            self.snake.increaseLength()
            self.food.moveToNewLocation()
            return 'FOOD'

    def reset(self):
        self.snake = snake.Snake()
        self.apple = food.Food()


    def run(self,action):
        done = False
        running = True
        self.visuals.updateScreen(self.food, self.snake, running,  self.snake.length-1, self.episode)
        reward = 0

        self.snake.direction = action
        self.snake.move()
        #time.sleep(self.snakeSpeed/100*((0.9)**(self.snake.length-1)))

        #evaluation
        if(self.collisionCheck()=='FOOD'):
            reward = +40
        if (self.collisionCheck()):
            print(f"Episode:{self.episode}")
            print(self.snake.length-1)
            done = True
            reward -= 30
            self.episode += 1  
        return self.evaluateEnvironment(), reward, done

    def evaluateEnvironment(self):
        state = []
        state.append(int(self.snake.direction == 0))
        state.append(int(self.snake.direction == 1))
        state.append(int(self.snake.direction == 2))
        state.append(int(self.snake.direction == 3))
        state.append(int(self.food.y < self.snake.y[0]))
        state.append(int(self.food.y > self.snake.y[0]))
        state.append(int(self.food.x < self.snake.x[0]))
        state.append(int(self.food.x > self.snake.x[0]))
        state.append(self.isUnsafe(0)) 
        state.append(self.isUnsafe(1))
        state.append(self.isUnsafe(2))
        state.append(self.isUnsafe(3))
        return tuple(state)
    
    def isUnsafe(self, direction):
        for i in range(self.snake.length-1):
            if (direction == 0):
                #wall ahead
                if(self.snake.y[0]==10):
                    return 1
                #body ahead
                if(self.snake.x[0] == self.snake.x[i] and self.snake.y[0]==self.snake.y[i+1]+10):
                    return 1
            if (direction == 1):
                #wall ahead
                if(self.snake.y[0]==790):
                    return 1
                #body ahead
                if(self.snake.x[0] == self.snake.x[i] and self.snake.y[0]==self.snake.y[i+1]+10):
                    return 1
            if (direction == 2):
                #wall ahead
                if(self.snake.x[0]==10):
                    return 1
                #body ahead
                if(self.snake.y[0] == self.snake.y[i] and self.snake.x[0]==self.snake.x[i+1]-10):
                    return 1
            if (direction == 3):
                #wall ahead
                if(self.snake.x[0]==1190):
                    return 1
                #body ahead
                if(self.snake.y[0] == self.snake.y[i] and self.snake.x[0]==self.snake.x[i+1]+10):
                    return 1
        else:
          return 0
        