import pygame
import AI.ReinforcementLearning as ReinforcementLearning
import matplotlib as plt
import numpy as np

def main():
    # Start the training process
    qLearning = ReinforcementLearning.ReinforecementLearning()
    qLearning.train('SARSA')




    #put Values into the array
    #reward=[]
    #index = []
    #for i in range(100):
    #    qLearning.demonstrateResult(10000, 'QLEARNING')
    #    reward.append(qLearning.game.snake.length)
    #    qLearning.game.reset()
    #    index.append(i)
         #x         #y
    #print(np.sum(reward)/100)s
    pygame.quit()  

if __name__ == "__main__":
    main()
