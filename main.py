import pygame
import AI.ReinforcementLearning as ReinforcementLearning

def main():
    # Start the training process
    qLearning = ReinforcementLearning.ReinforecementLearning()
    qLearning.train('SARSA')

    #    qLearning.demonstrateResult(10000, 'QLEARNING')

    pygame.quit()  

if __name__ == "__main__":
    main()
