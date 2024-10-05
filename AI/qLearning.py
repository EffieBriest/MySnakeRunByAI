import numpy as np
import AI.agent
import Projects.MySnakeRunByAI.Game.snakeGame as snakeGame
import pickle

class QLearning:

    def __init__(self, showVisuals):
        self.discountRate = 0.95
        self.learningRate = 0.01
        self.eps = 1.0
        self.table = np.zeros((2,2,2, 2,2,2, 2,2,2, 2,2,2, 4))
        self.agent = AI.agent.Agent(self.table)
        self.game = snakeGame.Game(showVisuals)
        self.epsDiscount = 0.9992
        self.minEps = 0.001
        self.numMaxEpisodes = 10000
        self.data = []
        
        
    def train(self):
        self.agent.setTable(self.table)
        while self.game.episode in range(1, self.numMaxEpisodes + 1):
            # print updates
            if (self.game.episode == 50) or (self.game.episode == 500) or (self.game.episode == 1000) or (self.game.episode == 1500) or (self.game.episode == 2500) or (self.game.episode == 3000) or (self.game.episode == 3500) or (self.game.episode == 5000) or (self.game.episode == 6000) or (self.game.episode==10000):
                with open(f'Projects\\MySnakeRunByAI\\AI\\pickle\{self.game.episode}.pickle', 'wb') as file:
                    pickle.dump(self.table, file)
            done = False
            stepsWithoutFood = 0
            currentState = self.game.evaluateEnvironment()
            self.eps = self.eps * self.epsDiscount
            while not done:
                # choose action and take it
                action = self.agent.chooseAction(currentState, self.game.snake.direction, self.eps)
                newState, reward, done = self.game.run(action)
                    
                # Bellman Equation Update
                self.table[currentState][action] = (1 - self.learningRate)* self.table[currentState][action] + self.learningRate* (reward + self.discountRate * max(self.table[newState]))
                self.agent.updateAgentTableValue(currentState, action, self.table[currentState][action])
                a = self.table[currentState]
                currentState = newState

                stepsWithoutFood += 1
                if(stepsWithoutFood ==1000):
                     reward -= 1
                if(done):
                    self.data.append(self.game.snake.length-1)
                    self.game.reset()
        #dump score into file to plot later
        with open(f'Projects\\MySnakeRunByAI\\AI\pickle\\rewardDeathNegativeR40DN30ON1.pickle', 'wb') as file:
                    pickle.dump(self.data, file)

    def demonstrateResult(self, file, mode):
            # pass in pickle file with q table (stored in directory pickle with file name being episode #.pickle)
        epsilon = 0
        if mode == 'QLEARNING':
             epsilon = 0
        if mode == 'SARSA':
             epsilon = 0.001
        filename = f'Projects\\MySnakeRunByAI\\AI\pickle\\{file}.pickle'
        with open(filename, 'rb') as file:
            table = np.zeros((2,2,2, 2,2,2, 2,2,2, 2,2,2, 4))
            table = pickle.load(file)
            self.table = table
        self.agent.setTable(self.table)
        done = False
        currentState = self.game.evaluateEnvironment()
        while not done:
                # choose action and take it
                action = self.agent.chooseAction(currentState, self.game.snake.direction, epsilon)
                newState, reward, done = self.game.run(action)
                currentState = newState
