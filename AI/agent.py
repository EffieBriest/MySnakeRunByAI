import numpy as np
import random

class Agent:

    def __init__(self, table):
        self.table = table


    #def chooseAction(self, state, oldDirection, epsilon):
        # select random action (exploration)
        #if (random.random() < epsilon):
        #    return random.choice([0, 1, 2, 3])
        # select best action (exploitation)
        #else:
        #    qValuesDependingState = self.table[state] 
        #    return np.argmax(qValuesDependingState)

    def chooseAction(self, state, oldDirection, epsilon):
        # select random action (exploration)
        actionsAvailable = [0,1,2,3]
        if (random.random() < epsilon):
            newAction = random.choice([0, 1, 2, 3])
            if(self.isActionValid(oldDirection, newAction)):
                return newAction
            else:
                actionsAvailable.remove(newAction)
                return random.choice(actionsAvailable)
        # select best action (exploitation)
        qValuesDependingState = self.table[state] 
        newAction = np.argmax(qValuesDependingState)
        if(self.isActionValid(oldDirection, newAction)):
            return newAction
        else:
            i = np.argmax(qValuesDependingState)
            leftValues = np.delete(qValuesDependingState,i)
            if(len(leftValues)<=2):
                print("here went something wrong")
            return np.argmax(leftValues)

    def setTable(self, table):
        self.table = table

    def updateAgentTableValue(self, currentState, action, newValue):
        self.table[currentState][action] = newValue
        
    def isActionValid(self, direction, action):
        if(direction == 0 and action == 1):
            return False
        if(direction == 1 and action == 0):
            return False
        if(direction == 2 and action == 3):
            return False
        if(direction == 3 and action == 2):
            return False
        return True
        