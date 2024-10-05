import matplotlib.pyplot as plt        
import pickle
import numpy as np

#get values from file
filename = f'Projects\\MySnakeRunByAI\\AI\\pickle\\SARSA\\dataR40DN30ON1two.pickle'
with open(filename, 'rb') as file:
    table = np.zeros(10000)
    table = pickle.load(file)
    

#group those values
    groupedValues = []

#put Values into the array
    index = []
    for i in range(100):
         groupedValues.append(np.sum(table[i*100:(i+1)*100])/100)
         
         index.append(i)
         #x         #y
plt.plot(index, groupedValues)
plt.xlabel('episode in chunks of 100')
plt.ylabel('Averaged Score')
plt.show()
