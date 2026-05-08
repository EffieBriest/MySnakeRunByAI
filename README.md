# Reinforcement Learning for Snake 
*Author: 吳斐力 (Felix Uhl)*

Reinforcement learning (RL) is a type of machine learning where an agent learns to interact with an environment by taking actions and observing the resulting states and rewards. The agent's objective is to develop a policy, a strategy for choosing actions, that maximizes cumulative rewards over time.   

### The Essence of Reinforcement Learning 
Imagine a child learning to ride a bicycle. They start with no knowledge of how to balance, pedal, or steer. Through trial and error, they experiment with different actions, observing the consequences. Successful actions, like staying upright and moving forward, are rewarded with a sense of accomplishment and progress. Unsuccessful actions, like falling off, result in discomfort and the need to try again. Gradually, the child learns to associate actions with their outcomes, refining their strategy until they can ride proficiently. This process of learning through trial and error, guided by feedback in the form of rewards and penalties, is the essence of reinforcement learning (RL).

### Key Concepts
* **Agent:** The learner and decision-maker (e.g., a robot, an AI playing a game).
* **Environment:** The world or system the agent interacts with (e.g., a physical environment, a game, a simulation).
* **State:** A representation of the current situation in the environment (e.g., the robot's position, the game board configuration).
* **Action:** A move the agent can make that influences the environment (e.g., move a joint, make a move in the game).
* **Reward:** A signal indicating the desirability of an outcome (e.g., reaching a goal, winning a game).
* **Policy:** A mapping from states to actions, dictating the agent's behavior.

### Importance of Reinforcement Learning
RL is a versatile approach with applications in various domains:

* **Robotics:** Training robots to perform complex tasks like grasping objects, navigating unfamiliar terrain, and collaborating with humans.
* **Game playing:** Creating AI agents that can master games like chess, Go, and video games.
* **Control systems:** Optimizing control systems for applications such as traffic light management, resource allocation, and personalized recommendations.
* **Finance:** Developing trading strategies, managing portfolios, and assessing risk.



## Markov Decision Process (MDP)
Reinforcement learning problems can be formally modeled as Markov Decision Processes (MDPs). An MDP provides a framework for making decisions in situations where outcomes are partly random. Think of it like navigating a maze where some doors might teleport you randomly to other locations.

![Folie1](https://hackmd.io/_uploads/SkmtYIA0R.png)

### Formal Definition 
A Markov Decision Process is defined by a tuple $(S, A, P, R, \gamma)$:

* $S$: A finite set of **states**. Each state represents a possible situation in the environment (e.g., the player's location in a game
* $A$: A finite set of **actions** that the agent can take in each state (e.g., move up, down, left, or right).
* $P$: The **state transition probability matrix**. $$P^a_{s,s'} = \mathbb{P}(S_{t+1}=s'| S_t=s, A_t = a)$$ represents the probability of transitioning to state $s'$ from state $s$ when action a is taken. This captures the randomness in the environment.
* $R$: The **reward function**. $$R_s^a=\mathbb{E}[R_{t+1}|S_t=s, A_t=a]$$ is the expected immediate reward received after taking action a in state s and transitioning to the next state.
* $\gamma$: The **discount factor**. A value between 0 and 1 that determines the importance of future rewards relative to immediate rewards. A higher discount factor prioritizes long-term gains.

### Key Properties and Concepts
* **Markov Property:** The future state depends only on the current state and action, not the entire history. This "memoryless" property simplifies the problem.
* **Total Reward:** The sum of discounted rewards over time:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots
= \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$

* **Policy** $\pi$: A policy is a function that maps states to actions, essentially defining the agent's strategy:

$$
\pi(a \mid s) = \mathbb{P}(A_t = a \mid S_t = s)
$$

It can be deterministic, always choosing the same action in a given state, or stochastic, choosing actions with some probability.

* **Value Function:** A value function estimates the "goodness" of being in a particular state.

  * **State-value function** $v_\pi(s)$: The expected total reward from a state $s$ when following policy $\pi$:

$$
v_\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s]
= \sum_{a \in A} \pi(a \mid s) q_\pi(s,a)
$$

  * **Action-value function** $q_\pi(s,a)$: The expected total reward from a state $s$ after taking action $a$ and then following policy $\pi$:

$$
q_\pi(s,a) = \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a]
= R_s^a + \gamma \sum_{s' \in S} P_{ss'}^a v_\pi(s')
$$

* **Optimal Policy** $\pi^*$: The policy that maximizes the expected total reward for all states. This is the ultimate goal of reinforcement learning.


### Why MDPs Matter in Reinforcement Learning
MDPs provide a formal way to represent and reason about sequential decision-making problems. They allow us to:

* **Define the problem clearly:** By specifying states, actions, transitions, and rewards.
* **Evaluate policies:** Using value functions to assess the long-term consequences of different strategies.
* **Find optimal policies:** Employing algorithms like Q-learning and SARSA to determine the best course of action.

Understanding MDPs is crucial for grasping the theoretical foundations of reinforcement learning and for effectively applying RL algorithms to solve real-world problems.



## RL for the Snake Game with Open AI Gym

### Game Implementation 
The Snake game is implemented using the Pygame library in Python. This involves creating the game window, displaying the snake, generating food, and handling user input (or AI agent actions).

#### Key Components
* `Food` Class: Handles the random generation of food within the game window, ensuring it doesn't overlap with the snake's body.

* `Snake` Class: Manages the snake's properties (length, position, direction) and its actions (movement, growing after eating food). It also includes collision detection (with itself or the walls).

* `Game` Class: Controls the overall game logic, including initializing the game, updating the game state, handling collisions, and displaying the score.

#### State Representation
Defining the state space is crucial in reinforcement learning. In this Snake game, we can represent the state using a combination of factors:

* Snake's Direction: Is the snake moving up, down, left, or right?
* Food Location: Is the food above, below, to the left, or to the right of the snake's head?
* Danger: Are there immediate obstacles (walls or the snake's body) in the directions the snake could move?

This combination of factors creates a discrete state space. For example, one possible state could be "moving right, food above, danger ahead and to the right."

    
### OpenAI Gym
OpenAI Gym is a popular Python library designed to facilitate the development and comparison of reinforcement learning (RL) algorithms. It provides a standardized interface for interacting with various environments, ranging from simple toy problems to complex simulations.   

##### Key Components:

* Environments:
Pre-built Environments: Gym offers a wide range of pre-built environments, such as:
    * Classic control problems (CartPole, Pendulum)   
    * Atari games   
    * Robotics simulations   
    * Custom Environments: You can create your own custom environments by defining the observation space, action space, and the dynamics of the environment.   
* Interaction Loop:
    The core interaction loop in Gym involves the following     steps:
    * Observation: The agent receives an observation from the environment, which represents the current state of the environment.   
    * Action: The agent selects an action based on the observation and its current policy.   
    * Reward: The environment executes the action and returns a reward to the agent, indicating the immediate consequence of the action.   
    * New Observation: The environment transitions to a new state and provides a new observation to the agent.   
    * Termination: The episode ends when a terminal state is reached or a maximum number of steps is exceeded.
**Implementation: Essential Components**

Gym environments, the fundamental building blocks of reinforcement learning, typically consist of four core functions:

* Initialization:
    - Defines the environment's parameters and state space.
    - Enables the creation of environment instances for use in other parts of the system.
* Step:
    - Takes an action as input from the agent.
    - Executes the action within the environment.
    - Evaluaties the action
    - Returns a tuple containing:
        *  The new observation (state) of the environment.
        * The reward associated with the taken action.
        * A boolean indicating whether the episode has terminated.
        * Additional information, such as diagnostic data.
* Reset:
    * Resets the environment to its initial state.
    * Returns the initial observation (state) to the agent.
* Render:
    * Visualizes the environment's state.
    * Can be used for debugging, analysis, or human interaction.

**Why These Functions Are Crucial**
Reinforcement learning libraries like Stable Baselines 3 rely on these four functions to interact with the environment during training and evaluation. By adhering to this standard structure, you ensure compatibility with a wide range of RL frameworks and libraries.

#### Implementation of the Environment 

To leverage this library, you typically follow these steps:

* Environment Setup:
    * Create a Gym environment that defines the rules and dynamics of the problem domain.
* Model Selection and Configuration:
    * Choose a suitable reinforcement learning algorithm from the library's extensive collection.
    * Configure the algorithm's hyperparameters to fine-tune its behavior.
* Learning:
    * Initiate the training process, allowing the agent to learn optimal policies through interaction with the environment.
    * The library handles the training loop, including gradient updates and policy improvement.
* Model Saving and Logging:
    * The library automatically saves trained models to a specified directory for future use or further analysis.
    * Training logs are generated and can be visualized using TensorBoard, providing insights into the learning process.



## Reinforcement Learning Algorithms
Reinforcement learning algorithms provide the means for an agent to learn an optimal policy within an MDP. These algorithms specify how the agent updates its understanding of the environment and improves its decision-making over time. Here we'll explore fundamental algorithms: Q-learning, SARSA, and Policy Gradient Methods.


### Classic Reinforcement Learning Algorithms

#### Q-Learning
Q-learning is an **off-policy** temporal difference learning algorithm. "Off-policy" means it learns about the optimal policy independent of the agent's current actions. It focuses on learning the optimal action-value function (Q-values), which estimate the expected total reward for taking a specific action in a given state and then following the optimal policy.

##### Key Mechanisms
* **Q-table:** A table where rows represent states and columns represent actions. Each cell (state-action pair) stores a Q-value.
* **Bellman Equation:** Used to iteratively update the Q-values based on observed rewards and the estimated Q-values of subsequent states. $$q_{new}(s, a) = (1 - \alpha) q_{old}(s, a) + \alpha \overbrace{\left((R_{t+1} + \gamma \max_{a'} q(s', a') \right)}^{\text{learned value}}$$ where:
    * $\alpha$ is the learning rate (how much to adjust the old Q-value).
    * $R_{t+1}$ is the immediate reward received after taking action $a$.
    * $\gamma$ is the discount factor.
    * $s'$ is the next state.
    * $\max_{a'} q(s', a')$ is the maximum Q-value among all possible actions in the next state (representing the expected value of following the optimal policy thereafter).
* **Exploration-Exploitation:** Q-learning often uses an $\epsilon$-greedy strategy to balance exploration (trying new actions) and exploitation (choosing actions with the highest known Q-values).
![Q-table Iteration](https://hackmd.io/_uploads/r11I5IAAR.png)


#### SARSA 
SARSA (State-Action-Reward-State-Action) is an on-policy temporal difference learning algorithm. "On-policy" means it learns the value of the policy being followed by the agent. It updates Q-values based on the action taken in the next state according to the current policy.

##### Key Mechanisms
* **Q-table:** Similar to Q-learning, SARSA also uses a Q-table to store Q-values.
* **Bellman Equation:** SARSA's update rule is slightly different from Q-learning: $$q_{\text{new}}(s, a) = (1-\alpha) q_{old}(s,a) + \alpha \overbrace{\left(R_{t+1}+\gamma q(s',a')\right)}^{\text{learned value}}$$
* **Exploration-Exploitation:** Like Q-learning, SARSA can use $\epsilon$-greedy or other strategies to balance exploration and exploitation.

#### Implementation of a custom QLearning/SARSA Model

In each episode:
* Observe the current state.
* Select an action based on the current policy (ε-greedy).
* Execute the action in the game environment.
* Observe the new state and reward.
* Update the Q-value for the previous state-action pair using the Bellman equation.


##### Training and Evaluation
* Algorithm Selection: Choose either Q-learning or SARSA to train the agent.
* Initialization: Initialize the Q-table (for Q-learning or SARSA) with zeros or small random values.
* Exploration-Exploitation: Use an ε-greedy strategy to balance exploration (random actions) and exploitation (choosing actions with the highest Q-values).
* Training Loop: Run the game for a certain number of episodes or until the agent achieves a desired level of performance.
* Evaluation: Monitor the agent's performance by tracking metrics like average score, number of steps survived, and how often it eats food.

![SARSAvsQLEARNINGEvualuation](https://hackmd.io/_uploads/BkXUGTl1Jg.jpg)
(States: Directions, Food, Danger 1 rectangle next to the snake head)


### Advanced RL-Algorithms
#### Neural Network

A neural network is a computational model inspired by the structure and function of the human brain. It consists of interconnected nodes, called neurons, organized in layers. Each neuron receives inputs, processes them, and produces an output. The connections between neurons are weighted, and these weights determine the strength of the connections.   

**Key components**

* **Input layer**: Receives data to be processed.
* **Hidden layers**: Perform complex computations on the data.
* **Output layer**: Produces the final result.
* **Weights**: Determine the strength of connections between neurons.
* **Activation function**: Introduces non-linearity into the network.

**How neural networks learn**:

* **Training**: The network is trained on a dataset of input-output pairs. Weights are adjusted iteratively using algorithms to minimize the error between the predicted and actual outputs.
* **Testing**: The trained network is evaluated on a separate dataset to assess its performance.
![Neural-Networks-Architecture](https://hackmd.io/_uploads/SJyQtuvg1l.png)

#### Deep Q-Network (DQN)

DQN is a deep learning technique used to solve reinforcement learning problems. It combines the power of deep neural networks with the principles of Q-learning to learn optimal policies in complex environments.

**The Role of Deep Neural Networks**

In DQN, a deep neural network is used to approximate the Q-function. The network takes the current state as input and outputs the Q-values for all possible actions.   
 The network's parameters are updated using gradient descent to minimize the loss function, which is typically the mean squared error between the predicted Q-values and the target Q-values.

**Experience Replay**

One of the key innovations in DQN is the use of experience replay. This technique stores the agent's experiences (state, action, reward, next state) in a replay buffer. By randomly sampling experiences from the replay buffer, the agent can learn from diverse and non-sequential data. This helps to break the correlation between consecutive experiences and improves the stability of the learning process.

**Target Network**
To further stabilize the training process, DQN employs a target network. The target network is a copy of the main network that is updated less frequently. This helps to reduce oscillations and improve convergence.

**DQN Algorithm**
1. **Initialize:**
   - Initialize the Q-network with random weights.
   - Initialize the replay buffer.
2. **Experience Collection:**
   - Interact with the environment, taking actions based on the current Q-network.
   - Store the experiences (state, action, reward, next state) in the replay buffer.
3. **Training:**
   - Sample a batch of experiences from the replay buffer.
   - Compute the target Q-values using the target network:
   \begin{align*}
   q_{target} = r + \gamma * max_a' \ q(s', a')
   \end{align*}    
   - Calculate the loss between the predicted Q-values and the target Q-values: 
   \begin{align*}
    Loss = \mathbb{E}(q(s, a) - q_{target})^2
   \end{align*}
   - Update the weights of the Q-network using gradient descent: 
   \begin{align*}
   W = W - \alpha  \nabla Loss
   \end{align*}
4. **Target Network Update:**
   - Periodically copy the weights of the Q-network to the target network.
5. **Repeat:**
   - Continue steps 2-4 until convergence or a maximum number of iterations.

By combining the power of deep neural networks with the principles of Q-learning and experience replay, DQN has been successfully applied to a wide range of complex reinforcement learning problems, including game playing and robotics.


#### Policy Gradient Methods
Policy gradient methods are a class of algorithms used in reinforcement learning that directly optimize the policy, which is a function that maps states to actions. Unlike value-based methods that learn the value of states, policy gradient methods directly optimize the parameters of the policy to maximize expected reward.

##### Proximal Policy Optimization (PPO)
**PPO** is a powerful policy gradient method that has gained significant popularity in reinforcement learning. It addresses some of the limitations of traditional policy gradient methods, such as the difficulty of choosing a suitable step size

1. **Surrogate Objective Function**
   - Instead of directly maximizing the expected return, PPO uses a surrogate objective function that is easier to optimize.
   - This surrogate function is based on the ratio of the new policy's probability to the old policy's probability for a given action.
2. **Clipping**
   - To prevent large policy updates that can lead to instability, PPO clips the ratio of the new policy to the old policy.
   - This clipping ensures that the policy updates are bounded, making the optimization process more stable.

## Algorithm

1. **Initialize**
   - Initialize the policy network parameters, $\theta$.
   - Set the initial policy, $\pi_\theta$.

2. **Collect Data**
   - Collect a batch of trajectories using the current policy, $\pi_\theta$.
   - Each trajectory consists of a sequence of states, actions, and rewards.

3. **Calculate Advantage Function**
   - Estimate the advantage function for each state-action pair in the trajectory:

$$
\hat{A}(s,a) = q(s,a) - v(s)
$$

   - The advantage function measures how much better an action is compared to the average action.

4. **Update Policy**

Update the policy parameters, $\theta$, by maximizing the clipped surrogate objective function:

$$
L^{\mathrm{CLIP}}(\theta) =
\mathbb{E}_t \left[
\min \left(
r_t(\theta)\hat{A}_t,
\mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t
\right)
\right]
$$

where

$$
r_t(\theta) =
\frac{\pi_\theta(a_t \mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t \mid s_t)}
$$

- $r_t(\theta)$ is the ratio of the new policy's action probability to the old policy's action probability.
- $\epsilon$ is the clipping parameter.
- $\hat{A}_t$ is the estimated advantage at time step $t$.

   - $r_t(\theta)$ is the ratio of the new policy's action probability to the old policy's action probability.
   - $\epsilon$ is the clipping parameter.
   - $\hat{A}_t$ is the estimated advantage at time step $t$.
   - The optimization is typically performed using stochastic gradient descent or Adam.

5. **Repeat**
   - Repeat steps 2–4 until convergence or until the desired performance level is reached.

**Advantages**:
* Simple to implement
* Can be used with a variety of environments
* Can learn complex policies

**Disadvantages:**
* Can be slow to converge
* Can be sensitive to the choice of learning rate and discount factor
* Can suffer from high variance in the gradient estimates

##### Actor-Critic Methods
Actor-critic methods combine the best of both worlds: value-based and policy-based approaches. They consist of two components:

## Key Concepts

### Actor-Critic Architecture

* **Actor:** A neural network that learns a policy, $\pi(a \mid s)$, which maps states to actions.

* **Critic:** A neural network that learns a value function, $v(s)$, which estimates the expected return from a given state.

* **Advantage Function:**  
  The advantage function $A(s,a)$ measures how much better taking action $a$ in state $s$ is compared to the average action. It is calculated as:

$$
A(s,a) = q(s,a) - v(s)
$$

Here, $q(s,a)$ is the action-value function, which estimates the expected return from taking action $a$ in state $s$ and following the policy thereafter.

* **Policy Gradient Update:**  
  The actor's parameters are updated using the policy gradient:

$$
\nabla_\theta J(\theta)
\approx
\mathbb{E}
\left[
A(s,a) \nabla_\theta \log \pi(a \mid s)
\right]
$$

* **Value Function Update:**  
  The critic's parameters are updated using a temporal difference error:

$$
\delta = r + \gamma V(s') - V(s)
$$

The critic is then updated in the direction of:

$$
\nabla_w V(s) \approx \delta \nabla_w V(s)
$$

**Algorithm**:

1. Initialize:
    * Initialize the actor and critic neural networks with random parameters.
2. Collect Experience:
    *  Run multiple environments in parallel.
    * For each environment:
    * Sample an action from the actor's policy.
    * Execute the action in the environment and observe the next state and reward.
    * Store the transition $(s, a, r, s')$ in a replay buffer.
3. Update Networks:
    * Sample a batch of transitions from the replay buffer.
    * Calculate the advantage function for each transition.
    * Update the actor's parameters using the policy gradient.
    * Update the critic's parameters using the TD error.
4. Repeat:
    * Repeat steps 2 and 3 until the desired performance is achieved.
    
**Advantages of A2C**:

* Efficient Training: By running multiple environments in parallel, A2C can train faster than single-threaded methods.
* Stable Learning: The advantage function helps stabilize the policy gradient updates.
Reduced Variance: The value function provides a baseline that reduces the variance of the policy gradient.
* Flexibility: A2C can be applied to a wide range of reinforcement learning problems.

**Disadvantages of A2C**:

* Synchronization Overhead: The need to synchronize the actor and critic networks can slow down training.
* Less Sample Efficiency: A2C may require more samples than other methods to converge.


#### stable_baselines3: A Reliable Reinforcement Learning Library

Stable Baselines3 is a powerful reinforcement learning library built on top of TensorFlow and PyTorch. It provides a set of state-of-the-art RL algorithms, making it a popular choice for researchers and practitioners.

Key Features:

* State-of-the-Art Algorithms: It includes a wide range of algorithms, such as:
   * DQN
   * PPO
   * A2C
   * SAC
   * TD3
* Modular Design: The library is modular, allowing for easy customization and experimentation.
* Easy to Use: It provides a simple and intuitive API for training and evaluating RL agents.
* Well-Documented: The library is well-documented, with clear examples and tutorials.

#### Implementation of the stable_baselines3 models
The *MlpPolicy* in Stable-Baselines3 (SB3) is a Multi-Layer Perceptron (MLP) neural network used for reinforcement learning algorithms like PPO, A2C, SAC, TD3, and DQN. It handles environments with vectorized observations and approximates the policy (actor) and value function (critic).

Key Features:
* Default Architecture: Fully connected layers with customizable sizes and activation functions.
* Customization: Use policy_kwargs to adjust the network architecture, e.g., number of layers or units.

#### Training and Evaluation of the three stable_baseline3 Agents

##### Hyperparameter Tuning 
While Stable Baselines 3 offers a diverse set of reinforcement learning algorithms, achieving optimal performance often hinges on careful hyperparameter tuning. As the provided chart illustrates, algorithms like PPO(adjusted parameters), A2C(original) and DQN(original) can be particularly susceptible to this. 

* Sensitivity: All three algorithms exhibit sensitivity to their hyperparameters. Selecting the right values significantly impacts their learning performance.
* PPO's Robustness: Compared to A2C, PPO appears to be less sensitive to hyperparameters.
* A2C's Overfitting: This phenomenon occurs when the algorithm learns to perform well on the specific training data but fails to generalize to unseen scenarios.

##### Addressing Hyperparameter Tuning
+ Experimentation: Experimenting with different hyperparameter values is crucial to identify the best configuration for your specific task and algorithm. Tools like grid search or random search can be employed to efficiently explore the hyperparameter space.
* Documentation and Community: Stable-baselines3 provides documentation and community resources that offer guidance on common hyperparameters and their impact on different algorithms.

![Evaluation StableBaseline3 Algorithms](https://hackmd.io/_uploads/HJWtxw_4kx.png)

#### Importance of the observation space on a 12x12 Grid

To investigate the impact of observation space size, a Proximal Policy Optimization (PPO) algorithm was applied to the snake game with three different observation space configurations:

**Agents Vision**

* PPO12: Limited vision, observing only the four squares directly adjacent to its head.
* PPO32: Moderate vision, encompassing a 5x5 grid centered around the head.
* PPO88: Extensive vision, covering a 9x9 grid centered around the head.

##### Unexpected Results:

Contrary to expectations, the results revealed a diminishing return on increasing the observation space size. While PPO88, with the widest field of view, performed better than PPO12, the improvement wasn't as significant as anticipated.

![Visablity Vergleich](https://hackmd.io/_uploads/BkWSmkj41g.png)

**Distance Food to Snake head**
Building on the previous experiment, we investigated the impact of providing the agent with the specific distance (X and Y coordinates) to the food. We compared two configurations of PPO32:

* PPO32 (No Distance): The agent only knows its current direction, the food's presence and a square of 5x5 units.
* PPO32 (Distance): The agent receives its current direction, food presence, a square of 5x5 units and X and Y distance to the food.
##### Results:
The chart demonstrates a significant performance improvement when providing distance information. This suggests that knowledge of the food's relative position plays a crucial role in effective decision-making for the snake agent.

![Distance Vergleich](https://hackmd.io/_uploads/rJLrm1iNyl.png)
## Conclusion

This report demonstrates the effectiveness of reinforcement learning in training an AI agent to play the Snake game. By carefully designing the observation space and leveraging powerful algorithms like PPO, A2C, and DQN, we achieved significant performance improvements. This research highlights the potential of reinforcement learning to tackle complex problems and offers insights into the importance of information representation and algorithm selection in RL.

## Reference
* Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction (2nd ed.). MIT Press. [link](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)
* Van de Laar, T. (2019). Introduction to Reinforcement Learning. Towards Data Science. [link](https://towardsdatascience.com/introduction-to-reinforcement-learning-temporal-difference-sarsa-q-learning-e8f22669c366)
* Vandelaer, C. (2017). Reinforcement Learning: An Introduction (Part 1). Medium. [link](https://towardsdatascience.com/introduction-to-reinforcement-learning-temporal-difference-sarsa-q-learning-e8f22669c366)
* Doshi, K. (2020). Reinforcement Learning Explained visually (Part 4): Q-learning, step-by-step. [link](https://towardsdatascience.com/reinforcement-learning-explained-visually-part-4-q-learning-step-by-step-b65efb731d3e)
* GeeksforGeeks. (2024). Differences between Q-learning and SARSA. [link](https://www.geeksforgeeks.org/differences-between-q-learning-and-sarsa/)
* CMU. (2024). Markov Decision Process. [link](https://www.cs.cmu.edu/~./15281/coursenotes/mdps/index.html)
