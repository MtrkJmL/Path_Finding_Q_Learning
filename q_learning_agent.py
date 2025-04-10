import numpy as np
import random

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.99, 
                 exploration_rate=1.0, exploration_decay=0.995, min_exploration_rate=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros((state_size, action_size))
    
    def choose_action(self, state):
        # Exploration vs Exploitation
        if random.random() < self.exploration_rate:
            return random.randint(0, self.action_size - 1)
        else:
            return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state, done):
        # Q-learning update rule
        current_q = self.q_table[state, action]
        if done:
            next_q = reward
        else:
            next_q = reward + self.discount_factor * np.max(self.q_table[next_state])
        
        # Update Q-value
        self.q_table[state, action] = current_q + self.learning_rate * (next_q - current_q)
        
        # Decay exploration rate
        if done:
            self.exploration_rate = max(self.min_exploration_rate, 
                                      self.exploration_rate * self.exploration_decay)
    
    def save_q_table(self, filename):
        np.save(filename, self.q_table)
    
    def load_q_table(self, filename):
        self.q_table = np.load(filename) 