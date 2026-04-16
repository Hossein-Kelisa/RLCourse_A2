import numpy as np
from ShortCutEnvironment import ShortcutEnvironment

class QLearningAgent(object):

    def __init__(self, n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=1.0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # TO DO: Initialize variables if necessary
        self.Q = np.zeros((n_states, n_actions))
        
    def select_action(self, state):
        # TO DO: Implement policy
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            action = np.argmax(self.Q[state])
        return action
        
    def update(self, state, action, reward, next_state, done): # Augment arguments if necessary
        # TO DO: Implement Q-learning update
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.Q[next_state])
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
    
    def train(self, env, n_episodes):
        # TO DO: Implement the agent loop that trains for n_episodes. 
        # Return a vector with the the cumulative reward (=return) per episode
        episode_returns = []
        for episode in range(n_episodes):
            env.reset()
            state = env.state()
            cumulative_reward = 0
            while not env.done():
                action = self.select_action(state)
                reward = env.step(action)
                next_state = env.state()
                self.update(state, action, reward, next_state, env.done())
                state = next_state
                cumulative_reward += reward
            episode_returns.append(cumulative_reward)
        return episode_returns


class SARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=1.0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # TO DO: Initialize variables if necessary
        self.Q = np.zeros((n_states, n_actions))
        
    def select_action(self, state):
        # TO DO: Implement policy
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            action = np.argmax(self.Q[state])
        return action
        
    def update(self, state, action, reward, next_state, next_action, done): # Augment arguments if necessary
        # TO DO: Implement SARSA update
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * self.Q[next_state, next_action]
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error

    def train(self, env, n_episodes):
        # TO DO: Implement the agent loop that trains for n_episodes. 
        # Return a vector with the the cumulative reward (=return) per episode
        episode_returns = []
        for episode in range(n_episodes):
            env.reset()
            state = env.state()
            action = self.select_action(state)
            cumulative_reward = 0
            while not env.done():
                reward = env.step(action)
                next_state = env.state()
                if not env.done():
                    next_action = self.select_action(next_state)
                else:
                    next_action = None
                self.update(state, action, reward, next_state, next_action, env.done())
                state = next_state
                action = next_action
                cumulative_reward += reward
            episode_returns.append(cumulative_reward)
        return episode_returns


class ExpectedSARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=1.0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # TO DO: Initialize variables if necessary
        self.Q = np.zeros((n_states, n_actions))
        
    def select_action(self, state):
        # TO DO: Implement policy
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            action = np.argmax(self.Q[state])
        return action
    
    def expected_value(self, state):
        # Expected value under epsilon-greedy policy
        best_action = np.argmax(self.Q[state])
        probs = np.ones(self.n_actions) * (self.epsilon / self.n_actions)
        probs[best_action] += (1 - self.epsilon)
        return np.dot(probs, self.Q[state])
        
    def update(self, state, action, reward, next_state, done): # Augment arguments if necessary
        # TO DO: Implement Expected SARSA update
        # Expected SARSA update (Eq. 6.9):
        # Q(s,a) <- Q(s,a) + alpha * [reward + gamma * E[Q(s',a')] - Q(s,a)]
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * self.expected_value(next_state)
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error

    def train(self, env, n_episodes):
        # TO DO: Implement the agent loop that trains for n_episodes. 
        # Return a vector with the the cumulative reward (=return) per episode
        episode_returns = []
        for episode in range(n_episodes):
            env.reset()
            state = env.state()
            cumulative_reward = 0
            while not env.done():
                action = self.select_action(state)
                reward = env.step(action)
                next_state = env.state()
                self.update(state, action, reward, next_state, env.done())
                state = next_state
                cumulative_reward += reward
            episode_returns.append(cumulative_reward)
        return episode_returns    


class nStepSARSAAgent(object):

    def __init__(self, n_actions, n_states, n, epsilon=0.1, alpha=0.1, gamma=1.0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.n = n
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # TO DO: Initialize variables if necessary
        self.Q = np.zeros((n_states, n_actions))
        
    def select_action(self, state):
        # TO DO: Implement policy
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            action = np.argmax(self.Q[state])
        return action
        
    def update(self, states, actions, rewards, done): # Augment arguments if necessary
        # TO DO: Implement n-step SARSA update
        n = len(rewards)
        G = sum((self.gamma ** i) * r for i, r in enumerate(rewards))

        if not done:
            G += (self.gamma ** n) * self.Q[states[-1], actions[-1]]
        td_error = G - self.Q[states[0], actions[0]]
        self.Q[states[0], actions[0]] += self.alpha * td_error
    
    def train(self, env, n_episodes):
        # TO DO: Implement the agent loop that trains for n_episodes. 
        # Return a vector with the the cumulative reward (=return) per episode
        episode_returns = []
        for episode in range(n_episodes):
            env.reset()
            state = env.state()
            action = self.select_action(state)
            cumulative_reward = 0
            states = [state]
            actions = [action]
            rewards = []
            while not env.done() or len(states) > 1:
                if not env.done():
                    reward = env.step(action)
                    cumulative_reward += reward
                    next_state = env.state()
                    next_action = self.select_action(next_state)
                    states.append(next_state)
                    actions.append(next_action)
                    rewards.append(reward)
                    action = next_action
                if len(rewards) == self.n or env.done():
                    self.update(states, actions, rewards, env.done())
                    states.pop(0)
                    actions.pop(0)
                    rewards.pop(0)
            episode_returns.append(cumulative_reward)
        return episode_returns  
    
    
    