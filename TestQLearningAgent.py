# Write your experiments in here! You can use the plotting helper functions from the previous assignment if you want.
from ShortCutEnvironment import ShortcutEnvironment
from ShortCutAgents import QLearningAgent

env = ShortcutEnvironment()
agent = QLearningAgent(n_actions=4, n_states=144, epsilon=0.1, alpha=0.1)
returns = agent.train(env, n_episodes=100)

print('First reward:', returns[0])   # should be very negative (robot is lost)
print('Last reward:', returns[-1])   # should be less negative (robot learned!)
print('QLearning works!')
