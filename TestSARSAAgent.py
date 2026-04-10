from ShortCutEnvironment import ShortcutEnvironment
from ShortCutAgents import SARSAAgent

env = ShortcutEnvironment()
agent = SARSAAgent(n_actions=env.action_size(), n_states=env.state_size(), epsilon=0.1, alpha=0.1)
returns = agent.train(env, n_episodes=100)

print('First reward:', returns[0])
print('Last reward:', returns[-1])
print('SARSA works!')