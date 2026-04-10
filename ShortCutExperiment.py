import numpy as np
import matplotlib.pyplot as plt
from ShortCutEnvironment import ShortcutEnvironment, WindyShortcutEnvironment
from ShortCutAgents import QLearningAgent, SARSAAgent, ExpectedSARSAAgent, nStepSARSAAgent


def smooth(rewards, window=10):
    return np.convolve(rewards, np.ones(window) / window, mode='valid')


def run_repetitions(agent_type, n_rep, n_episodes, alpha=0.1, epsilon=0.1, n=1):
    all_returns = []  # will hold one list per repetition

    for rep in range(n_rep):
        # Create a FRESH environment and agent each repetition!
        env = ShortcutEnvironment()

        if agent_type == 'qlearning':
            agent = QLearningAgent(
                n_actions=env.action_size(),
                n_states=env.state_size(),
                epsilon=epsilon, alpha=alpha)

        elif agent_type == 'sarsa':
            agent = SARSAAgent(
                n_actions=env.action_size(),
                n_states=env.state_size(),
                epsilon=epsilon, alpha=alpha)

        elif agent_type == 'expectedsarsa':
            agent = ExpectedSARSAAgent(
                n_actions=env.action_size(),
                n_states=env.state_size(),
                epsilon=epsilon, alpha=alpha)

        elif agent_type == 'nstepsarsa':
            agent = nStepSARSAAgent(
                n_actions=env.action_size(),
                n_states=env.state_size(),
                n=n, epsilon=epsilon, alpha=alpha)

        # Train and collect returns
        returns = agent.train(env, n_episodes)
        all_returns.append(returns)

    # Average over all repetitions
    avg_returns = np.mean(all_returns, axis=0)
    return avg_returns


# # (quick test of run_repetitions):
# result = run_repetitions('qlearning', n_rep=5, n_episodes=100, alpha=0.1)
# print('Shape:', result.shape)   # should print: Shape: (100,)
# print('Last reward:', result[-1])
# print('run_repetitions works!')


# # QUESTION 1b — Q-Learning: single run + 100 repetitions:

print("Question 1b: Running single Q-Learning experiment (10000 episodes)...")
env = ShortcutEnvironment()
agent = QLearningAgent(n_actions=env.action_size(), n_states=env.state_size(),
                       epsilon=0.1, alpha=0.1)
agent.train(env, 10000)
print("Greedy policy after 10000 episodes:")
env.render_greedy(agent.Q)

print("\nRunning 100 repetitions x 1000 episodes...")
avg_returns_q = run_repetitions('qlearning', n_rep=100, n_episodes=1000,
                                 alpha=0.1, epsilon=0.1)

plt.figure(figsize=(9, 5))
plt.plot(smooth(avg_returns_q), label='Q-Learning (alpha=0.1)')
plt.xlabel('Episode')
plt.ylabel('Average cumulative reward')
plt.title('Q-Learning learning curve (100 repetitions)')
plt.legend()
plt.tight_layout()
plt.savefig('q1b_qlearning_curve.png')
plt.show()
print("Saved: q1b_qlearning_curve.png")



# # QUESTION 1c — Q-Learning: compare different alpha values


print("\nQuestion 1c: Testing different alpha values...")
alphas = [0.01, 0.1, 0.5, 0.9]

plt.figure(figsize=(9, 5))
for alpha in alphas:
    avg_returns = run_repetitions('qlearning', n_rep=100, n_episodes=1000,
                                   alpha=alpha, epsilon=0.1)
    plt.plot(smooth(avg_returns), label=f'alpha={alpha}')
    print(f"  alpha={alpha} done.")

plt.xlabel('Episode')
plt.ylabel('Average cumulative reward')
plt.title('Q-Learning: effect of learning rate alpha (100 repetitions)')
plt.legend()
plt.tight_layout()
plt.savefig('q1c_alpha_comparison.png')
plt.show()
print("Saved: q1c_alpha_comparison.png")



# QUESTION 2b:


#  Part A: single run, show greedy path 

print("\nQ2b: Running single SARSA experiment (10000 episodes)...")
env = ShortcutEnvironment()
agent = SARSAAgent(n_actions=env.action_size(), n_states=env.state_size(),
                   epsilon=0.1, alpha=0.1)
agent.train(env, n_episodes=10000)
print("Greedy policy (SARSA):")
env.render_greedy(agent.Q)


#  Part B: 100 repetitions, plot learning curve 

print("\nRunning 100 repetitions x 1000 episodes...")
avg_returns_sarsa = run_repetitions('sarsa', n_rep=100, n_episodes=1000,
                                    alpha=0.1, epsilon=0.1)
avg_returns_q = run_repetitions('qlearning', n_rep=100, n_episodes=1000,
                                alpha=0.1, epsilon=0.1)

plt.figure(figsize=(9, 5))
plt.plot(smooth(avg_returns_sarsa), label='SARSA (alpha=0.1)')
plt.plot(smooth(avg_returns_q), label='Q-Learning (alpha=0.1)')
plt.xlabel('Episode')
plt.ylabel('Average cumulative reward')
plt.title('SARSA vs Q-Learning (100 repetitions)')
plt.legend()
plt.tight_layout()
plt.savefig('q2b_sarsa_curve.png')
plt.show()
print("Saved: q2b_sarsa_curve.png")



# QUESTION 2c:

print("\nQ2c: Testing different alpha values for SARSA...")
alphas = [0.01, 0.1, 0.5, 0.9]

plt.figure(figsize=(9, 5))
for alpha in alphas:
    avg_returns = run_repetitions('sarsa', n_rep=100, n_episodes=1000,
                                   alpha=alpha, epsilon=0.1)
    plt.plot(smooth(avg_returns), label=f'alpha={alpha}')
    print(f'  alpha={alpha} done.')

plt.xlabel('Episode')
plt.ylabel('Average cumulative reward')
plt.title('SARSA: effect of alpha (100 repetitions)')
plt.legend()
plt.tight_layout()
plt.savefig('q2c_sarsa_alpha.png')
plt.show()
print("Saved: q2c_sarsa_alpha.png")




# Question 3:

print("\nQ3: Windy environment experiments...")


# Q-learniing in windy environment

print("Running Q-Learning in windy environment (10000 episodes)...")
env_windy = WindyShortcutEnvironment()
agent_q = QLearningAgent(n_actions=env_windy.action_size(),
                         n_states=env_windy.state_size(),
                         epsilon=0.1, alpha=0.1)
agent_q.train(env_windy, n_episodes=10000)
print("Greedy policy - Q-Learning (windy):")
env_windy.render_greedy(agent_q.Q)


# SARSA in windy environment

print("Running SARSA in windy environment (10000 episodes)...")
env_windy2 = WindyShortcutEnvironment()
agent_s = SARSAAgent(n_actions=env_windy2.action_size(),
                     n_states=env_windy2.state_size(),
                     epsilon=0.1, alpha=0.1)
agent_s.train(env_windy2, n_episodes=10000)
print("Greedy policy - SARSA (windy):")
env_windy2.render_greedy(agent_s.Q)



# Question 4b:

# partA: single run, show greedy path

print("\nQ4b: Running single Expected SARSA experiment (10000 episodes)...")
env = ShortcutEnvironment()
agent = ExpectedSARSAAgent(n_actions=env.action_size(), n_states=env.state_size(),
                            epsilon=0.1, alpha=0.1)
agent.train(env, n_episodes=10000)
print("Greedy policy (Expected SARSA):")
env.render_greedy(agent.Q)


# partB: 100 repetitions, compare all 3 agents

print("\nRunning 100 repetitions x 1000 episodes...")
avg_returns_esarsa = run_repetitions('expectedsarsa', n_rep=100, n_episodes=1000,
                                     alpha=0.1, epsilon=0.1)
avg_returns_sarsa = run_repetitions('sarsa', n_rep=100, n_episodes=1000,
                                    alpha=0.1, epsilon=0.1)
avg_returns_q = run_repetitions('qlearning', n_rep=100, n_episodes=1000,
                                alpha=0.1, epsilon=0.1)

plt.figure(figsize=(9, 5))
plt.plot(smooth(avg_returns_esarsa), label='Expected SARSA')
plt.plot(smooth(avg_returns_sarsa), label='SARSA')
plt.plot(smooth(avg_returns_q), label='Q-Learning')
plt.xlabel('Episode')
plt.ylabel('Average cumulative reward')
plt.title('Expected SARSA vs SARSA vs Q-Learning (100 repetitions)')
plt.legend()
plt.tight_layout()
plt.savefig('q4b_expectedsarsa_curve.png')
plt.show()
print("Saved: q4b_expectedsarsa_curve.png")




# Question 4c:

print("\nQ4c: Testing different alpha values for Expected SARSA...")
alphas = [0.01, 0.1, 0.5, 0.9]

plt.figure(figsize=(9, 5))
for alpha in alphas:
    avg_returns = run_repetitions('expectedsarsa', n_rep=100, n_episodes=1000,
                                   alpha=alpha, epsilon=0.1)
    plt.plot(smooth(avg_returns), label=f'alpha={alpha}')
    print(f'  alpha={alpha} done.')

plt.xlabel('Episode')
plt.ylabel('Average cumulative reward')
plt.title('Expected SARSA: effect of alpha (100 repetitions)')
plt.legend()
plt.tight_layout()
plt.savefig('q4c_expectedsarsa_alpha.png')
plt.show()
print("Saved: q4c_expectedsarsa_alpha.png")




# Question 5b:

print("\nQ5b: Running single n-step SARSA experiment (10000 episodes)...")
env = ShortcutEnvironment()
agent = nStepSARSAAgent(n_actions=env.action_size(), n_states=env.state_size(),
                        n=5, epsilon=0.1, alpha=0.1)
agent.train(env, n_episodes=10000)
print("Greedy policy (n-step SARSA, n=5):")
env.render_greedy(agent.Q)

# Question 5c:

print("\nQ5c: Testing different values of n...")
n_values = [1, 2, 5, 10, 25]

plt.figure(figsize=(9, 5))
for n in n_values:
    avg_returns = run_repetitions('nstepsarsa', n_rep=100, n_episodes=1000,
                                   alpha=0.1, epsilon=0.1, n=n)
    plt.plot(smooth(avg_returns), label=f'n={n}')
    print(f'  n={n} done.')

plt.xlabel('Episode')
plt.ylabel('Average cumulative reward')
plt.title('n-step SARSA: effect of n (100 repetitions)')
plt.legend()
plt.tight_layout()
plt.savefig('q5c_nstep_comparison.png')
plt.show()
print("Saved: q5c_nstep_comparison.png")



# Question 6 - final comparison:

print("\nQ6: Final comparison of all algorithms...")

avg_q = run_repetitions('qlearning', n_rep=100, n_episodes=1000,
                         alpha=0.9, epsilon=0.1)
print("Q-Learning done.")

avg_sarsa = run_repetitions('sarsa', n_rep=100, n_episodes=1000,
                             alpha=0.9, epsilon=0.1)
print("SARSA done.")

avg_esarsa = run_repetitions('expectedsarsa', n_rep=100, n_episodes=1000,
                              alpha=0.9, epsilon=0.1)
print("Expected SARSA done.")

avg_nstep = run_repetitions('nstepsarsa', n_rep=100, n_episodes=1000,
                             alpha=0.1, epsilon=0.1, n=25)
print("n-step SARSA done.")

plt.figure(figsize=(9, 5))
plt.plot(smooth(avg_q),      label='Q-Learning (alpha=0.9)')
plt.plot(smooth(avg_sarsa),  label='SARSA (alpha=0.9)')
plt.plot(smooth(avg_esarsa), label='Expected SARSA (alpha=0.9)')
plt.plot(smooth(avg_nstep),  label='n-step SARSA (n=25, alpha=0.1)')
plt.xlabel('Episode')
plt.ylabel('Average cumulative reward')
plt.title('Final comparison: all algorithms (100 repetitions)')
plt.legend()
plt.tight_layout()
plt.savefig('q6_final_comparison.png')
plt.show()
print("Saved: q6_final_comparison.png")
