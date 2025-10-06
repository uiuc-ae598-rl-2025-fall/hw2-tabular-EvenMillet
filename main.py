import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from collections import defaultdict


# Environment setup
class FrozenLakeMDP:
    def __init__(self, size=4, discount=0.95, is_slippery=True):
        self.size = size
        self.discount = discount
        self.is_slippery = is_slippery
        self.map = ["SFFF", "FHFH", "FFFH", "HFFG"]

        # state space
        self.states = list(range(size * size))

        # action space
        self.actions = [0, 1, 2, 3]
        self.action_names = ['Left', 'Down', 'Right', 'Up']

        # terminal states
        self.terminal_states = []
        for i in range(size):
            for j in range(size):
                state = i * size + j
                if self.map[i][j] == 'H':  # Hole
                    self.terminal_states.append(state)
                elif self.map[i][j] == 'G':  # Goal
                    self.terminal_states.append(state)

        # rewards
        self.rewards = np.zeros(size * size)
        for state in self.states:
            i, j = state // size, state % size
            if self.map[i][j] == 'G':
                self.rewards[state] = 1.0

        # transition model
        self.transitions = np.zeros((len(self.states), len(self.actions), len(self.states)))
        for state in self.states:
            if state in self.terminal_states:
                # terminal states
                for action in self.actions:
                    self.transitions[state, action, state] = 1.0
                continue

            i, j = state // size, state % size
            for action in self.actions:
                if self.is_slippery:
                    # considering slippery behavior
                    intended_action = action
                    possible_actions = [
                        (intended_action, 1 / 3),  # Intended direction
                        ((intended_action - 1) % 4, 1 / 3),  # One direction to the right
                        ((intended_action + 1) % 4, 1 / 3)  # One direction to the left
                    ]
                else:
                    # non-slippery
                    possible_actions = [(action, 1.0)]

                for a, prob in possible_actions:
                    next_i, next_j = i, j

                    if a == 0:  # Left
                        next_j = max(j - 1, 0)
                    elif a == 1:  # Down
                        next_i = min(i + 1, size - 1)
                    elif a == 2:  # Right
                        next_j = min(j + 1, size - 1)
                    elif a == 3:  # Up
                        next_i = max(i - 1, 0)

                    next_state = next_i * size + next_j
                    self.transitions[state, action, next_state] += prob

    def reset(self):
        return 0

    def step(self, state, action):
        # Sample next state according to transition probabilities
        probs = self.transitions[state, action, :]
        next_state = np.random.choice(self.states, p=probs)
        reward = self.rewards[next_state]
        done = next_state in self.terminal_states
        return next_state, reward, done

    def get_next_state_reward(self, state, action):
        probs = self.transitions[state, action, :]
        next_states = np.where(probs > 0)[0]
        p = probs[next_states]
        rewards = self.rewards[next_states]
        return list(zip(next_states, p, rewards))

# define three tabular algorithms
class FrozenLakeRL:
    def __init__(self, mdp):
        self.mdp = mdp
        self.gamma = mdp.discount
        self.n_states = len(mdp.states)
        self.n_actions = len(mdp.actions)

    def epsilon_greedy_policy(self, Q, state, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(self.mdp.actions)
        else:
            return np.argmax(Q[state])

    def generate_episode(self, Q, epsilon):
        episode = []
        state = self.mdp.reset()
        done = False

        while not done:
            action = self.epsilon_greedy_policy(Q, state, epsilon)
            next_state, reward, done = self.mdp.step(state, action)
            episode.append((state, action, reward))
            state = next_state

        return episode

    def mc_control(self, n_episodes=10000, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        Q = np.random.rand(self.n_states, self.n_actions) * 0.01
        returns = defaultdict(list)
        epsilon = epsilon_start

        episode_returns = []

        for episode_num in range(n_episodes):
            episode = self.generate_episode(Q, epsilon)

            # calculate episode return
            episode_return = sum([x[2] for x in episode])
            episode_returns.append(episode_return)

            # track visited state-action pairs for first-visit
            visited = set()

            G = 0
            for t in range(len(episode) - 1, -1, -1):
                state, action, reward = episode[t]
                G = self.gamma * G + reward

                if (state, action) not in visited:
                    visited.add((state, action))
                    returns[(state, action)].append(G)
                    Q[state, action] = np.mean(returns[(state, action)])

            epsilon = max(epsilon_end, epsilon * epsilon_decay)

        return Q, episode_returns

    def sarsa(self, n_episodes=10000, alpha=0.1, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        Q = np.random.rand(self.n_states, self.n_actions) * 0.01
        epsilon = epsilon_start

        episode_returns = []

        for episode_num in range(n_episodes):
            state = self.mdp.reset()
            action = self.epsilon_greedy_policy(Q, state, epsilon)

            episode_return = 0
            done = False

            while not done:
                next_state, reward, done = self.mdp.step(state, action)
                episode_return += reward

                if not done:
                    next_action = self.epsilon_greedy_policy(Q, next_state, epsilon)
                    td_target = reward + self.gamma * Q[next_state, next_action]
                    td_error = td_target - Q[state, action]
                    Q[state, action] += alpha * td_error

                    state = next_state
                    action = next_action
                else:
                    td_target = reward
                    td_error = td_target - Q[state, action]
                    Q[state, action] += alpha * td_error

            episode_returns.append(episode_return)
            epsilon = max(epsilon_end, epsilon * epsilon_decay)

        return Q, episode_returns

    def q_learning(self, n_episodes=10000, alpha=0.1, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        Q = np.random.rand(self.n_states, self.n_actions) * 0.01
        epsilon = epsilon_start

        episode_returns = []

        for episode_num in range(n_episodes):
            state = self.mdp.reset()
            episode_return = 0
            done = False

            while not done:
                action = self.epsilon_greedy_policy(Q, state, epsilon)

                next_state, reward, done = self.mdp.step(state, action)
                episode_return += reward

                if not done:
                    td_target = reward + self.gamma * np.max(Q[next_state])
                else:
                    td_target = reward

                td_error = td_target - Q[state, action]
                Q[state, action] += alpha * td_error
                state = next_state

            episode_returns.append(episode_return)
            epsilon = max(epsilon_end, epsilon * epsilon_decay)

        return Q, episode_returns


def moving_average(data, window=100):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode='valid')


def plot_learning_curves(returns_mc, returns_sarsa, returns_qlearning, window=100):
    plt.figure(figsize=(12, 6))

    # smooth the curves
    smooth_mc = moving_average(returns_mc, window)
    smooth_sarsa = moving_average(returns_sarsa, window)
    smooth_qlearning = moving_average(returns_qlearning, window)

    plt.plot(smooth_mc, label='Monte Carlo Control', linewidth=2)
    plt.plot(smooth_sarsa, label='SARSA', linewidth=2)
    plt.plot(smooth_qlearning, label='Q-Learning', linewidth=2)

    plt.xlabel('Episode', fontsize=18)
    plt.ylabel('Episode Return (smoothed)', fontsize=18)
    plt.title('Learning Curves: Episode Return vs Episodes', fontsize=18)
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_value_function(Q, mdp, title="State Value Function"):
    V = np.max(Q, axis=1).reshape(mdp.size, mdp.size)

    plt.figure(figsize=(8, 8))
    sns.heatmap(V, annot=True, fmt='.3f', cmap='viridis', cbar=True,
                square=True, xticklabels=False, yticklabels=False,
                annot_kws={'size': 14})
    plt.title(title, fontsize=18)
    plt.tight_layout()
    plt.show()


def plot_policy(Q, mdp, title="Optimal Policy"):
    policy = np.argmax(Q, axis=1)
    arrows = {0: '←', 1: '↓', 2: '→', 3: '↑'}

    plt.figure(figsize=(8, 8))
    for i in range(mdp.size):
        for j in range(mdp.size):
            state = i * mdp.size + j
            cell = mdp.map[i][j]

            if cell == 'H':
                plt.text(j, i, 'H', ha='center', va='center',
                         fontsize=24, color='red', weight='bold')
            elif cell == 'G':
                plt.text(j, i, 'G', ha='center', va='center',
                         fontsize=24, color='green', weight='bold')
            else:
                plt.text(j, i, arrows[policy[state]], ha='center', va='center',
                         fontsize=24, color='blue')

    plt.xlim(-0.5, mdp.size - 0.5)
    plt.ylim(-0.5, mdp.size - 0.5)
    plt.gca().invert_yaxis()
    plt.title(title, fontsize=18)
    plt.grid(True)
    plt.xticks(range(mdp.size))
    plt.yticks(range(mdp.size))
    plt.tight_layout()
    plt.show()


def plot_trajectory(mdp, Q, title="Example Trajectory", max_steps=200):
    np.random.seed(42)
    state = mdp.reset()
    trajectory = [state]

    done = False
    steps = 0

    # follow greedy policy
    while not done and steps < max_steps:
        action = np.argmax(Q[state])
        next_state, reward, done = mdp.step(state, action)
        trajectory.append(next_state)
        state = next_state
        steps += 1

    grid = np.zeros((mdp.size, mdp.size))
    for i in range(mdp.size):
        for j in range(mdp.size):
            state_idx = i * mdp.size + j
            cell = mdp.map[i][j]
            if cell == 'H':
                grid[i, j] = 3  # Hole
            elif cell == 'G':
                grid[i, j] = 2  # Goal
            elif cell == 'S':
                grid[i, j] = 0  # Start
            else:
                grid[i, j] = 0  # Frozen

    # mark trajectory
    for state_idx in trajectory:
        i, j = state_idx // mdp.size, state_idx % mdp.size
        if mdp.map[i][j] not in ['H', 'G']:
            grid[i, j] = 1  # Path

    cmap = ListedColormap(['white', 'blue', 'green', 'red'])

    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap=cmap)

    # create a dictionary to track last visit to each state
    state_last_visit = {}
    for idx, state_idx in enumerate(trajectory):
        state_last_visit[state_idx] = idx

    # Add text annotations - only show the last visit number for each state
    for state_idx, last_idx in state_last_visit.items():
        i, j = state_idx // mdp.size, state_idx % mdp.size
        plt.text(j, i, str(last_idx), ha='center', va='center',
                 fontsize=16, color='white', weight='bold')

    # Add labels for special cells
    for i in range(mdp.size):
        for j in range(mdp.size):
            state_idx = i * mdp.size + j
            if mdp.map[i][j] == 'H':
                plt.text(j, i - 0.3, 'HOLE', ha='center', va='center',
                         fontsize=14, color='white', weight='bold')
            elif mdp.map[i][j] == 'G':
                plt.text(j, i - 0.3, 'GOAL', ha='center', va='center',
                         fontsize=14, color='white', weight='bold')

    final_state = trajectory[-1]
    success = mdp.map[final_state // mdp.size][final_state % mdp.size] == 'G'

    plt.title(f"{title}\nSteps: {len(trajectory) - 1}, Success: {success}",
              fontsize=18)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()

    return trajectory


if __name__ == "__main__":
    # test in both slippery and non-slippery scenarios
    for is_slippery in [True, False]:
        print("------------------------------------------")
        print(f"Running experiments with is_slippery={is_slippery}")
        print("------------------------------------------")

        mdp = FrozenLakeMDP(size=4, discount=0.95, is_slippery=is_slippery)

        rl = FrozenLakeRL(mdp)

        # run algorithms
        Q_mc, returns_mc = rl.mc_control(n_episodes=10000)

        Q_sarsa, returns_sarsa = rl.sarsa(n_episodes=10000, alpha=0.1)

        Q_qlearning, returns_qlearning = rl.q_learning(n_episodes=10000, alpha=0.1)

        # plot learning curves
        plot_learning_curves(returns_mc, returns_sarsa, returns_qlearning)

        # plot value functions
        plot_value_function(Q_mc, mdp, title=f"Value Function - MC Control (slippery={is_slippery})")
        plot_value_function(Q_sarsa, mdp, title=f"Value Function - SARSA (slippery={is_slippery})")
        plot_value_function(Q_qlearning, mdp, title=f"Value Function - Q-Learning (slippery={is_slippery})")

        # plot policies
        plot_policy(Q_mc, mdp, title=f"Optimal Policy - MC Control (slippery={is_slippery})")
        plot_policy(Q_sarsa, mdp, title=f"Optimal Policy - SARSA (slippery={is_slippery})")
        plot_policy(Q_qlearning, mdp, title=f"Optimal Policy - Q-Learning (slippery={is_slippery})")

        # plot example trajectories
        plot_trajectory(mdp, Q_mc, title=f"Example Trajectory - MC Control (slippery={is_slippery})")
        plot_trajectory(mdp, Q_sarsa, title=f"Example Trajectory - SARSA (slippery={is_slippery})")
        plot_trajectory(mdp, Q_qlearning, title=f"Example Trajectory - Q-Learning (slippery={is_slippery})")

        # print final results
        print(f"\nFinal average returns (slippery={is_slippery}):")
        print(f"MC Control: {np.mean(returns_mc[-100:]):.4f}")
        print(f"SARSA: {np.mean(returns_sarsa[-100:]):.4f}")
        print(f"Q-Learning: {np.mean(returns_qlearning[-100:]):.4f}")