import numpy as np
import matplotlib.pyplot as plt
# 咏淇的代码，图形很完美
class E_Greedy_Bandit:
    def __init__(self, n_arm, epsilon, n_pulls=2000):
        self.n_arm = n_arm  # Number of arms
        self.epsilon = epsilon  # Exploration rate
        self.n_pulls = n_pulls  # Total number of pulls
        self.true_mean = np.random.normal(0, 1, n_arm)  # Generate the actual mean rewards for each arm
        self.Q = np.zeros(n_arm)  # Array of estimated average rewards for each arm, initialised to zero
        self.N = np.zeros(n_arm)  # Array tracking the number of times each arm has been pulled, initialised to zero
        self.rewards = np.zeros(n_pulls)  # Store rewards over time
        self.optimal_action_counts = np.zeros(n_pulls)  # Track optimal action selection
        self.optimal_action = np.argmax(self.true_mean) # # Identify the optimal action initially

    def pull_arm(self): # Select an arm based on the epsilon-greedy policy
        if np.random.rand() < self.epsilon: # Exploration: randomly select an arm based on the exploration rate
            arm = np.random.choice(self.n_arm)  # Exploration
        else:
            arm = np.argmax(self.Q)  # Exploitation: select the arm with the highest estimated reward
        return arm

    def update(self, arm, reward, pull):    # Update the estimated reward for the selected arm
        self.N[arm] += 1    # Increment the pull count for the selected arm
        self.Q[arm] += (reward - self.Q[arm]) / self.N[arm] # Update the estimated value of the selected arm
        self.rewards[pull] = reward
        self.optimal_action_counts[pull] = 1 if arm == self.optimal_action else 0

    def run(self):  # Run the bandit simulation for the specified number of pulls
        for pull in range(self.n_pulls):
            arm = self.pull_arm()
            reward = np.random.normal(self.true_mean[arm], 1)  # Reward follows a normal distribution
            self.update(arm, reward, pull)
        return self.rewards, self.optimal_action_counts
# Parameter settings
arm = 10  # Number of arms
epsilon_list = [0, 0.01, 0.1]  # Exploration probabilities
n_pulls = 2000  # Total number of pulls
n_experiments = 1000    # Number of repeated experiments

# 创建一个数据字典，键是探索比例，值是摇臂次数2000次，用于保存每一次摇臂的平均回报
average_rewards = {eps: np.zeros(n_pulls) for eps in epsilon_list}
# 创建一个数据字典，键是探索比例，值是摇臂次数2000次，用于保存选择了最优臂的比例
average_optimal_actions = {eps: np.zeros(n_pulls) for eps in epsilon_list}

# Running simulations
for epsilon in epsilon_list:
    # Repeat experiment 1000 times
    for exp in range(n_experiments):
        bandit = E_Greedy_Bandit(arm, epsilon, n_pulls)   # Initialise the bandit
        rewards, optimal_actions = bandit.run()
        average_rewards[epsilon] += rewards
        average_optimal_actions[epsilon] += optimal_actions

    average_rewards[epsilon] /= n_experiments
    average_optimal_actions[epsilon] = (average_optimal_actions[epsilon] / n_experiments) * 100  # Convert to %

# Plot results
plt.figure(figsize=(12, 5))
for epsilon in epsilon_list:
    plt.plot(average_rewards[epsilon], label=f"Epsilon = {epsilon}")
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.title("Average Reward over Time")
plt.legend()
plt.show()

plt.figure(figsize=(12, 5))
for epsilon in epsilon_list:
    plt.plot(average_optimal_actions[epsilon], label=f"Epsilon = {epsilon}")
plt.xlabel("Steps")
plt.ylabel("% Optimal Action")
plt.title("Optimal Action Selection over Time")
plt.legend()
plt.show()