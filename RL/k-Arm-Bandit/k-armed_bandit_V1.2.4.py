import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class EpsilonGreedyBandit:
    """
    Epsilon-Greedy 多臂赌博机算法实现。
    """
    def __init__(self, n_arms, epsilon, n_pulls=2000):
        """
        初始化多臂赌博机。
        :param n_arms: 臂的数量
        :param epsilon: 探索概率
        :param n_pulls: 总拉动次数
        """
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.n_pulls = n_pulls
        self.Q = np.zeros(n_arms)  # 每个臂的估计值
        self.N = np.zeros(n_arms)  # 每个臂被拉动的次数
        self.optimal_arm = None  # 最佳臂的索引
        self.optimal_action_counts = np.zeros(n_pulls)  # 记录每一步是否选择了最佳臂

    def set_optimal_arm(self, true_means):
        """
        设置最佳臂的索引。
        :param true_means: 每个臂的真实奖励均值
        """
        self.optimal_arm = np.argmax(true_means)

    def pull_arm(self):
        """
        根据 epsilon-greedy 策略选择一个臂。
        :return: 选择的臂的索引
        """
        return np.random.randint(self.n_arms) if np.random.rand() < self.epsilon else np.argmax(self.Q)

    def update(self, arm, reward, step):
        """
        更新臂的估计值和统计信息。
        :param arm: 被拉动的臂的索引
        :param reward: 获得的奖励
        :param step: 当前步骤
        """
        self.N[arm] += 1
        self.Q[arm] += (reward - self.Q[arm]) / self.N[arm]
        self.optimal_action_counts[step] = int(arm == self.optimal_arm)

    def run(self, true_means):
        """
        运行赌博机实验。
        :param true_means: 每个臂的真实奖励均值
        :return: 每一步的奖励和最优臂选择次数
        """
        self.set_optimal_arm(true_means)
        rewards = np.zeros(self.n_pulls)
        for step in range(self.n_pulls):
            arm = self.pull_arm()
            reward = np.random.normal(true_means[arm], 1)
            self.update(arm, reward, step)
            rewards[step] = reward
        return rewards, self.optimal_action_counts


def run_experiments(n_arms_list, epsilon_list, n_pulls, n_experiments):
    """
    运行多组实验并收集结果。
    :param n_arms_list: 臂的数量列表
    :param epsilon_list: 探索概率列表
    :param n_pulls: 每次实验的拉动次数
    :param n_experiments: 每组参数的实验次数
    :return: 包含所有实验结果的 DataFrame
    """
    results = {
        "epsilon": [],
        "n_arms": [],
        "sequence": [],
        "average_reward": [],
        "average_optimal_actions": [],
    }

    for epsilon in epsilon_list:
        for n_arms in n_arms_list:
            cumulative_rewards = np.zeros(n_pulls)
            cumulative_optimal_actions = np.zeros(n_pulls)

            for _ in range(n_experiments):
                true_means = np.random.normal(0, 1, n_arms)
                bandit = EpsilonGreedyBandit(n_arms, epsilon, n_pulls)
                rewards, optimal_actions = bandit.run(true_means)
                cumulative_rewards += rewards
                cumulative_optimal_actions += optimal_actions

            avg_rewards = cumulative_rewards / n_experiments
            avg_optimal_actions = cumulative_optimal_actions / n_experiments

            for step in range(n_pulls):
                results["epsilon"].append(epsilon)
                results["n_arms"].append(n_arms)
                results["sequence"].append(step + 1)
                results["average_reward"].append(avg_rewards[step])
                results["average_optimal_actions"].append(avg_optimal_actions[step])

    return pd.DataFrame(results)


def plot_results(df, n_arms_list, epsilon_list):
    """
    绘制实验结果。
    :param df: 包含实验结果的 DataFrame
    :param n_arms_list: 臂的数量列表
    :param epsilon_list: 探索概率列表
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))

    for idx, n_arms in enumerate(n_arms_list):
        filtered_data = df[df["n_arms"] == n_arms]
        # 绘制平均回报折线图
        ax_reward = axes[0, idx]
        for epsilon in epsilon_list:
            data = filtered_data[filtered_data["epsilon"] == epsilon]
            ax_reward.plot(data["sequence"], data["average_reward"], label=f"ε={epsilon}")
        ax_reward.set_title(f"Average Reward (n_arms={n_arms})")
        ax_reward.set_xlabel("Steps")
        ax_reward.set_ylabel("Average Reward")
        ax_reward.legend()

        # 绘制最优动作比例折线图
        ax_optimal = axes[1, idx]
        for epsilon in epsilon_list:
            data = filtered_data[filtered_data["epsilon"] == epsilon]
            ax_optimal.plot(data["sequence"], data["average_optimal_actions"] * 100, label=f"ε={epsilon}")
        ax_optimal.set_title(f"Optimal Actions % (n_arms={n_arms})")
        ax_optimal.set_xlabel("Steps")
        ax_optimal.set_ylabel("Optimal Actions (%)")
        ax_optimal.legend()

    plt.tight_layout()
    plt.show()


# 参数设置
n_arms_list = [5, 10, 20]
epsilon_list = [0, 0.01, 0.1]
n_pulls = 2000
n_experiments = 1000

# 运行实验并保存结果
df = run_experiments(n_arms_list, epsilon_list, n_pulls, n_experiments)
df.to_csv("optimized_experiment_results.csv", index=False)

print("Experiment completed! Results saved to optimized_experiment_results.csv")

# 绘制结果
plot_results(df, n_arms_list, epsilon_list)