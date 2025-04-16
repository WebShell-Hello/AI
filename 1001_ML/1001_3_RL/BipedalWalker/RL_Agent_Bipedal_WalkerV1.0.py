
# Implement a deep reinforcement learning agent for a game or environment of OpenAI Gym or Gymnasium.
# Use the lunar_lander environment: https://gymnasium.farama.org/environments/box2d/lunar_lander/.
# Please plot the learning progress of your method from 0 to 1000 episodes.
# You can have a figure to show rewards and another figure to show training loss.
# Please use a video or gifs or figures to demonstrate how your agent works.

# 1，optuna训练200次, 找最优超参数（如何学习复习的规则），即学习和复习的节奏
# 2，PPO模型训练1000个episode，找最优神经网络权重（如何执行的策略），即操作顺序

# 1个episode是基于参数的1轮实验，1个模型可能会需要很多的step来完成，但是每次环境不一样可能会导致不同的独立实验需要的步数不一样
# 4，奖励值是以一次episode为周期汇总的，因此奖励曲线的x轴的长度等于episode的数值
# 5，损失值是以步数为周期汇总的
# 奖励曲线的x轴的长度=episode的次数
# 损失曲线的x轴的长度=1000轮实验的实际总步数/study.best_params['n_steps']

import optuna
import imageio
import datetime
import matplotlib.pyplot as plt
from typing import Optional
import gymnasium
from gymnasium.wrappers import RecordVideo
from functools import partial
from stable_baselines3 import PPO
from stable_baselines3.common.logger import Logger, KVWriter, CSVOutputFormat
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

class SilentLogger(Logger):
    def __init__(self, log_file: Optional[str] = None):
        output_formats = []
        if log_file:
            output_formats.append(CSVOutputFormat(log_file))
        super().__init__(folder=None, output_formats=output_formats)
    def record(self, key: str, value, exclude: Optional[KVWriter] = None) -> None:
        super().record(key, value, exclude)
    def dump(self, step: int = 0) -> None:
        super().dump(step)

# 合并回调类，同时记录奖励、损失并控制 episode 数量
class TrainingCallback(BaseCallback):
    def __init__(self, total_episodes=1000, verbose=0):
        super().__init__(verbose)
        self.total_episodes = total_episodes
        self.episode_count = 0
        self.episode_rewards = []  # 手动记录每个 episode 奖励
        self.current_episode_reward = 0
        self.loss_history = [] # 从日志记录损失
        self.policy_loss,self.value_loss,self.entropy = [],[],[] # 记录损失的成分
    #记录奖励
    def _on_step(self) -> bool:
        self.current_episode_reward += sum(self.locals['rewards'])
        dones = self.locals.get('dones', [False] * self.model.env.num_envs)
        for done in dones:
            if done:
                self.episode_count += 1
                self.episode_rewards.append(self.current_episode_reward)
                self.current_episode_reward = 0
                if self.verbose > 0:
                    print(f"Episode: {self.episode_count}/{self.total_episodes}", end="\r")
                if self.episode_count >= self.total_episodes:
                    return False
        return True
    #记录损失
    def _on_rollout_end(self) -> None:
        if hasattr(self.model, 'logger'):
            loss = self.model.logger.name_to_value.get('train/loss')
            if loss is not None: self.loss_history.append(loss)
            # policy_loss = self.model.logger.name_to_value.get('train/policy_gradient_loss')
            # if policy_loss is not None: self.policy_loss.append(policy_loss)
            # value_loss = self.model.logger.name_to_value.get('train/value_loss')
            # if value_loss is not None: self.value_loss.append(value_loss)
            # entropy = self.model.logger.name_to_value.get('train/entropy')
            # if entropy is not None: self.entropy.append(loss)

# 自动调整学习参数，（条件限制：下限，上限，变化幅度）
def OptunaTrial(trial,model_type,parallel):
    hyperparams = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "n_steps": trial.suggest_int("n_steps", 512, 2048, step=256),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "gamma": trial.suggest_float("gamma", 0.95, 0.999, step=0.001),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.99, step=0.01),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.3, step=0.05),
        "ent_coef": trial.suggest_float("ent_coef", 0.01, 0.1, log=True),
        "n_epochs": trial.suggest_int("n_epochs", 5, 15)
    }
    # Create the environment，初次创建的环境只是一个完整的游戏场景，飞行器没驱动，没动作
    env = DummyVecEnv([lambda: Monitor(gymnasium.make(model_type)) for _ in range(parallel)])
    model = PPO("MlpPolicy", env, verbose=0, **hyperparams)
    try:
        model.learn(total_timesteps=100_000)
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
        trial.report(mean_reward, step=0)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    finally:
        env.close()
    return mean_reward

# 基于调整好的学习参数训练模型的动作参数
def TrainPpoAgent(best_params, model_type,model_name,total_episodes=1000,total_timesteps=10_000_000,parallel = 8):
    # 创建单环境，启用 human 渲染模式
    # env = Monitor(gym.make(model_type, render_mode="human"))
    env = DummyVecEnv([lambda: Monitor(gymnasium.make(model_type)) for _ in range(parallel)])
    model = PPO("MlpPolicy", env, verbose=1, **best_params)  # verbose=1 启用日志

    # 配置静默日志记录器，不输出到控制台，保存在本地
    logger = SilentLogger(log_file=f"{model_name}_training_log.csv")
    model.set_logger(logger)
    # 记录1000轮训练的参数
    callback = TrainingCallback(total_episodes, verbose=1)
    print(f"tarting training until {total_episodes} episodes...")
    # 训练，限制不超过100万步长，让它学会控制飞行器平稳降落，并将训练数据更新到callback中
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(model_name)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10) # 测试模型的表现，让它玩 10 次游戏，计算平均奖励和标准差。
    print(f"Mean Reward = {mean_reward}, Std Reward = {std_reward}")
    env.close()
    return model, callback

# 录制GIF动图
def recordGif(model_type,model, filename, fps=30, duration=10, random_seed=42):
    with Monitor(gymnasium.make(model_type, render_mode="rgb_array")) as env:
        frames = []
        obs, _ = env.reset(seed = random_seed)
        for _ in range(fps * duration):
            frame = env.render()
            if frame is None:
                raise ValueError("Environment render() returned None. Check render_mode.")
            frames.append(frame)
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env.step(action)
            if done:
                obs, _ = env.reset()
        imageio.mimsave(filename, frames, fps=fps)
        print(f"GIF saved to {filename}")

# 保存视频
def recordVideo(model_type,model, videoName, random_seed=42):
    env = RecordVideo(
        Monitor(gymnasium.make(model_type, render_mode="rgb_array")),
        video_folder=".",
        name_prefix=videoName,
        episode_trigger=lambda x: True
    )
    with env:
        obs, _ = env.reset(seed = random_seed)
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, _ = env.step(action)
            if done or truncated: # 任务自然结束（着陆或坠毁）或时间用尽则停止
                break
    print(f"Video saved to {videoName}")

# 绘制训练曲线
def plotTrainingCurves(callback, model_name):
    plt.figure(figsize=(10, 16))
    # 奖励曲线
    plt.subplot(2, 1, 1)
    plt.plot(callback.episode_rewards)
    plt.title('Episode Rewards (0 to 1000 Episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Average Episode Reward')
    plt.legend()
    plt.grid(True)
    # 损失曲线,假设每个 episode 平均 300 步（LunarLander-v3 的典型回合长度在 100-400 步之间）.Rollout = 300*1000/(8*n_steps)
    plt.subplot(2, 1, 2)
    plt.plot(callback.total_loss, label="Total Loss")
    plt.title('Total Training Loss')
    plt.xlabel('Rollout')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{model_name}_training_curves.png")
    plt.close()
    print(f"The training curve has been saved to {model_name}_training_curves.png")

def main():
    model_type= "BipedalWalker-v3"
    # 使用 Optuna 进行超参数优化，采用概率建模，推测哪些参数更有可能带来更高的奖励，然后优先尝试那些参数。
    # 默认最优参数 https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/ppo/ppo.html#PPO
    # best_params = {'learning_rate': 0.001498919063649211, 'n_steps': 1024, 'batch_size': 256, 'gamma': 0.993, 'gae_lambda': 0.9600000000000001, 'clip_range': 0.25, 'ent_coef': 0.020190545985733898, 'n_epochs': 19}
    study = optuna.create_study(direction="maximize") # 设置奖励最大化，同时也有可能得到最差的结果
    study.optimize(partial(OptunaTrial,model_type=model_type,parallel=8), n_trials=200) # 调用并运行200次目标函数OptunaTrial，寻找最优超参数组合，并用类似打擂台的方式保存目前为止的最有参数和结果

    # 做完200次独立实验之后，打印最终的最佳参数组合和最佳成绩
    print("Best hyperparameters:", study.best_params
          , "\nBest accuracy:", study.best_value
          , "\nBest step:", study.best_params['n_steps'])

    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    # 打印optuna日志
    trials_df = study.trials_dataframe()
    completed_trials = trials_df[trials_df['state'] == 'COMPLETE']
    completed_trials.to_csv(f"{current_time}_ppo_lunarlander_optuna_log.csv", index=False)

    model_name = f"{current_time}_ppo_lunarlander"

    # 基于最佳参数生成飞行控制模型并获取回调数据,设置1000轮训练，并行数为8
    model, callback = TrainPpoAgent(study.best_params,model_type, model_name,1000,20_000_000,8)

    random_seed = int(datetime.datetime.now().timestamp())  # 添加随机种子，确保每一次跑数中gif和video的环境一样
    # 可视化：生成 Gif、视频和绘制训练曲线
    recordGif(model_type,model, f"{current_time}_ppo_lunarlander.gif", 30, 10,random_seed) #gif 帧率30，时长10秒
    recordVideo(model_type,model, f"{current_time}_ppo_lunarlander",random_seed)
    plotTrainingCurves(callback, model_name)

if __name__ == "__main__":
    main()