
# Implement a deep reinforcement learning agent for a game or environment of OpenAI Gym or Gymnasium.
# Use the lunar_lander environment: https://gymnasium.farama.org/environments/box2d/lunar_lander/.
# Please plot the learning progress of your method from 0 to 1000 episodes.
# You can have a figure to show rewards and another figure to show training loss.
# Please use a video or gifs or figures to demonstrate how your agent works.

# 1，optuna训练200次, 找最优超参数（外部规则）
# 2，PPO模型训练1000个episode，找最优神经网络权重（内部策略）

# 1个episode是基于参数的1轮实验，1个模型可能会需要很多的step来完成，但是每次环境不一样可能会导致不同的独立实验需要的步数不一样
# 4，奖励值是以一次episode为周期汇总的，因此奖励曲线的x轴的长度等于episode的数值
# 5，损失值是以步数为周期汇总的
# 奖励曲线的x轴的长度=episode的次数
# 损失曲线的x轴的长度=1000轮实验的实际总步数/study.best_params['n_steps']
        # 奖励组成：
        # 主奖励：成功着陆到 (0, 0) 坐标附近，奖励约为 100-140 分。
        # 燃料消耗惩罚：每使用一次推进器会减少奖励。
        # 时间惩罚：更快着陆奖励更高。
        # 接触奖励：腿接触地面加分（每只腿 10 分）。
        # 失败惩罚：坠毁（-100 分）或偏离目标太远。
import optuna,imageio,datetime,gymnasium,matplotlib.pyplot as plt
from typing import Optional
from gymnasium.wrappers import RecordVideo
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
        self.loss_history = []    # 从日志记录损失
        self.current_episode_reward = 0
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
            if loss is not None:
                self.loss_history.append(loss)

# 获取最佳参数，每一次trial实验时，会自动基于我设置的条件通过trial.suggest_xxx 方法生成参数，（条件限制：下限，上限，变化幅度）
def OptunaTrial(trial):
    hyperparams = {
        "learning_rate": trial.suggest_float("learning_rate", 0.00001, 0.01, log=True), # 学东西的速度
        "n_steps": trial.suggest_int("n_steps", 2048, 4096, step=256),               # 更新的步数，决定实验多少次再总结经验和记录损失
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]), #表示每次参数更新时使用的样本数量
        "gamma": trial.suggest_float("gamma", 0.95, 0.9995, step=0.0005),             # 未来奖励权重，看重长远还是眼前
        "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 0.99, step=0.01),     # 决定关注单步还是整场评估
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.3),       # 通过剪切概率比（probability ratio）,限制策略更新的幅度
        "ent_coef": trial.suggest_float("ent_coef", 0.0001, 0.1, log=True),           # 探索性：冒险精神大小，冒大险可能拿到最大收益，也可能犯最大失误
        "n_epochs": trial.suggest_int("n_epochs", 3, 10)                            # 复习几遍
    }

    # Create the environment，初次创建的环境只是一个完整的游戏场景，飞行器没驱动，没动作
    env = DummyVecEnv([lambda: Monitor(gymnasium.make("LunarLander-v3")) for _ in range(4)])
    model = PPO("MlpPolicy", env, verbose=0, **hyperparams)
    try:
        model.learn(total_timesteps=50_000)
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
        trial.report(mean_reward, step=0)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    finally:
        env.close()
    return mean_reward

# 基于最佳参数训练模型
def TrainPpoAgent(best_params, model_name):
    env = DummyVecEnv([lambda: Monitor(gymnasium.make("LunarLander-v3"))] * 4)
    model = PPO("MlpPolicy", env, verbose=1, **best_params)  # verbose=1 启用日志

    # 配置静默日志记录器，不输出到控制台，保存在本地
    logger = SilentLogger(log_file=f"{model_name}_training_log.csv")
    model.set_logger(logger)

    callback = TrainingCallback(total_episodes=1000, verbose=1)

    print("开始训练，直到 1000 个 episode...")
    # 训练，不超过1Million步长，让它学会控制飞行器平稳降落，并将训练数据更新到callback中
    model.learn(total_timesteps=10_000_000, callback=callback)
    model.save(model_name)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10) # 测试模型的表现，让它玩 10 次游戏，计算平均奖励和标准差。
    print(f"Mean Reward = {mean_reward}, Std Reward = {std_reward}")
    env.close()
    return model, callback

# 录制GIF动图
def recordGif(model, filename, fps=30, duration=10):
    with Monitor(gymnasium.make("LunarLander-v3", render_mode="rgb_array")) as env:
        frames = []
        obs, _ = env.reset()
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
def recordVideo(model, videoName):
    env = RecordVideo(
        Monitor(gymnasium.make("LunarLander-v3", render_mode="rgb_array")),
        video_folder=".",
        name_prefix=videoName,
        episode_trigger=lambda x: True
    )
    with env:
        obs, _ = env.reset()
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, _ = env.step(action)
            if done or truncated: # 任务自然结束（着陆或坠毁）或时间用尽则停止
                break
    print(f"Video saved to {videoName}")

# 绘制训练曲线
def plotTrainingCurves(callback, model_name):
    plt.figure(figsize=(20, 5))
    # 奖励曲线
    plt.subplot(1, 2, 1)
    plt.plot(callback.episode_rewards)
    plt.title('Episode Rewards (from Log)')
    plt.xlabel('Rollout')
    plt.ylabel('Average Episode Reward')
    plt.grid(True)
    # 损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(callback.loss_history)
    plt.title('Training Loss')
    plt.xlabel('Rollout')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{model_name}_training_curves.png")
    plt.close()
    print(f"The training curve has been saved to {model_name}_training_curves.png")

def main():
    # 使用 Optuna 进行超参数优化，采用概率建模，推测哪些参数更有可能带来更高的奖励，然后优先尝试那些参数。
    study = optuna.create_study(direction="maximize") # 设置奖励最大化，同时也有可能得到最差的结果
    study.optimize(OptunaTrial, n_trials=200) # 调用并运行150次目标函数OptunaTrial，寻找最优超参数组合，并用类似打擂台的方式保存目前为止的最有参数和结果
    # 做完200次独立实验之后，打印最终的最佳参数组合和最佳成绩
    print("Best hyperparameters:", study.best_params
          , "\nBest accuracy:", study.best_value
          , "\nBest step:", study.best_params['n_steps'])

    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    model_name = f"{current_time}_ppo_lunarlander"

    # 基于最佳参数生成飞行控制模型并获取回调数据
    model, callback = TrainPpoAgent(study.best_params, model_name)

    # 可视化：生成 Gif、视频和绘制训练曲线
    recordGif(model, f"{current_time}_ppo_lunarlander.gif", fps=30, duration=10) #gif 帧率30，时长10秒
    recordVideo(model, f"{current_time}_ppo_lunarlander")
    plotTrainingCurves(callback, model_name)

if __name__ == "__main__":
    main()