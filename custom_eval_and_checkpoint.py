from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np

def custome_schedule(total_timesteps, mode="fixed", num_runs=10):
    """
    Generate a schedule of timesteps.
    
    Args:
        total_timesteps (int): Total number of training timesteps.
        mode (str): 'fixed' for evenly spaced intervals, 'log' for logarithmically spaced points.
        num_runs (int): Number of run points desired.
    
    Returns:
        List[int]: A list of timesteps when evaluations should occur.
    """
    if mode == "fixed":
        interval = total_timesteps // num_runs
        return list(range(interval, total_timesteps + 1, interval))
    elif mode == "log":
        eval_points = np.unique(np.logspace(0, np.log10(total_timesteps), num=num_runs, dtype=int))
        return eval_points.tolist()
    else:
        raise ValueError("Unsupported mode. Use 'fixed' or 'log'.")

class CustomEvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_schedule, n_eval_episodes=5, best_model_save_path=None, log_path=None, verbose=1):
        super(CustomEvalCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_schedule = eval_schedule
        self.n_eval_episodes = n_eval_episodes
        self.best_mean_reward = -np.inf
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path

    def _on_step(self) -> bool:
        # Check if current timestep is in the evaluation schedule.
        if self.num_timesteps in self.eval_schedule:
            rewards = []
            for _ in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                done = False
                episode_reward = 0.0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, _ = self.eval_env.step(action)
                    episode_reward += reward
                rewards.append(episode_reward)
            mean_reward = np.mean(rewards)
            if self.verbose:
                print(f"Evaluation at timestep {self.num_timesteps}: mean reward: {mean_reward:.2f}")
            
            # Optionally, save the best model.
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.best_model_save_path is not None:
                    model_path = os.path.join(self.best_model_save_path, f'best_model.zip')
                    self.model.save(model_path)
                    if self.verbose:
                        print(f"New best model saved to {model_path}")
        return True


class CustomCheckpointCallback(BaseCallback):
    def __init__(self, checkpoint_schedule, checkpoint_dir, name_prefix="ckpt", verbose=1):
        """
        Callback for saving a checkpoint at specific timesteps.
        
        Args:
            checkpoint_schedule (List[int]): A list of timesteps at which to save checkpoints.
            checkpoint_dir (str): Directory where the checkpoints will be saved.
            name_prefix (str): Prefix for the checkpoint file names.
            verbose (int): Verbosity level.
        """
        super(CustomCheckpointCallback, self).__init__(verbose)
        self.checkpoint_schedule = checkpoint_schedule
        self.checkpoint_dir = checkpoint_dir
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        # Check if the current timestep is in our checkpoint schedule.
        if self.num_timesteps in self.checkpoint_schedule:
            model_path = os.path.join(self.checkpoint_dir, f"{self.name_prefix}_{self.num_timesteps}.zip")
            self.model.save(model_path)
            if self.verbose:
                print(f"Checkpoint saved at timestep {self.num_timesteps} to {model_path}")
        return True

