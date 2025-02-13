import argparse
import os
import stable_baselines3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from gymAPI import make_unity_env

"""
example of how to run it:

python test.py --model PPO --model_path models/best_model.zip --path ./AAAI-MacOS.app --n-eval-episodes 5 --use-raycasts --max-steps 1000
"""

# Global list to hold all parameters (so that the test env uses the same settings as training)
parameters = []

def make_see_env(path):
    """
    Creates a Unity environment with rendering enabled (i.e. to “see” what's happening)
    and applies the additional training/testing parameters.
    """
    # Here, we assume max_steps is passed as a parameter; you can adjust as needed.
    return make_unity_env(max_steps=args.max_steps, file_name=path, additional_args=parameters)


def test(model_used, model_path, env_path, n_eval_episodes=1):
    # Wrap the environment with Monitor for better logging
    env = Monitor(make_see_env(env_path))
    model = model_used.load(model_path, env=env)
    
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, render=True)
    print(f"Evaluation over {n_eval_episodes} episodes: Mean Reward = {mean_reward:.2f} +/- {std_reward:.2f}")
    env.close()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Test a trained model with Unity environment")
    # Basic test options
    argparser.add_argument("--model", default="PPO", help="Type of model to test (e.g. PPO, A2C, SAC, etc.)")
    argparser.add_argument("--model_path", default="", help="Path to the trained model (e.g. models/best_model.zip)")
    argparser.add_argument("--path", type=str, default="Build/RL Environment", help="Path to the Unity environment executable")
    argparser.add_argument("--n-eval-episodes", type=int, default=1, help="Number of evaluation episodes")
    
    # Environment / Agent settings (kept identical to training)
    argparser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    argparser.add_argument("--seed", type=int, default=None, help="Training seed (should be same as during training)")
    argparser.add_argument("--use-camera", action="store_true", default=False, help="Agent will use a camera")
    argparser.add_argument("--use-raycasts", action="store_true", default=False, help="Agent will use raycasts")
    argparser.add_argument("--grayscale", action="store_true", default=False, help="Grayscale camera feed")
    argparser.add_argument("--camera-resolution", type=int, default=32, help="Camera resolution (KxK)")
    argparser.add_argument("--num-rays", type=int, default=8, help="Number of rays per side")
    argparser.add_argument("--ray-angle", type=int, default=45, help="Angle for each ray")
    argparser.add_argument("--ground-reward", action="store_true", default=False, help="Agent gets a reward for head-off-ground")
    
    # Level-specific parameters (if applicable)
    argparser.add_argument("--L0-freq", type=float, default=0, help="L0 frequency")
    argparser.add_argument("--L0-minDiff", type=int, default=0, help="L0 minimum difficulty")
    argparser.add_argument("--L0-maxDiff", type=int, default=0, help="L0 maximum difficulty")
    argparser.add_argument("--L1-freq", type=float, default=0, help="L1 frequency")
    argparser.add_argument("--L1-minDiff", type=int, default=0, help="L1 minimum difficulty")
    argparser.add_argument("--L1-maxDiff", type=int, default=0, help="L1 maximum difficulty")
    
    args = argparser.parse_args()

    # Extend the global parameters list so that the same settings are passed to the Unity environment.
    parameters.extend([
        f"--seed={args.seed}",
        f"--max-steps={args.max_steps}",
        f"--use-camera={args.use_camera}",
        f"--use-raycasts={args.use_raycasts}",
        f"--grayscale={args.grayscale}",
        f"--camera-resolution={args.camera_resolution}",
        f"--num-rays={args.num_rays}",
        f"--ray-angle={args.ray_angle}",
        f"--ground-reward={args.ground_reward}",
        f"--L0-freq={args.L0_freq}",
        f"--L0-minDiff={args.L0_minDiff}",
        f"--L0-maxDiff={args.L0_maxDiff}",
        f"--L1-freq={args.L1_freq}",
        f"--L1-minDiff={args.L1_minDiff}",
        f"--L1-maxDiff={args.L1_maxDiff}"
    ])

    # Retrieve the model class from stable_baselines3 (e.g. PPO, A2C, SAC, etc.)
    model_class = getattr(stable_baselines3, args.model)
    
    # Call the test function
    test(model_class, args.model_path, args.path, args.n_eval_episodes)
