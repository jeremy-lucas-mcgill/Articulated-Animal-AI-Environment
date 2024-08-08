from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from gymAPI import UnityEnvAPI  # Import the UnityEnvAPI class
import numpy as np

# Create the environment
env = UnityEnvAPI("Build/RL Environment")

# Load the model
model = PPO.load("ppo_policy2.zip")

# Run the environment using the loaded model
obs = env.reset()
done = False
total_reward = 0

try:
    while not done:
        # Use the model to predict the action
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        env.render() 

except KeyboardInterrupt:
    print("Running interrupted.")

finally:
    # Ensure the environment is closed
    env.close()
