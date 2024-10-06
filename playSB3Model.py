from stable_baselines3 import PPO
from gymAPI import UnityEnvAPI

# Create the environment
env = UnityEnvAPI("Build/RL Environment")

# Load the model
model = PPO.load("model_name.zip")

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
    # Close the environment
    env.close()
