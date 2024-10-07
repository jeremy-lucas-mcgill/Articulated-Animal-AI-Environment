from stable_baselines3 import PPO
from gymAPI import make_unity_env

# Create the environment
env = make_unity_env()

# Load the model
model = PPO.load("model_name.zip")

# Run the environment using the loaded model
obs, _ = env.reset()
done = False
total_reward = 0

try:
    while not done:
        # Use the model to predict the action
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        env.render()

except KeyboardInterrupt:
    print("Running interrupted.")

finally:
    # Close the environment
    env.close()
