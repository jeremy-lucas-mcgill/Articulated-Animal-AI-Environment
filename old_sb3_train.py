from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from gymAPI import UnityEnvAPI
import torch

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create the training environment
env = UnityEnvAPI("Build/RL Environment", no_graphics=False)
env = Monitor(env)

# Set the model
model_name = "model_name.zip"

try:
    model = PPO.load(model_name, env, device=device)
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Saved model not found. Creating a new model.")
    model = PPO("MlpPolicy", env, verbose=1, device=device)

try:
    # Train the model
    model.learn(total_timesteps=10000)
except KeyboardInterrupt:
    print("Training interrupted. Saving the model...")
finally:
    # Close the environment
    env.close()

# Save the model
model.save(model_name)
