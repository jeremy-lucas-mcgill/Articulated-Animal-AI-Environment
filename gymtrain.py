from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from gymAPI import UnityEnvAPI  # Import the UnityEnvAPI class
import torch

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create the training environment with a specific worker_id and wrap it with Monitor
env = UnityEnvAPI("Build/RL Environment", no_graphics=False)
env = Monitor(env)
print(f"Observation space shape: {env.observation_space.shape}")
try:
    model = PPO.load("run3.zip", env, device=device)
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Saved model not found. Creating a new model.")
    # Create a new model
    model = PPO(
        'MlpPolicy', 
        env, 
        verbose=1, 
        device=device
    )

try:
    # Train the model with the evaluation callback
    model.learn(total_timesteps=10000)
except KeyboardInterrupt:
    print("Training interrupted. Saving the model...")
finally:
    # Ensure the environments are closed
    env.close()

# Save the model
model.save("model_name")