from gymAPI import make_unity_env

# Create the environment
env = make_unity_env()

# Load the model
model = "Load Your Model Here"

# Run the environment using the loaded model
obs, _ = env.reset()
done = False
total_reward = 0

try:
    while not done:
        # Implement your model training logic here
        pass

except KeyboardInterrupt:
    print("Training interrupted.")

finally:
    # Close the environment
    env.close()
