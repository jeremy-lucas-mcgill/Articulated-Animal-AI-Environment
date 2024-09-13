from gymAPI import UnityEnvAPI

#Create the environment
env = UnityEnvAPI("Build/RL Environment")

#Load the model
model = "Load Your Model Here"

#Run the environment using the loaded model
obs = env.reset()
done = False
total_reward = 0

try:
    while not done:
        #Implement your model running logic here
        pass

except KeyboardInterrupt:
    print("Running interrupted.")

finally:
    #Close the environment
    env.close()
