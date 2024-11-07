import matplotlib.pyplot as plt
import numpy as np

eval_file_dirs = ["data/SAC/0_steps/"]

#####################

timesteps = []
rewards = []
ep_lengths = []

for i in range(len(eval_file_dirs)):
    evaluations = np.load(eval_file_dirs[i] + "evaluations.npz")

    timesteps.extend(evaluations["timesteps"])
    rewards.extend(np.mean(evaluations["results"], axis=1))
    ep_lengths.extend(np.mean(evaluations["ep_lengths"], axis=1))
    num_episodes_averaged = evaluations["results"].shape[1]

#### PLOT REWARDS

if len(rewards) > 1:
    fig, ax1 = plt.subplots()

    ax1.plot(timesteps, rewards, color="blue")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Reward", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    # ax1.set_ylim(
    #     [min(rewards) - 0.05 * (max(rewards) - min(rewards)), max(rewards) * 1.05]
    # )

    # Add title and legend
    plt.title("Evaluation Reward (avg. over {} episodes)".format(num_episodes_averaged))
    fig.tight_layout()  # To ensure no label overlap

    # Show grid and plot
    plt.grid(True)
    plt.show()

print("Rewards: {}".format(rewards))
