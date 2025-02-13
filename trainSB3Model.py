import os
import subprocess
import time
import numpy as np
from torch import nn
import stable_baselines3
from stable_baselines3 import *
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor
from gymAPI import make_unity_env
import argparse
import torch
import random
from stable_baselines3.common.logger import configure
import shutil


if __name__ == "__main__":
    ###########################
    ##    HYPERPARAMETERS    ##
    ###########################

    MODEL_TYPE = None
    MODEL_KWARGS = {

    }
    POLICY_KWARGS = {

    }

    ###########################
    ##   ARGUMENT  PARSING   ##
    ###########################

    argparser = argparse.ArgumentParser()
    ##RL Arguments
    argparser.add_argument("--model", required=True, help="Type of model you want to train.")
    argparser.add_argument("--silent", action="store_true", help="Silent mode (no logs)")
    argparser.add_argument("--n-eval-episodes", type=int, default=1, help="Eval episodes")
    argparser.add_argument("--n-evals", type=int, default=1, help="Number of evaluations")
    argparser.add_argument("--n-checkpoints", type=int, default=1, help="Checkpoints")
    argparser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint")
    argparser.add_argument("--name", type=str, default=None, help="Name of run")
    argparser.add_argument("--max-steps", type=int, default=1_000, help="Max steps per episode")
    argparser.add_argument("--n-steps", type=int, default=1_000, help="Total steps")
    argparser.add_argument("--path", type=str, default="Build/RL Environment", help="Path to environment.")
    argparser.add_argument("--no-graphics", action="store_true", default=False, help="Handles displaying graphics.")
    argparser.add_argument("--seed", type=int, default=None, help="Set a training seed. Defaults to a random value")

    #hyperparameters arguments
    # argparser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate for the optimizer")
    # argparser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    # argparser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    # argparser.add_argument("--ent-coef", type=float, default=0.0, help="Entropy coefficient")

    #Agent Arguments
    argparser.add_argument("--ground-reward", action="store_true", default=False, help="Agent gets a reward if its head is off the ground.")
    argparser.add_argument("--use-camera", action="store_true", default=False, help="Agent will use a camera with a default resolution of 84x84.")
    argparser.add_argument("--use-raycasts", action="store_true", default=False, help="Agent will use a raycasts with a default 8 rays per direction with a max angle of 45 degrees.")
    argparser.add_argument("--grayscale", action="store_true", default=False, help="Agent's camera will be grayscaled. Default is not grayscale.")
    argparser.add_argument("--camera-resolution", type=int, default=32, help="Change agent's camera resolution to size KxK. Default is 32x32.")
    argparser.add_argument("--num-rays", type=int, default=8, help="Change agent's number of rays. Default is 8 per side.")
    argparser.add_argument("--ray-angle", type=int, default=45, help="Change agent's ray angle. Default is 45 degrees per side.")

    #Test Arguments
    argparser.add_argument("--L0-freq", type=float, default=0, help="L0 frequency")
    argparser.add_argument("--L0-minDiff", type=int, default=0, help="L0 min difficulty")
    argparser.add_argument("--L0-maxDiff", type=int, default=0, help="L0 max difficulty")
    argparser.add_argument("--L1-freq", type=float, default=0, help="L1 frequency")
    argparser.add_argument("--L1-minDiff", type=int, default=0, help="L1 min difficulty")
    argparser.add_argument("--L1-maxDiff", type=int, default=0, help="L1 max difficulty")

    args = argparser.parse_args()
    print(args)

    parameters = [
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
    ]

    ###########################
    ##   TRAINING SETTINGS   ##
    ###########################

    SILENT = args.silent
    N_EVAL_EPISODES = args.n_eval_episodes
    TOTAL_TIMESTEPS = args.n_steps
    N_EVALS = args.n_evals
    N_CHECKPOINTS = args.n_checkpoints
    PATH = args.path
    MAX_STEPS = args.max_steps
    GRAPHICS = args.no_graphics

    MODEL_TYPE = getattr(stable_baselines3, args.model)

    if args.name is None:
        args.name = MODEL_TYPE.__name__

    # --------------------------------------
    # SEED MANAGEMENT
    # --------------------------------------
    if args.seed is None:
        args.seed = random.randint(0, 2**31 - 1)
    print(f"Using seed: {args.seed}")

    # Seed Python, NumPy, Torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    if args.ckpt is not None:
        CHECKPOINT = args.ckpt.strip()
        LOG_DIR_SUFFIX = "/" + os.path.splitext(os.path.basename(CHECKPOINT))[0]
    else:
        CHECKPOINT = None
        LOG_DIR_SUFFIX = "/seed-" + str(args.seed)

    # --------------------------------------
    # DIRECTORY SETUP
    # --------------------------------------
    models_dir = f"models/{args.name.strip()}" + LOG_DIR_SUFFIX
    logs_dir = f"logs/{args.name.strip()}" + LOG_DIR_SUFFIX

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    ##########################
    ##  ENVIRONMENT  SETUP  ##
    ##########################

    try:
        worker1 = random.randint(0,65534)
        worker2 = worker1 + 1
        env = make_unity_env(no_graphics=GRAPHICS, worker_id=worker1, max_steps=MAX_STEPS, file_name=PATH,additional_args=parameters)
        env = Monitor(env)

        eval_env = make_unity_env(no_graphics=GRAPHICS, worker_id=worker2, max_steps=MAX_STEPS, file_name=PATH,additional_args=parameters)
        eval_env = Monitor(eval_env)

        ##########################
        ## MODEL INITIALIZATION ##
        ##########################

        print("\nBeginning training.\n")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        if CHECKPOINT is None:
            print("Creating new model.")
            model = MODEL_TYPE(
                policy="MlpPolicy",
                env=env,
                verbose=0,
                policy_kwargs=POLICY_KWARGS,
                tensorboard_log=logs_dir,    # Use logs_dir for TensorBoard
                **MODEL_KWARGS,
            )
        else:
            print("Loading model from checkpoint.")
            model = MODEL_TYPE.load(
                path=CHECKPOINT,
                env=env,
            )

        logger = configure(logs_dir, ["stdout", "csv", "log", "tensorboard"])
        model.set_logger(logger)

        ##########################
        ##    TRAINING  LOOP    ##
        ##########################

        checkpoint_callback = CheckpointCallback(
            save_freq=TOTAL_TIMESTEPS // N_CHECKPOINTS,
            save_path=models_dir,
            name_prefix="ckpt",
            verbose=0,
        )

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=models_dir,
            log_path=logs_dir,
            eval_freq=TOTAL_TIMESTEPS // N_EVALS,
            n_eval_episodes=N_EVAL_EPISODES,
            deterministic=True,
            render=False,
            verbose=0,
        )

        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[
                eval_callback,
                checkpoint_callback,
            ],
            log_interval=None if SILENT else 1,
            reset_num_timesteps=False,
            # progress_bar=False if SILENT else True,
        )
    except Exception as e:
        print(f"Error during training: {e}")
        for directory in [models_dir, logs_dir]:
            if os.path.isdir(directory) and not os.listdir(directory):
                shutil.rmtree(directory)
    except KeyboardInterrupt:
        model_filename = models_dir + "/ckpt_{}".format(model.num_timesteps)
        print(
            "Training interrupted. Saving current model state to {}.".format(
                model_filename
            )
        )
        model.save(model_filename)
    finally:
        env.close()
        eval_env.close()
        print("Unity environment closed.")
