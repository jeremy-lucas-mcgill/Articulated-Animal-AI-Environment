import os
import numpy as np
from torch import nn
from stable_baselines3 import *
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor
from gymAPI import make_unity_env
import argparse
import torch

if __name__ == "__main__":
    ###########################
    ##    HYPERPARAMETERS    ##
    ###########################

    MODEL_TYPE = SAC
    MODEL_KWARGS = {
        # "batch_size": 4,
    }
    POLICY_KWARGS = {
        # "net_arch": dict(pi=[128, 128, 128], qf=[128, 128, 128]),
        # "activation_fn": nn.Tanh,
        # "log_std_init": -1,
    }

    ###########################
    ##   ARGUMENT  PARSING   ##
    ###########################

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--silent",
        action="store_true",
        help="Pass this flag to prevent printing logs to stdout",
    )
    argparser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=5,
        help="Number of episodes to evaluate the model on",
    )
    argparser.add_argument(
        "--n-evals",
        type=int,
        default=int(100),
        help="Number of evaluations to save during training",
    )
    argparser.add_argument(
        "--n-checkpoints",
        type=int,
        default=int(10),
        help="Number of checkpoints to save during training",
    )
    argparser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to checkpoint to continue training from (must point to .zip file, with or without the .zip extension in the path)",
    )
    argparser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Subfolder path to save training results in",
    )
    argparser.add_argument(
        "--n-steps",
        type=int,
        default=int(1_000_000),
        help="Total timesteps to train policy for",
    )

    args = argparser.parse_args()
    print(args)
    SILENT = args.silent
    N_EVAL_EPISODES = args.n_eval_episodes
    TOTAL_TIMESTEPS = args.n_steps
    N_EVALS = args.n_evals
    N_CHECKPOINTS = args.n_checkpoints
    if args.name is None:
        args.name = MODEL_TYPE.__name__

    if args.ckpt is not None:
        CHECKPOINT = args.ckpt.lstrip().rstrip()
        LOG_DIR_SUFFIX = "/" + os.path.splitext(os.path.basename(CHECKPOINT))[0]
    else:
        CHECKPOINT = None
        LOG_DIR_SUFFIX = "/0_steps"

    log_dir = "data/{}/".format(args.name.strip()) + LOG_DIR_SUFFIX
    # make dir if it doesnt exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    ##########################
    ##  ENVIRONMENT  SETUP  ##
    ##########################

    env = make_unity_env()
    env = Monitor(env)

    eval_env = make_unity_env(worker_id=1, max_steps=1000)
    eval_env = Monitor(eval_env)

    ##########################
    ## MODEL INITIALIZATION ##
    ##########################

    print("\nBeginning training.\n")
    # Check if CUDA is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if CHECKPOINT is None:
        model = MODEL_TYPE(
            policy="MlpPolicy",
            env=env,
            verbose=0,
            policy_kwargs=POLICY_KWARGS,
            **MODEL_KWARGS,
        )
    else:
        model = MODEL_TYPE.load(
            path=CHECKPOINT,
            env=env,
        )

    ##########################
    ##    TRAINING  LOOP    ##
    ##########################

    checkpoint_callback = CheckpointCallback(
        save_freq=TOTAL_TIMESTEPS // N_CHECKPOINTS,
        save_path=log_dir,
        name_prefix="ckpt",
        verbose=0,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=TOTAL_TIMESTEPS // N_EVALS,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=0,
    )

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[
                eval_callback,
                checkpoint_callback,
            ],
            log_interval=None if SILENT else 1,
            reset_num_timesteps=False,
            progress_bar=False if SILENT else True,
        )
    except KeyboardInterrupt:
        model_filename = log_dir + "/ckpt_{}".format(model.num_timesteps)
        print(
            "Training interrupted. Saving current model state to {}.".format(
                model_filename
            )
        )
        model.save(model_filename)
