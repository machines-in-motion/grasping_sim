#!/usr/bin/env python3
import os, sys, os.path as osp, shutil, argparse, math
import yaml
from baselines.common import set_global_seeds, tf_util as U
from baselines.ppo1.mlp_policy import MlpPolicy

from grasping_env.utils import instantiate_hyperparams, ordered_dump
from grasping_env import load_config, make_from_config


def main():
    pwd = osp.dirname(osp.realpath(__file__))
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('config', type=str, help="Specify the config file.")
    parser.add_argument('-j', type=int, help="Job ID.")
    parser.add_argument('-p', default="", type=str, help="Prefix for saving the logs.")
    parser.add_argument('-n', default=-1, type=int, help="Max number of time steps." +
                        "If n > 0 overrides the number of time steps specified in the config file.")
    parser.add_argument('-s', default=20, type=int, help="Save the policy every S iterations.")
    parser.add_argument('-i', default=-1, type=int, help="Index of specific pre-grasp to use.")

    args = parser.parse_args()
    job_id = args.j
    prefix = args.p
    log_dir = osp.join(pwd, "logs", prefix, str(job_id))
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    if not osp.isfile(osp.join(pwd, "logs", prefix, "config.yaml")):
        shutil.copy2(args.config, osp.join(pwd, "logs", prefix, "config.yaml"))
    sys.stdout = open(osp.join(pwd, "logs", prefix, str(job_id) + ".out"), 'a')
    sys.stderr = open(osp.join(pwd, "logs", prefix, str(job_id) + ".err"), 'a')

    config_path = osp.join(log_dir, "config.yaml")
    if osp.isfile(config_path):
        # If already created, avoid instantiating.
        config = load_config(config_path)
        # In case the num_timesteps got updated.
        args_config = load_config(args.config)
        config["num_timesteps"] = args_config["num_timesteps"]
    else:
        config = load_config(args.config)
        header = instantiate_hyperparams(config, job_id)
        print(header, flush=True)
        with open(config_path, 'w') as file:
            ordered_dump(config, file)
            # Make sure write is instantaneous.
            file.flush()
            os.fsync(file)

    max_timesteps = args.n if args.n > 0 else config["num_timesteps"]

    # Environment.
    config["log_dir"] = log_dir
    env = make_from_config(config)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    if osp.isfile(osp.join(log_dir, 'job_info.yaml')):
        with open(osp.join(log_dir, 'job_info.yaml'), 'r') as file:
            try:
                job_info = yaml.load(file)
                env.n_episodes = episodes_so_far = job_info['episodes_so_far']
                timesteps_so_far = job_info['timesteps_so_far']
                iters_so_far = job_info['iters_so_far']
            except yaml.YAMLError as exc:
                print(exc)
                sys.exit(1)

    # Filter only a single pregrasp.
    if args.i >= 0:
        env.obj_pregrasps[env.object] = [env.obj_pregrasps[env.object][args.i]]
        env.obj_phys_scores[env.object] = [env.obj_phys_scores[env.object][args.i]]

    # Algorithm.
    sess = U.make_session(num_cpu=config["num_cpu"])
    sess.__enter__()

    if config["seed"] >= 0:
        set_global_seeds(config["seed"])

    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=env.observation_space, ac_space=env.action_space,
                         hid_sizes=config["hid_sizes"],
                         gaussian_fixed_var=config["gaussian_fixed_var"],
                         use_obfilter=config["use_obfilter"])

    # Do not change before, since agent and environment use different log_every value.
    config["log_every"] = args.s
    try:
        if config["type"] == "trpo":
            from baselines.trpo_mpi import trpo_mpi
            trpo_mpi.learn(env, policy_fn,
                           max_timesteps=max_timesteps,
                           episodes_so_far=episodes_so_far,
                           timesteps_so_far=timesteps_so_far,
                           iters_so_far=iters_so_far,
                           **config)
        elif config["type"] == "ppo":
            from baselines.ppo1 import pposgd_simple
            pposgd_simple.learn(env, policy_fn,
                                max_timesteps=max_timesteps,
                                episodes_so_far=episodes_so_far,
                                timesteps_so_far=timesteps_so_far,
                                iters_so_far=iters_so_far,
                                **config)
    except Exception as err:
        print("Caught an exception: {}".format(err))
        env.close()
        sys.exit(1)

    env.close()

if __name__ == '__main__':
    main()
