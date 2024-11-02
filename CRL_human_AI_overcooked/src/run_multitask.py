import datetime
import os
import pprint
import time
import math as mth
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.loggingus import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath
import numpy as np
import copy

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer, Best_experience_Buffer
from components.transforms import OneHot



def run_multitask(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)

def run_sequential(args, logger):

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        #"policy": {"vshape": (env_info["n_agents"],)}
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)
    off_buffer = ReplayBuffer(scheme, groups, args.off_buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
    if args.checkpoint_path != '':
        load_mac_weight(mac, args.checkpoint_path, logger)
    
    with open(args.human_models_dir, 'r') as f:
        lines = f.readlines()
        human_models = [line.replace('\n', '') for line in lines]
        
    partner_human_num = len(human_models)
    best_avg_reward_old = 0
    
    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    # Init runner and Give runner the mac
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    while runner.t_env <= args.t_max:

        # critic running log
        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "q_taken_mean": [],
            "q_max_mean": [],
            "q_min_mean": [],
            "q_max_var": [],
            "q_min_var": []
        }
        
        for human_index in range(0, partner_human_num):

            mac.init_human_model(human_models[human_index])

            # Run for a whole episode at a time
            episode_batch = runner.run(test_mode=False)
            buffer.insert_episode_batch(episode_batch)
            off_buffer.insert_episode_batch(episode_batch)

            if buffer.can_sample(args.batch_size) and off_buffer.can_sample(args.off_batch_size):
                #train critic normall
                uni_episode_sample = buffer.uni_sample(args.batch_size)
                off_episode_sample = off_buffer.uni_sample(args.off_batch_size)
                max_ep_t = max(uni_episode_sample.max_t_filled(), off_episode_sample.max_t_filled())
                uni_episode_sample = process_batch(uni_episode_sample[:, :max_ep_t], args)
                off_episode_sample = process_batch(off_episode_sample[:, :max_ep_t], args)
                learner.train_critic(uni_episode_sample, best_batch=off_episode_sample, log=running_log)

                #train actor
                episode_sample = buffer.sample_latest(args.batch_size)
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = process_batch(episode_sample[:, :max_ep_t], args)
                learner.train(episode_sample, runner.t_env, running_log)

            # Execute test runs once in a while
            if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

                logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
                logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                    time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
                last_time = time.time()
                last_test_T = runner.t_env
                avg_reward = test_multitask(args, mac, logger, runner, human_models, task_num=partner_human_num)
                if avg_reward > best_avg_reward_old:
                    logger.console_logger.info('best_test_return_mean_partner: {}'.format(avg_reward))
                    save_path = os.path.join(args.local_results_path, "models", args.unique_token, 'best')
                    os.makedirs(save_path, exist_ok=True)
                    logger.console_logger.info("Saving models to {}".format(save_path))
                    learner.save_models(save_path)
                    best_avg_reward_old = avg_reward

            if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
                model_save_time = runner.t_env
                save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
                os.makedirs(save_path, exist_ok=True)
                logger.console_logger.info("Saving models to {}".format(save_path))
                learner.save_models(save_path)

            episode += args.batch_size_run
            if (runner.t_env - last_log_T) >= args.log_interval:
                logger.log_stat("episode", episode, runner.t_env)
                logger.print_recent_stats()
                last_log_T = runner.t_env

    # test performance
    load_path = os.path.join(args.local_results_path, "models", args.unique_token, str(human_index))
    load_mac_weight(mac, load_path, logger)
    avg_reward = test_multitask(args, mac, logger, runner, human_models, task_num=partner_human_num)
    logger.console_logger.info("\naverage score: {}".format(avg_reward))   
            
    runner.close_env()
    logger.console_logger.info("Finished Training")
        
    
def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config


def process_batch(batch, args):

    if batch.device != args.device:
        batch.to(args.device)
    return batch

def load_mac_weight(mac, checkpoint_path, logger):
    if checkpoint_path == "":
        return 

    if not os.path.isdir(checkpoint_path):
        logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(checkpoint_path))
        return
    
    timesteps = []
    timestep_to_load = 0

    best_checkpoint_path = None
    for name in os.listdir(checkpoint_path):
        full_name = os.path.join(checkpoint_path, name)
        if os.path.isdir(full_name) and name.isdigit():
            timesteps.append(int(name))
        if os.path.isdir(full_name) and name == 'best':
            best_checkpoint_path = full_name

    if best_checkpoint_path is not None:
        model_path = best_checkpoint_path
    else:
        timestep_to_load = max(timesteps)
        model_path = os.path.join(checkpoint_path, str(timestep_to_load))

    logger.console_logger.info("Loading model from {}".format(model_path))
    mac.load_models(model_path)
    
def test_multitask(args, mac, logger, runner, human_models, task_num=12):
    
    n_test_runs = max(1, args.test_nepisode // runner.batch_size)
    total_test_rewards = []
    for j in range(0, task_num):
        mac.init_human_model(human_models[j])
        test_rewards = []
        for _ in range(n_test_runs):
            runner.run(test_mode=True, keep_best_value=False)
            test_rewards.append(runner.test_rewards)
            logger.console_logger.info(
                "task{} : {}\t".format(j, np.mean(test_rewards))
            )
        total_test_rewards.append(np.mean(test_rewards))
        logger.log_stat("test_task{}_return".format(j) , total_test_rewards[j], runner.t_env)
    
    return np.mean(total_test_rewards)


