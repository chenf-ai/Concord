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



def run_CL(_run, _config, _log):

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
    runner.close_env()
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

    with open(args.human_models_dir, 'r') as f:
        lines = f.readlines()
        human_models = [line.replace('\n', '') for line in lines]
    if args.human_num == 2:
        human_models = [[human_models[0], human_models[0]],
                        [human_models[0], human_models[1]],
                        [human_models[0], human_models[2]],
                        [human_models[1], human_models[1]],
                        [human_models[1], human_models[2]],
                        [human_models[2], human_models[2]],]
        
    partner_human_num = len(human_models)
    performance_task = {}

    if args.use_ER:
        replay_buffer = ReplayBuffer(scheme, groups, partner_human_num * 20 * args.batch_size, env_info["episode_limit"] + 1,
                                preprocess=preprocess,
                                device="cpu" if args.buffer_cpu_only else args.device)

    buffer_temp = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer_temp.scheme, groups, args)
    if args.checkpoint_path != '':
        load_mac_weight(mac, args.checkpoint_path, logger)
    
    
    for human_index in range(args.CL_start_index, partner_human_num):
        
        logger.console_logger.info("Beginning training for {} timesteps with partner{}".format(args.t_max, human_index))
        
        mac.init_human_model(human_models[human_index])
        if human_index > 0 and human_index > args.CL_start_index:
            load_path = os.path.join(args.local_results_path, "models", args.unique_token, str(human_index-1))
            load_mac_weight(mac, load_path, logger)
        if args.use_Concord:
            logger.console_logger.info("Set task_id = {} for hypernet".format(human_index))
            mac.set_task_id(human_index)
            logger.console_logger.info(
                "keep old hypernet"
            )
            mac.agent.keep_old_hypernet()

        buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                            preprocess=preprocess,
                            device="cpu" if args.buffer_cpu_only else args.device)
        off_buffer = ReplayBuffer(scheme, groups, args.off_buffer_size, env_info["episode_limit"] + 1,
                            preprocess=preprocess,
                            device="cpu" if args.buffer_cpu_only else args.device)
        
        # Init runner and Give runner the mac
        runner = r_REGISTRY[args.runner](args=args, logger=logger)
        runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
        runner.t_env = human_index * args.t_max

        # Learner
        learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

        if args.use_cuda:
            learner.cuda()
        
        # start training
        episode = 0
        last_test_T = runner.t_env
        last_log_T = runner.t_env
        last_test_CL_T = runner.t_env
        model_save_time = -1

        start_time = time.time()
        last_time = start_time
        
        best_test_returns_old = 0
        best_old_time = 0

        if args.use_ER and human_index > 0:
            start_index = replay_buffer.episodes_in_buffer // 640
            for replay_human_index in range(start_index, human_index):
                load_path1 = os.path.join(args.local_results_path, "models", args.unique_token, str(replay_human_index))
                load_path2 = args.checkpoint_path.replace('/{}/'.format(human_index-1), '/{}/'.format(replay_human_index))
                if os.path.exists(load_path1):
                    load_mac_weight(mac, load_path1, logger)
                else:
                    load_mac_weight(mac, load_path2, logger)
                mac.init_human_model(human_models[replay_human_index])
                logger.console_logger.info("start generate episode memory for task{}".format(replay_human_index))
                runner.run(test_mode=False)
                for _ in range(0, 160):
                    episode_batch = runner.run(test_mode=True, keep_best_value=False)
                    replay_buffer.insert_episode_batch(episode_batch)
                logger.console_logger.info("the length of CL replay buffer up to {}".format(replay_buffer.episodes_in_buffer))
            mac.init_human_model(human_models[human_index])

        if args.use_ewc and human_index > 0:
            logger.console_logger.info(
                "calculate fisher EWC"
            )
            mac.init_human_model(human_models[human_index-1])
            ewc_buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                                preprocess=preprocess,
                                device="cpu" if args.buffer_cpu_only else args.device)
            load_path1 = find_weight_path(os.path.join(args.local_results_path, "models", args.unique_token, str(human_index-1)))
            load_path2 = find_weight_path(args.checkpoint_path)
            load_path = load_path1 if human_index > args.CL_start_index else load_path2
            ewc_learner = le_REGISTRY[args.learner](mac, ewc_buffer.scheme, logger, args)
            ewc_learner.load_models(load_path)
            ewc_learner.cuda()
            ewc_t = 0
            runner.run(test_mode=False)
            while ewc_t < 10:
                episode_batch = runner.run(test_mode=True, keep_best_value=False)
                ewc_buffer.insert_episode_batch(episode_batch)
                if ewc_buffer.can_sample(args.batch_size):
                    episode_sample = ewc_buffer.sample_latest(args.batch_size)
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = process_batch(episode_sample[:, :max_ep_t], args)
                ewc_learner.train(episode_sample, runner.t_env, None, cal_fisher=True)
                ewc_t += 1
            mac.agent.keep_old_param()
            logger.console_logger.info(
                "keep old fisher and param for EWC"
            )
            mac.init_human_model(human_models[human_index])

        while runner.t_env <= args.t_max * (human_index + 1):

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

            # Run for a whole episode at a time
            episode_batch = runner.run(test_mode=False)
            buffer.insert_episode_batch(episode_batch)
            off_buffer.insert_episode_batch(episode_batch)

            # assert args.batch_size == args.off_batch_size == args.batch_size_run
            if buffer.can_sample(args.batch_size) and off_buffer.can_sample(args.off_batch_size):
                uni_episode_sample = buffer.uni_sample(args.batch_size)
                off_episode_sample = off_buffer.uni_sample(args.off_batch_size)
                max_ep_t = max(uni_episode_sample.max_t_filled(), off_episode_sample.max_t_filled())
                uni_episode_sample = process_batch(uni_episode_sample[:, :max_ep_t], args)
                off_episode_sample = process_batch(off_episode_sample[:, :max_ep_t], args)
                learner.train_critic(uni_episode_sample, best_batch=off_episode_sample, log=running_log)
                    
                # train actor
                need_replay_BC =False
                if args.use_ER and replay_buffer.can_sample(args.batch_size * args.replay_ratio):
                    episode_sample = buffer.sample_latest(args.batch_size)
                    replay_buffer_sample = replay_buffer.uni_sample(int(args.batch_size * args.replay_ratio))
                    episode_sample.update(replay_buffer_sample.data.transition_data,
                        slice(int(args.batch_size * (1 - args.replay_ratio)), args.batch_size),
                        slice(0, replay_buffer_sample.max_seq_length),
                        mark_filled=False)
                    need_replay_BC = True
                else:
                    episode_sample = buffer.sample_latest(args.batch_size)
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = process_batch(episode_sample[:, :max_ep_t], args)
                learner.train(episode_sample, runner.t_env, running_log, need_replay_BC=need_replay_BC)

            # Execute test runs once in a while
            n_test_runs = max(1, args.test_nepisode // runner.batch_size)
            if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
                logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max * (human_index + 1)))
                logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                    time_left(last_time, last_test_T, runner.t_env, args.t_max * (human_index + 1)), time_str(time.time() - start_time)))
                last_time = time.time()
                last_test_T = runner.t_env
                for _ in range(n_test_runs):
                    runner.run(test_mode=True)

            if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == -1):
                model_save_time = runner.t_env
                save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(human_index), str(runner.t_env))
                os.makedirs(save_path, exist_ok=True)
                logger.console_logger.info("Saving models to {}".format(save_path))
                learner.save_models(save_path)
            
            if args.save_model and (runner.best_test_returns > best_test_returns_old 
                                    or (runner.t_env - best_old_time > 30000
                                        and runner.test_wins == best_test_returns_old
                                        and best_test_returns_old != 0)):
                logger.console_logger.info('best_test_return_mean: {}'.format(runner.best_test_returns))
                save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(human_index), 'best_' + str(runner.t_env))
                os.makedirs(save_path, exist_ok=True)
                logger.console_logger.info("Saving models to {}".format(save_path))
                learner.save_models(save_path)
                best_test_returns_old = runner.best_test_returns
                best_old_time = runner.t_env

            if (runner.t_env - last_test_CL_T) >= args.t_max // args.test_all_num:
                save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(human_index), str(runner.t_env))
                os.makedirs(save_path, exist_ok=True)
                logger.console_logger.info("Saving models to {}".format(save_path))
                learner.save_models(save_path)
                test_CL(args, mac, logger, runner, human_models, human_index)
                last_test_CL_T = runner.t_env

            episode += args.batch_size_run
            if (runner.t_env - last_log_T) >= args.log_interval:
                logger.log_stat("episode", episode, runner.t_env)
                logger.print_recent_stats()
                last_log_T = runner.t_env

        # test performance
        logger.console_logger.info(
            "test performance after stage{}: ".format(human_index)
        )
        load_path = os.path.join(args.local_results_path, "models", args.unique_token, str(human_index))
        load_mac_weight(mac, load_path, logger)
        test_CL(args, mac, logger, runner, human_models, human_index, performance_task=performance_task)
            
        runner.close_env()
    logger.console_logger.info("Finished Training")

    # calculate metrics after whole training process
    res = [[performance_task[str(i)][str(j)] for j in range(partner_human_num)] \
            for i in range(partner_human_num)]
    backward_transfer = [0] * partner_human_num
    forward_transfer =  [0] * partner_human_num
    average_score = 0
    for i in range(partner_human_num):
        average_score += float(res[partner_human_num-1][i])
        if i < partner_human_num - 1:
            backward_transfer[i] = float(res[partner_human_num-1][i]) - float(res[i][i])
        if i > 0:
            forward_transfer[i] = float(res[i-1][i])
    average_score /= partner_human_num
    backward_transfer_mean = sum(backward_transfer) / (partner_human_num-1)
    forward_transfer_mean = sum(forward_transfer) / (partner_human_num-1)
    logger.console_logger.info('average_score: {}'.format(average_score))
    logger.console_logger.info('backward_transfer_mean: {}'.format(backward_transfer_mean))
    logger.console_logger.info('forward_transfer_mean: {}'.format(forward_transfer_mean))
    
def args_sanity_check(config, _log):

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


def find_weight_path(checkpoint_path):
    
    if checkpoint_path == "":
        return None
    
    if not os.path.exists(checkpoint_path):
        return None

    timesteps = []
    timestep_to_load = 0

    best_checkpoint_path = None
    cur_best_step = 0

    for name in os.listdir(checkpoint_path):
        full_name = os.path.join(checkpoint_path, name)

        if os.path.isdir(full_name) and name.isdigit():
            timesteps.append(int(name))
        elif os.path.isdir(full_name) and 'best_' in name:
            if int(name.replace('best_', '')) > cur_best_step:
                cur_best_step = int(name.replace('best_', ''))
                best_checkpoint_path = full_name
        elif os.path.isdir(full_name) and name == 'best':
            best_checkpoint_path = full_name
            break

    if best_checkpoint_path is not None:
        model_path = best_checkpoint_path
    else:
        timestep_to_load = max(timesteps)
        model_path = os.path.join(checkpoint_path, str(timestep_to_load))
    
    return model_path

def load_mac_weight(mac, checkpoint_path, logger):
    if checkpoint_path == "":
        return 

    if not os.path.isdir(checkpoint_path):
        logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(checkpoint_path))
        return
    
    model_path = find_weight_path(checkpoint_path)

    logger.console_logger.info("Loading model from {}".format(model_path))
    mac.load_models(model_path)

def test_CL(args, mac, logger, runner, human_models, cur_human_index, performance_task=None):
    
    n_test_runs = max(1, args.test_nepisode // runner.batch_size)
    total_test_wins = []
    for j in range(0, len(human_models)):
        mac.init_human_model(human_models[j])
        if args.use_Concord:
            logger.console_logger.info("Set task_id = {} for hypernet".format(j))
            mac.set_task_id(j)
        test_wins = []
        for _ in range(n_test_runs):
            runner.run(test_mode=True, keep_best_value=False)
            test_wins.append(runner.test_wins)
        logger.console_logger.info(
            "task{} : {}\t".format(j, np.mean(test_wins))
        )
        total_test_wins.append(np.mean(test_wins))

        if performance_task is None:
            continue
        if str(cur_human_index) not in performance_task.keys():
            performance_task[str(cur_human_index)] = {str(j): np.mean(test_wins)}
        else:
            performance_task[str(cur_human_index)][str(j)] = np.mean(test_wins)

    mac.init_human_model(human_models[cur_human_index])
    if args.use_Concord:
        logger.console_logger.info("Set task_id = {} for hypernet".format(cur_human_index))
        mac.set_task_id(cur_human_index)
