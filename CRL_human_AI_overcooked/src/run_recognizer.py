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
import numpy
import json

from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer, Best_experience_Buffer
from components.transforms import OneHot
from modules.agents.rnn_agent import Recognizer



def run_recognizer(_run, _config, _log):

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

    rec_buffer = ReplayBuffer(scheme, groups, args.batch_size * 2 * 12, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)
    test_rec_buffer = ReplayBuffer(scheme, groups, args.batch_size * 2 * 12, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](rec_buffer.scheme, groups, args)
    mac.cuda()
    
    with open(args.human_models_dir, 'r') as f:
        lines = f.readlines()
        human_models = [line.replace('\n', '') for line in lines]

    # here here here here here here !
    partner_human_num = len(human_models)

    # Init runner and Give runner the mac
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    recognizer = Recognizer(args.vecobs_dim + 6 + 2, 64).cuda()
    
    loss_fn = th.nn.MSELoss(reduction='mean')
    optimiser = th.optim.Adam(params=recognizer.parameters(), lr=0.001)

    # hyper-network models
    # agent_load_path = []
    
    experience_num = args.recognizer_experience_num
    if not args.evaluate:
        assert args.batch_size == args.batch_size_run
    else:
        assert True # args.batch_size_run == 1
    
    # gernerate experience
    if not args.evaluate:
        
        if not (args.recognizer_json_dir != '' and os.path.exists(args.recognizer_json_dir)):

            for human_index in range(0, args.recognizer_human_index + 1):
                
                mac.init_human_model(human_models[human_index])
                
                # train data
                for _ in range(0, experience_num):
                
                    if args.use_Concord:
                        logger.console_logger.info("Set task_id = {} for hypernet".format(human_index))
                        mac.set_task_id(human_index)
                    load_mac_weight(mac, agent_load_path[human_index], logger)
                    episode_batch = runner.run(test_mode=True, keep_best_value=False)
                    rec_buffer.insert_episode_batch(episode_batch[:16])
                    if args.use_Concord:
                        logger.console_logger.info("Set task_id = {} for hypernet".format(0))
                        mac.set_task_id(0)
                    load_mac_weight(mac, agent_load_path[0], logger)
                    episode_batch = runner.run(test_mode=True, keep_best_value=False)
                    rec_buffer.insert_episode_batch(episode_batch[:16])
                logger.console_logger.info("generate train episodes of task{}".format(human_index))
        
                # test data
                if args.use_Concord:
                    logger.console_logger.info("Set task_id = {} for hypernet".format(human_index))
                    mac.set_task_id(human_index)
                load_mac_weight(mac, agent_load_path[human_index], logger)
                episode_batch = runner.run(test_mode=True, keep_best_value=False)
                test_rec_buffer.insert_episode_batch(episode_batch[:16])
                if args.use_Concord:
                    logger.console_logger.info("Set task_id = {} for hypernet".format(0))
                    mac.set_task_id(0)
                load_mac_weight(mac, agent_load_path[0], logger)
                episode_batch = runner.run(test_mode=True, keep_best_value=False)
                test_rec_buffer.insert_episode_batch(episode_batch[:16])
                logger.console_logger.info("generate test episodes of task{}".format(human_index))
                
            if args.recognizer_json_dir != '':
                buffer_to_json(rec_buffer, args, logger, args.recognizer_json_dir)
                buffer_to_json(test_rec_buffer, args, logger, args.recognizer_json_dir.replace('train', 'test'))
    else:

        for human_index in range(0, partner_human_num):
            mac.init_human_model(human_models[human_index])
            sample_ID_list = [args.recognizer_human_index, 0]
            for rec_sample_num in range(0, experience_num):
                sample_ID = sample_ID_list[rec_sample_num % len(sample_ID_list)]
                if args.use_Concord:
                    logger.console_logger.info("Set task_id = {} for hypernet".format(sample_ID))
                    mac.set_task_id(sample_ID)
                load_mac_weight(mac, agent_load_path[sample_ID], logger)
                episode_batch = runner.run(test_mode=True, keep_best_value=False)
                test_rec_buffer.insert_episode_batch(episode_batch[:1])
        logger.console_logger.info("generate test episodes of task{}".format(args.recognizer_human_index))  

    if args.evaluate:

        path = args.checkpoint_path
        recognizer.load_state_dict(th.load("{}/recognizer.th".format(path), map_location=lambda storage, loc: storage))
        recognizer.eval()
        if args.use_Concord:
            logger.console_logger.info("Set task_id = {} for hypernet".format(args.recognizer_human_index))
            mac.set_task_id(args.recognizer_human_index)
        load_mac_weight(mac, agent_load_path[args.recognizer_human_index], logger)

        total_rewards = []
        for human_index in range(0, partner_human_num):
            
            if experience_num > 0:
                # recognizer outputs embdding like few-shot
                episode_sample = test_rec_buffer.sample_by_index(human_index * experience_num, experience_num)
                max_ep_t = episode_sample.max_t_filled()
                batch = process_batch(episode_sample[:, :max_ep_t], args)
                bs = batch.batch_size
                max_t = batch.max_seq_length
            
                hidden_states = recognizer.init_hidden().unsqueeze(0).expand(experience_num, -1, -1)
                mean_predict_embedding = []
                for t_episode in range(0, max_t):
                    _, inputs_Human = build_inputs(batch, t_episode, bs)
                    inputs_Human = th.cat([inputs_Human[:, :args.vecobs_dim], inputs_Human[:, batch['obs'].shape[-1]:]], dim=1)
                    predict_embedding, hidden_states = recognizer(inputs_Human, hidden_states, args.recogizer_fuse)
                    if t_episode > 380:
                        mean_predict_embedding.append(predict_embedding)

                mean_predict_embedding = th.cat(mean_predict_embedding, dim=0).mean(0).detach().clone()
                target_embedding = mac.agent.hypernet.embed[human_index].detach().clone()
                print(mean_predict_embedding)
                print(target_embedding)

                mac.init_human_model(human_models[human_index])
                mac.set_given_embedding(mean_predict_embedding)
                
            else:
                # recognizer outputs embdding like zero-shot
                mac.init_human_model(human_models[human_index])
                mac.recognizer = recognizer

            n_test_runs = max(1, args.test_nepisode // runner.batch_size)
            for _ in range(0, n_test_runs):
                runner.run(test_mode=True)
            total_rewards.append(runner.test_rewards)
            logger.console_logger.info("test_rewards = {} when cooperating with Human{}".format(runner.test_rewards, human_index))
            mac.set_given_embedding(None)
        logger.console_logger.info("total_rewards = {}".format(np.mean(total_rewards)))

        json_file = './results/recognizer_data/test_result/temp/test_after_task{}'.format(args.recognizer_human_index)
        with open(json_file,'w') as f:
            json.dump(total_rewards, f)
        logger.console_logger.info("Keep result in {}".format(json_file))
        
        runner.close_env()
        return 

    runner.close_env()
    logger.console_logger.info("Begin Training")
    train_batch, test_batch = None, None
    if args.recognizer_json_dir != '' and os.path.exists(args.recognizer_json_dir):
        train_data_dir = args.recognizer_json_dir
        test_data_dir = args.recognizer_json_dir.replace('train', 'test')
        train_batch = json_to_batch(args, logger, train_data_dir)
        logger.console_logger.info("Successfully load train data from {}".format(train_data_dir))
        test_batch = json_to_batch(args, logger, test_data_dir)
        logger.console_logger.info("Successfully load test data from {}".format(test_data_dir))

    if args.use_Concord:
        logger.console_logger.info("Set task_id = {} for hypernet".format(args.recognizer_human_index))
        mac.set_task_id(args.recognizer_human_index)
    load_mac_weight(mac, agent_load_path[args.recognizer_human_index], logger)
    
    test_interval = 100
    save_model_interval = 200
    
    for t_train in range(10000):

        recognizer.train()
        if t_train % test_interval == 0:
            recognizer.eval()

        for human_index in range(0, args.recognizer_human_index + 1):
            
            if train_batch is not None and test_batch is not None:
                assert args.batch_size == 32
                if t_train % test_interval == 0:
                    batch = {k: test_batch[k][human_index * args.batch_size: human_index * args.batch_size + args.batch_size] \
                             for k in test_batch.keys()}
                else:
                    batch = {k: train_batch[k][human_index * args.batch_size: human_index * args.batch_size + args.batch_size] \
                             for k in train_batch.keys()}
                bs = 32
                max_t = 401
            else:
                if t_train % 100 == 0:
                    episode_sample = test_rec_buffer.sample_by_index(human_index * args.batch_size, args.batch_size)
                else:
                    episode_sample = rec_buffer.uni_sample_for_recognizer(human_index * args.batch_size * experience_num, 
                                                                        args.batch_size * experience_num, args.batch_size)
                max_ep_t = episode_sample.max_t_filled()
                batch = process_batch(episode_sample[:, :max_ep_t], args)
                bs = batch.batch_size
                max_t = batch.max_seq_length
            
            loss = None
            target_embedding = mac.agent.hypernet.embed[human_index].detach().clone()
            hidden_states = recognizer.init_hidden().unsqueeze(0).expand(args.batch_size, -1, -1)
            for t_episode in range(0, max_t):
                _, inputs_Human = build_inputs(batch, t_episode, bs)
                inputs_Human = th.cat([inputs_Human[:, :args.vecobs_dim], inputs_Human[:, batch['obs'].shape[-1]:]], dim=1)
                
                predict_embedding, hidden_states = recognizer(inputs_Human, hidden_states, args.recogizer_fuse)
                if loss is None:
                    loss = (args.recognizer_gamma**(max_t-t_episode-1)) * loss_fn(target_embedding, predict_embedding)
                else:
                    loss += (args.recognizer_gamma**(max_t-t_episode-1)) * loss_fn(target_embedding, predict_embedding)
                
            if t_train % test_interval != 0:
                optimiser.zero_grad()
                loss.backward()
                if not args.recognizer_rnn:
                    th.nn.utils.clip_grad_norm_(recognizer.parameters(), 5)
                optimiser.step()
            
            if t_train % 10 == 0:
                pre = 'test:' if t_train % test_interval == 0 else 'train:'
                logger.console_logger.info('{} {}'.format(pre, predict_embedding[0,:3].cpu().detach().numpy().tolist()))
                logger.console_logger.info('{} {}'.format(pre, target_embedding[:3].cpu().detach().numpy().tolist()))
                logger.console_logger.info('{} {} of {} loss: {}'.format(pre, human_index, t_train, loss))
            
        if t_train % save_model_interval == 0:
            save_path = os.path.join(args.checkpoint_path, str(args.recognizer_human_index), str(t_train))
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))
            th.save(recognizer.state_dict(), "{}/recognizer.th".format(save_path))
        
def args_sanity_check(config, _log):

    # set CUDA flags
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
    cur_best_step = 0
    
    # Go through all files in args.checkpoint_path
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

    logger.console_logger.info("Loading model from {}".format(model_path))
    mac.load_models(model_path)

def build_inputs(batch, t, bs):
    inputs = []
    inputs.append(batch["obs"][:, t])  # b1av
    inputs.append(batch["actions_onehot"][:, t])
    inputs.append(th.eye(2).cuda().unsqueeze(0).expand(bs, -1, -1))
    inputs = th.cat([x.reshape(bs, 2, -1) for x in inputs], dim=2)
    inputs = inputs.permute(1, 0, 2)
    return inputs[0], inputs[1]

def buffer_to_json(buffer, args, logger, json_file):
    
    episode_sample = buffer.sample_by_index(0, buffer.episodes_in_buffer)
    max_ep_t = episode_sample.max_t_filled()
    batch = process_batch(episode_sample[:, :max_ep_t], args)
    res_batch = {"obs": batch["obs"].cpu().numpy().tolist(), 
                 "actions_onehot": batch["actions_onehot"].cpu().numpy().tolist()}
    with open(json_file,'w') as f:
        json.dump(res_batch, f)
    logger.console_logger.info("Keep data in {}".format(json_file))

def json_to_batch(args, logger, json_file):

    with open(json_file, "r", encoding="utf-8") as f:
        batch = json.load(f)
    batch['obs'] = th.Tensor(batch['obs']).cuda()
    batch['actions_onehot'] = th.Tensor(batch['actions_onehot']).cuda()
        
    return batch
    
    
    
    


