export CUDA_VISIBLE_DEVICES=2
nohup \
python src/main.py --config=offpg_CL --is_CL=True \
--test_nepisode=32 \
--test_interval=20000 \
--log_interval=20000 \
--runner_log_interval=20000 \
--learner_log_interval=20000 \
--t_max=5000000 \
--checkpoint_path='' \
--CL_start_index=0 \
--mac=non_shared_mac \
--agent=rnn \
--use_ER=True \
--replay_ratio=0.25 \
--reg_beta=1.0 \
--human_models_dir='./human_models/random1_human_model_dir.txt' \
--env-config=overcooked_old with env_args.map_name=random1 \
>> nohup_overcooked_random1_ER.out 2>&1 &
sleep 20

# nohup \
# python src/main.py --config=offpg_CL --is_CL=True \
# --test_nepisode=32 \
# --test_interval=20000 \
# --log_interval=20000 \
# --runner_log_interval=20000 \
# --learner_log_interval=20000 \
# --t_max=10000000 \
# --checkpoint_path='' \
# --CL_start_index=0 \
# --mac=non_shared_mac \
# --agent=rnn \
# --use_ER=True \
# --replay_ratio=0.25 \
# --reg_beta=1.0 \
# --human_models_dir='./human_models/unident_s8_human_model_dir.txt' \
# --env-config=overcooked_old with env_args.map_name=unident_s8 \
# >> nohup_overcooked_unident_s8_ER.out 2>&1 &
# sleep 20

# nohup \
# python src/main.py --config=offpg_CL --is_CL=True \
# --test_nepisode=32 \
# --test_interval=20000 \
# --log_interval=20000 \
# --runner_log_interval=20000 \
# --learner_log_interval=20000 \
# --t_max=7000000 \
# --checkpoint_path='' \
# --CL_start_index=0 \
# --mac=non_shared_mac \
# --agent=rnn \
# --use_ER=True \
# --replay_ratio=0.25 \
# --reg_beta=1.0 \
# --human_models_dir='./human_models/many_orders_human_model_dir.txt' \
# --env-config=overcooked_new with env_args.map_name=many_orders \
# >> nohup_overcooked_many_orders_ER.out 2>&1 &
# sleep 20

# nohup \
# python src/main.py --config=offpg_CL --is_CL=True \
# --test_nepisode=32 \
# --test_interval=20000 \
# --log_interval=20000 \
# --runner_log_interval=20000 \
# --learner_log_interval=20000 \
# --t_max=7000000 \
# --checkpoint_path='' \
# --CL_start_index=0 \
# --mac=non_shared_mac \
# --agent=rnn \
# --use_ER=True \
# --replay_ratio=0.25 \
# --reg_beta=1.0 \
# --human_models_dir='./human_models/unident_open_human_model_dir.txt' \
# --env-config=overcooked_new with env_args.map_name=unident_open \
# >> nohup_overcooked_unident_open_ER.out 2>&1 &
# sleep 20