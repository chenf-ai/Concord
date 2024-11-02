export CUDA_VISIBLE_DEVICES=2
nohup \
python src/main.py --config=offpg_CL --is_CL=True \
--test_nepisode=64 \
--test_interval=20000 \
--log_interval=20000 \
--runner_log_interval=20000 \
--learner_log_interval=20000 \
--test_all_num=5 \
--t_max=5000000 \
--mac=non_shared_mac_hyper \
--agent=Concord \
--use_Concord=True \
--reg_beta=2.0 \
--checkpoint_path='' \
--CL_start_index=0 \
--human_models_dir='./human_models/random1_human_model_dir.txt' \
--env-config=overcooked_old with env_args.map_name=random1 \
>> nohup_overcooked_random1_Concord.out 2>&1 &
sleep 20

# nohup \
# python src/main.py --config=offpg_CL --is_CL=True \
# --test_nepisode=64 \
# --test_interval=20000 \
# --log_interval=20000 \
# --runner_log_interval=20000 \
# --learner_log_interval=20000 \
# --test_all_num=5 \
# --t_max=10000000 \
# --mac=non_shared_mac_hyper \
# --agent=Concord \
# --use_Concord=True \
# --reg_beta=2.0 \
# --checkpoint_path='' \
# --CL_start_index=0 \
# --human_models_dir='./human_models/unident_s8_human_model_dir.txt' \
# --env-config=overcooked_old with env_args.map_name=unident_s8 \
# >> nohup_overcooked_unident_s8_Concord.out 2>&1 &
# sleep 20

# nohup \
# python src/main.py --config=offpg_CL --is_CL=True \
# --test_nepisode=64 \
# --test_interval=20000 \
# --log_interval=20000 \
# --runner_log_interval=20000 \
# --learner_log_interval=20000 \
# --test_all_num=5 \
# --t_max=7000000 \
# --mac=non_shared_mac_hyper \
# --agent=Concord \
# --use_Concord=True \
# --reg_beta=2.0 \
# --checkpoint_path='' \
# --CL_start_index=0 \
# --human_models_dir='./human_models/many_orders_human_model_dir.txt' \
# --env-config=overcooked_new with env_args.map_name=many_orders \
# >> nohup_overcooked_many_orders_Concord.out 2>&1 &
# sleep 20

# nohup \
# python src/main.py --config=offpg_CL --is_CL=True \
# --test_nepisode=64 \
# --test_interval=20000 \
# --log_interval=20000 \
# --runner_log_interval=20000 \
# --learner_log_interval=20000 \
# --test_all_num=5 \
# --t_max=7000000 \
# --mac=non_shared_mac_hyper \
# --agent=Concord \
# --use_Concord=True \
# --reg_beta=2.0 \
# --checkpoint_path='' \
# --CL_start_index=0 \
# --human_models_dir='./human_models/unident_open_human_model_dir.txt' \
# --env-config=overcooked_new with env_args.map_name=unident_open \
# >> nohup_overcooked_unident_open_Concord.out 2>&1 &
# sleep 20