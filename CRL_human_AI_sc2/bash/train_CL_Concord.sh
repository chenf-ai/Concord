export CUDA_VISIBLE_DEVICES=1
nohup \
python src/main.py --config=offpg_CL --is_CL=True \
--test_nepisode=512 \
--batch_size_run=4 \
--save_model_interval=60000 \
--test_interval=60000 \
--log_interval=60000 \
--runner_log_interval=60000 \
--learner_log_interval=60000 \
--t_max=3000000 \
--mac=non_shared_mac_hyper \
--agent=Concord \
--use_Concord=True \
--reg_beta=2.0 \
--checkpoint_path='' \
--CL_start_index=0 \
--human_num=1 \
--hyper_lr=0.0001 \
--human_models_dir='./human_models/3m_human_model_dir.txt' \
--env-config=sc2 with env_args.map_name='3m' \
>> nohup_sc2_3m_Concord.out 2>&1 &
sleep 20

# export CUDA_VISIBLE_DEVICES=2
# nohup \
# python src/main.py --config=offpg_CL --is_CL=True \
# --test_nepisode=512 \
# --batch_size_run=4 \
# --save_model_interval=40000 \
# --test_interval=20000 \
# --log_interval=20000 \
# --runner_log_interval=20000 \
# --learner_log_interval=20000 \
# --t_max=1500000 \
# --mac=non_shared_mac_hyper \
# --agent=Concord \
# --use_Concord=True \
# --reg_beta=2.0 \
# --checkpoint_path='' \
# --CL_start_index=0 \
# --human_num=1 \
# --hyper_lr=0.0001 \
# --human_models_dir='./human_models/2z1s_human_model_dir.txt' \
# --env-config=sc2 with env_args.map_name='2z1s' \
# >> nohup_sc2_2z1s_Concord.out 2>&1 &
# sleep 20

# export CUDA_VISIBLE_DEVICES=3
# nohup \
# python src/main.py --config=offpg_CL --is_CL=True \
# --test_nepisode=512 \
# --batch_size_run=4 \
# --save_model_interval=40000 \
# --test_interval=40000 \
# --log_interval=40000 \
# --runner_log_interval=40000 \
# --learner_log_interval=40000 \
# --t_max=2000000 \
# --mac=non_shared_mac_hyper \
# --agent=Concord \
# --use_Concord=True \
# --reg_beta=2.0 \
# --checkpoint_path='' \
# --CL_start_index=0 \
# --human_num=2 \
# --hyper_lr=0.0001 \
# --human_models_dir='./human_models/4m_human_model_dir.txt' \
# --env-config=sc2 with env_args.map_name='4m' \
# >> nohup_sc2_4m_Concord.out 2>&1 &
# sleep 20