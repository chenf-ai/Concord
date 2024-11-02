# export CUDA_VISIBLE_DEVICES=1
# nohup \
# python src/main.py --config=offpg_CL --is_recognizer=True \
# --batch_size_run=32 \
# --test_nepisode=512 \
# --mac=non_shared_mac_hyper \
# --agent=hypernet_handle \
# --use_hypernet=True \
# --checkpoint_path='./recognizer_data/train_result/temp_3m_train2/2/2000' \
# --recognizer_experience_num=1 \
# --recognizer_gamma=0.92 \
# --rec_max_t=17 \
# --recognizer_human_index=2 \
# --human_num=1 \
# --evaluate=True \
# --human_models_dir='./human_models/3m_human_model_dir.txt' \
# --env-config=sc2 with env_args.map_name=3m \
# >> nohup_sc2_3m_recognizer_hyper2_index2_test.out 2>&1 &
# sleep 20

# export CUDA_VISIBLE_DEVICES=0
# nohup \
# python src/main.py --config=offpg_CL --train_recognizer=True \
# --batch_size=64 \
# --batch_size_run=32 \
# --test_nepisode=512 \
# --mac=non_shared_mac_hyper \
# --agent=hypernet_handle \
# --use_hypernet=True \
# --checkpoint_path='./recognizer_data/train_result/temp_2z1s_train2/2/3600' \
# --recognizer_experience_num=1 \
# --recognizer_gamma=0.92 \
# --rec_max_t=38 \
# --recognizer_human_index=2 \
# --human_num=1 \
# --evaluate=True \
# --human_models_dir='./human_models/2z1s_human_model_dir2.txt' \
# --env-config=sc2 with env_args.map_name=2z1s \
# >> nohup_sc2_2z1s_recognizer_hyper2_index2_test.out 2>&1 &
# sleep 20

export CUDA_VISIBLE_DEVICES=0
nohup \
python src/main.py --config=offpg_CL --train_recognizer=True \
--batch_size=64 \
--batch_size_run=32 \
--test_nepisode=512 \
--mac=non_shared_mac_hyper \
--agent=hypernet_handle \
--use_hypernet=True \
--checkpoint_path='./recognizer_data/train_result/4m_ex_train2/5/2600' \
--recognizer_experience_num=1 \
--recognizer_gamma=0.92 \
--rec_max_t=20 \
--recognizer_human_index=5 \
--human_num=2 \
--evaluate=True \
--human_models_dir='./human_models/4m_human_model_dir.txt' \
--env-config=sc2 with env_args.map_name=4m \
>> nohup_sc2_4m_recognizer_hyper2_index5_test.out 2>&1 &
sleep 20