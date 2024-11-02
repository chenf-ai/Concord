export CUDA_VISIBLE_DEVICES=2
nohup \
python src/main.py --config=offpg_CL --is_recognizer=True \
--batch_size=32 \
--batch_size_run=32 \
--test_nepisode=32 \
--mac=non_shared_mac_hyper \
--agent=Concord \
--use_Concord=True \
--checkpoint_path='./results/recognizer_data/train_result/3m_train' \
--recognizer_experience_num=1 \
--recognizer_gamma=0.92 \
--rec_max_t=17 \
--recognizer_human_index=2 \
--human_num=1 \
--recognizer_json_dir='./results/recognizer_data/train_result/3m_ex_train.json' \
--human_models_dir='./human_models/3m_human_model_dir.txt' \
--env-config=sc2 with env_args.map_name=3m \
>> nohup_sc2_3m_recognizer_hyper2_train_index2.out 2>&1 &
sleep 10