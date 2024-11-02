export CUDA_VISIBLE_DEVICES=1
nohup \
python src/main.py --config=offpg_CL --is_recognizer=True \
--batch_size=64 \
--batch_size_run=32 \
--test_nepisode=64 \
--mac=non_shared_mac_hyper \
--agent=hypernet_handle \
--use_hypernet=True \
--checkpoint_path='./results/recognizer_data/train_result/unident_open_train/3000' \
--recognizer_experience_num=2 \
--recognizer_human_index=11 \
--evaluate=True \
--human_models_dir='./human_models/unident_open_human_model_dir.txt' \
--env-config=overcooked_new with env_args.map_name=unident_open \
>> nohup_overcooked_unident_open_recognizer_index11_test.out 2>&1 &
sleep 20