export CUDA_VISIBLE_DEVICES=3
nohup \
python src/main.py --config=offpg_CL --is_recognizer=True \
--batch_size=32 \
--batch_size_run=32 \
--test_nepisode=32 \
--mac=non_shared_mac_hyper \
--agent=hypernet_handle \
--use_hypernet=True \
--checkpoint_path='./results/recognizer_data/train_result/unident_open_train/' \
--recognizer_experience_num=1 \
--recognizer_gamma=0.994 \
--recognizer_human_index=11 \
--recognizer_json_dir='./results/recognizer_data/train_result/unident_open_ex_train.json' \
--human_models_dir='./human_models/unident_open_human_model_dir.txt' \
--env-config=overcooked_new with env_args.map_name=unident_open \
>> nohup_overcooked_unident_open_recognizer_index11.out 2>&1 &
sleep 20