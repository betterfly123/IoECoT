# API
python main.py \
  --model_name "mixtral-8x7b-32768" \
  --data_name "MELD" \
  --data_path "/path/to/datasets_root" \
  --task_type "ERC" \
  --method_type "ioecot" \
  --default_emo 5

# Local
python main.py \
  --model_name "chatglm" \
  --local_model_path "/path/to/your/chatglm" \
  --data_name "MELD" \
  --data_path "/path/to/datasets_root" \
  --task_type "ERC" \
  --method_type "ioecot" \
  --default_emo 5
