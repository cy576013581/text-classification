
python run_classifier.py --task_name=cus --do_train=true --do_eval=true --data_dir=训练文件地址 --vocab_file=模型地址/vocab.txt --bert_config_file=模型地址/bert_config.json --init_checkpoint=模型地址/bert_model.ckpt --max_seq_length=128 --train_batch_size=32 --learning_rate=2e-5 --num_train_epochs=3.0 --output_dir=./output
