#!/bin/bash
#description: 启动BERT分类模型
echo '正在启动 BERT SERVICE ...'
# /data/emotion_service  需替换为你的部署文件存放目录
cd /data/emotion_service
sudo rm -rf tmp*
sudo rm -rf nohup.out

# bert预训练模型路径所在地址
export BERT_BASE_DIR=/data/vocab_file/chinese_L-12_H-768_A-12
# 微调后的模型路径所在地址
export TRAINED_CLASSIFIER=/data/bert/out_emotion_30w

nohup python3.7 start.py -model_dir $TRAINED_CLASSIFIER -bert_model_dir $BERT_BASE_DIR -model_pb_dir $TRAINED_CLASSIFIER -mode CLASS -max_seq_len 128 -http_port 8091  -port 5585 -port_out 5586 &
