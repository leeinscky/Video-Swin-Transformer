
# 生成文件列表  https://mmaction2.readthedocs.io/zh_CN/latest/data_preparation.html#id6
    cd /Users/lizejian/cambridge/mphil_project/learn/Video-Swin-Transformer
    # 生成训练集文件列表
    python3 ./tools/data/build_file_list.py \
        kinetics400 \
        ./data/kinetics400/rawframes_train \
        --format rawframes \
        --subset train
    # 生成验证集文件列表
    python3 ./tools/data/build_file_list.py \
        kinetics400 \
        ./data/kinetics400/rawframes_val \
        --format rawframes \
        --subset val
    # 生成测试集文件列表
    python3 ./tools/data/build_file_list.py \
        kinetics400 \
        ./data/kinetics400/rawframes_test \
        --format rawframes \
        --subset test 

# 使用预训练模型进行推理
    # 推理 Inference
        # single-gpu testing 
            python tools/test.py <CONFIG_FILE> <CHECKPOINT_FILE> --eval top_k_accuracy
            注意：the following arguments are required: config, checkpoint 即这两个参数是必须的
            例如： 
            python tools/test.py \
                configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py \
                checkpoints/swin_tiny_patch244_window877_kinetics400_1k.pth \
                --eval top_k_accuracy

        # multi-gpu testing
            bash tools/dist_test.sh <CONFIG_FILE> <CHECKPOINT_FILE> <GPU_NUM> --eval top_k_accuracy

    # 训练 Training
        For example, to train a Swin-T model for Kinetics-400 dataset with 8 gpus, run:
        bash tools/dist_train.sh configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py 8 --cfg-options model.backbone.pretrained=<PRETRAIN_MODEL>
