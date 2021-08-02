import paddlex as pdx
from paddlex import transforms as T


# 定义训练和验证时的transforms
# API说明：https://github.com/PaddlePaddle/PaddleX/blob/release/2.0-rc/paddlex/cv/transforms/operators.py
train_transforms = T.Compose([
    T.MixupImage(mixup_epoch=-1), 
    T.RandomDistort(brightness_range=0.5, \
                    brightness_prob=0.5, \
                    contrast_range=0.5, \
                    contrast_prob=0.5, \
                    saturation_range=0.5, \
                    saturation_prob=0.5, \
                    hue_range=18, \
                    hue_prob=0.5, \
                    random_apply=True, \
                    count=3, \
                    shuffle_channel=False),
    T.RandomExpand(im_padding_value=[123.675, 116.28, 103.53]), 
    T.RandomCrop(),
    # T.RandomHorizontalFlip(), 
    T.BatchRandomResize(
        target_sizes=[
            544, 576, 608, 640, 
        ],
        interp='RANDOM'), 
    T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_transforms = T.Compose([
    T.Resize(
        target_size=544, interp='CUBIC'), T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 定义训练和验证所用的数据集
# API说明：https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/apis/datasets.md
train_dataset = pdx.datasets.VOCDetection(
    data_dir='/home/lulei/research/data/ansen/orange_1292',
    file_list='/home/lulei/research/data/ansen/orange_1292/train_list.txt',
    label_list='/home/lulei/research/data/ansen/orange_1292/labels.txt',
    transforms=train_transforms,
    shuffle=True)

eval_dataset = pdx.datasets.VOCDetection(
    data_dir='/home/lulei/research/data/ansen/orange_1292',
    file_list='/home/lulei/research/data/ansen/orange_1292/val_list.txt',
    label_list='/home/lulei/research/data/ansen/orange_1292/labels.txt',
    transforms=eval_transforms,
    shuffle=False)

# 初始化模型，并进行训练
# 可使用VisualDL查看训练指标，参考https://github.com/PaddlePaddle/PaddleX/tree/release/2.0-rc/tutorials/train#visualdl可视化训练指标
model = pdx.det.PPYOLOv2(
            num_classes=len(train_dataset.labels), 
            backbone='ResNet50_vd_dcn',
            label_smooth=True,
            # 检测框的置信度得分阈值，置信度低于阈值的框应该被忽略。默认值为0.01.
            nms_score_threshold=0.01,
            # 进行NMS时，根据置信度保留的最大检测框数。如果为-1则全部保留。默认为-1.
            nms_topk=-1,
            # 进行NMS后，每个图像要保留的总检测框数。默认为100.
            nms_keep_topk=50,
            # 进行NMS时，用于剔除检测框IOU的阈值。默认为0.45。
            nms_iou_threshold=0.75)

# API说明：https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/apis/models/detection.md
model.train(
    num_epochs=170,
    train_dataset=train_dataset,
    train_batch_size=25,
    eval_dataset=eval_dataset,
    learning_rate=0.005 / 1,
    warmup_steps=1000,
    warmup_start_lr=0.0,
    lr_decay_epochs=[105,135,150],
    save_interval_epochs=5,
    save_dir='output/ppyolov2_r50vd_dcn',
    use_vdl=True
    )