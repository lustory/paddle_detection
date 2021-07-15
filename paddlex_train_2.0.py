import paddlex as pdx
from paddlex import transforms as T


# 定义训练和验证时的transforms
# API说明：https://github.com/PaddlePaddle/PaddleX/blob/release/2.0-rc/paddlex/cv/transforms/operators.py
train_transforms = T.Compose([
    # T.MixupImage(mixup_epoch=250), 
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
        target_size=640, interp='CUBIC'), T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 定义训练和验证所用的数据集
# API说明：https://github.com/PaddlePaddle/PaddleX/blob/release/2.0-rc/paddlex/cv/datasets/voc.py#L29
train_dataset = pdx.datasets.VOCDetection(
    data_dir='/home/lulei/research/data/ansen/blue_530',
    file_list='train.txt',
    label_list='label_list.txt',
    transforms=train_transforms,
    shuffle=True)

eval_dataset = pdx.datasets.VOCDetection(
    data_dir='/home/lulei/research/data/ansen/blue_530',
    file_list='valid.txt',
    label_list='label_list.txt',
    transforms=eval_transforms,
    shuffle=False)

# 初始化模型，并进行训练
# 可使用VisualDL查看训练指标，参考https://github.com/PaddlePaddle/PaddleX/tree/release/2.0-rc/tutorials/train#visualdl可视化训练指标
num_classes = len(train_dataset.labels)
model = pdx.models.PPYOLOv2(
    num_classes=num_classes, 
    backbone='ResNet50_vd_dcn')

# API说明：https://github.com/PaddlePaddle/PaddleX/blob/release/2.0-rc/paddlex/cv/models/detector.py#L154
# 各参数介绍与调整说明：https://paddlex.readthedocs.io/zh_CN/develop/appendix/parameters.html
model.train(
    num_epochs=365,
    train_dataset=train_dataset,
    train_batch_size=15,
    eval_dataset=eval_dataset,
    learning_rate=0.005 / 12,
    warmup_steps=1000,
    warmup_start_lr=0.0,
    lr_decay_epochs=[243],
    save_interval_epochs=5,
    save_dir='output/ppyolov2_r50vd_dcn',
    use_vdl=True)