# 环境变量配置，用于控制是否使用GPU
# 说明文档：https://paddlex.readthedocs.io/zh_CN/develop/appendix/parameters.html#gpu
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from paddlex.det import transforms
import paddlex as pdx



# 定义训练和验证时的transforms
# API说明 https://paddlex.readthedocs.io/zh_CN/develop/apis/transforms/det_transforms.html
train_transforms = transforms.Compose([
    # transforms.MixupImage(mixup_epoch=250),
    transforms.RandomDistort(brightness_range=0.5, 
                            brightness_prob=0.5, 
                            contrast_range=0.5, 
                            contrast_prob=0.5, 
                            saturation_range=0.5, 
                            saturation_prob=0.5, 
                            hue_range=18, 
                            hue_prob=0.5),
    # 对输入图像进行放大，ratio：最大放大尺寸；prob：进行放大的概率；fill_value：填充值；
    # 该功能在resize和ResizeByShort之前使用。
    transforms.RandomExpand(ratio=4., prob=0.5, fill_value=[123.675, 116.28, 103.53]), 
    transforms.RandomCrop(), 
    transforms.ResizeByShort(short_size=[416,512,608], max_size=800),
    transforms.Resize(target_size=608, interp='RANDOM'), 
    transforms.RandomHorizontalFlip(prob=0.5),
    transforms.RandomHorizontalFlip(),
    # 归一化，使用默认参数
    transforms.Normalize(),
])

eval_transforms = transforms.Compose([
    transforms.Resize(target_size=608, interp='CUBIC'), 
    transforms.Normalize()
])

# 定义训练和验证所用的数据集
# API说明：https://paddlex.readthedocs.io/zh_CN/develop/apis/datasets.html#paddlex-datasets-vocdetection
train_dataset = pdx.datasets.VOCDetection(
    data_dir=r'D:\paddlex_workspace\datasets\D0005',
    file_list=r'D:\paddlex_workspace\datasets\D0005\train_list.txt',
    label_list=r'D:\paddlex_workspace\datasets\D0005\labels.txt',
    transforms=train_transforms,
    shuffle=True)
eval_dataset = pdx.datasets.VOCDetection(
    data_dir=r'D:\paddlex_workspace\datasets\D0005',
    file_list=r'D:\paddlex_workspace\datasets\D0005\val_list.txt',
    label_list=r'D:\paddlex_workspace\datasets\D0005\labels.txt',
    transforms=eval_transforms)

# 初始化模型，并进行训练
# 可使用VisualDL查看训练指标，参考https://paddlex.readthedocs.io/zh_CN/develop/train/visualdl.html
num_classes = len(train_dataset.labels)

# API说明: https://paddlex.readthedocs.io/zh_CN/develop/apis/models/detection.html#paddlex-det-ppyolo
model = pdx.det.PPYOLO(num_classes=num_classes)

# API说明: https://paddlex.readthedocs.io/zh_CN/develop/apis/models/detection.html#train
# 各参数介绍与调整说明：https://paddlex.readthedocs.io/zh_CN/develop/appendix/parameters.html
model.train(
    num_epochs=270,
    train_dataset=train_dataset,
    train_batch_size=6,
    eval_dataset=eval_dataset,
    learning_rate=0.000125,
    lr_decay_epochs=[210, 240],
    save_dir=r'.\output\xunluo_ppyolo_95train_270epochs',
    use_vdl=True,
    log_interval_steps=10,
    save_interval_epochs=5)