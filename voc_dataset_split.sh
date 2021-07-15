#! /bin/bash

## 图片路径&标注路径
image_dir=JPEGImages
anno_dir=Annotations

## 生成类别标签
echo -e "0\n1\n2\n3\n4\n5\n6\n7\n8\n9\n-1" > label_list.txt

## 生成 train.txt、valid.txt和test.txt列表文件
# 检索所有jpg图片，打乱顺序后将名称存入all_image_list.txt
ls $image_dir/*.jpg | shuf > all_image_list.txt
# 
awk -F"/" '{print $2}' all_image_list.txt | awk -F".jpg" '{print $1}' \
| awk -F"\t" '{print "'$image_dir/'"$1".jpg '$anno_dir'/"$1".xml"}' > all_list.txt

image_num=$(ls -lR $image_dir | grep ".jpg" | wc -l)

echo $image_num

# # 训练集、验证集、测试集划分
# 头部作为训练
head -n 530 all_list.txt > train.txt
# 其余作为测试
tail -n 100 all_list.txt > valid.txt



## 删除不用文件
rm -rf all_image_list.txt all_list.txt