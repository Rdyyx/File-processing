import os
import shutil
import random


def split_files(source_folder, new_folder_path, num_rate):
    """
    将 source_folder 中的文件随机分配到 train 和 test 目录下
    按比例 num_train:num_test 分配
    """
    # 获取所有源文件夹
    source_files = []
    if os.path.exists(source_folder):
        for file in os.listdir(source_folder):
            source_files.append(file)
            print(file)

        # 随机打乱文件顺序
        random.shuffle(source_files)

        # 计算分割点
        split = int(num_rate[0] / (num_rate[0] + num_rate[1]) * len(source_files))

        # 将文件分配到 train 和 test 目录下
        train_files = source_files[:split]
        test_files = source_files[split:]

        # 创建 train 和 test 目录
        new_train = new_folder_path + '/' + 'train'
        new_test = new_folder_path + '/' + 'test'
        os.makedirs(new_train, exist_ok=True)
        os.makedirs(new_test, exist_ok=True)

        # 移动文件
        for file in train_files:
            if os.path.exists(source_folder+'/'+file):
                new_file = new_train + '/' + file
                shutil.move(source_folder+'/'+file, new_file)

        for file in test_files:
            if os.path.exists(source_folder+'/'+file):
                new_file = new_test + '/' + file
                shutil.move(source_folder+'/'+file, new_file)


# 使用示例
source_folder = "/home/yuwenjing/data/kit23_new"  # 替换为你的实际路径
new_folder_path = "/home/yuwenjing/data/kit23_n"
split_files(source_folder, new_folder_path,(8,2))
