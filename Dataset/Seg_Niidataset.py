import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from scipy.ndimage import zoom


class NiiDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.samples = []
        if os.path.exists(data_path):  # kit23/train
            for case in os.listdir(data_path):  # case_00001
                # for label in class_map.keys():
                class_path = os.path.join(data_path, case) # kit23/train/case_00001
                imaging_path = tumor_path = None
                for filename in os.listdir(class_path): # imaging.nii.gz tumor.nii.gz
                    if filename.endswith('imaging.nii.gz'):
                        imaging_path = os.path.join(class_path, filename)

                    if filename.endswith('tumor.nii.gz'):
                        tumor_path = os.path.join(class_path, filename)
                if imaging_path is None:
                    print(f'In {class_path} imaging.nii.gz is not Find!')
                    break
                if tumor_path is None:
                    print(f'In {class_path} tumor.nii.gz is not Find!')
                    break
                self.samples.append((imaging_path, tumor_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        imaging_path, tumor_path = self.samples[idx]
        nii_image = nib.load(imaging_path)
        nii_label = nib.load(tumor_path)
        data = nii_image.get_fdata()
        label = nii_label.get_fdata()
        # nii_image = nib.load(file_path)
        # data = nii_image.get_fdata()

        # 调整尺寸为128x128x128
        if data.shape != (128, 128, 128):
            data = self.resize_data(data, (128, 128, 128))
        if label.shape != (128,128,128):
            label = self.resize_data(label, (128, 128, 128))

        # 添加通道维度并归一化
        data = np.expand_dims(data, axis=0)
        data = data.astype(np.float32)
        label = np.expand_dims(label, axis=0)
        label = label.astype(np.float32)

        # 使用Z-score标准化
        mean = np.mean(data)
        std = np.std(data)
        if std != 0:
            data = (data - mean) / std
        mean = np.mean(label)
        std = np.std(label)
        if std != 0:
            label = (label - mean) / std

        return torch.from_numpy(data), torch.from_numpy(label)

    @staticmethod
    def resize_data(data, target_shape):
        if data.ndim == 4:
            data = data[..., 0]
        factors = [target_shape[i] / data.shape[i] for i in range(3)]
        return zoom(data, factors, order=1)


if __name__ == "__main__":

    from torch.utils.data import DataLoader
    # dataset
    train_dataset = NiiDataset('data/kit23_n/train')
    test_dataset = NiiDataset('data/kit23_n/test')

    # dataloader
    batch_size = 1
    nw = 4
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw)
    val_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=nw)

    print(f"训练集样本总数: {len(test_dataset)}")
    for imgs, labels in train_loader:
        print("测试批次信息：")
        print(f"图像数量: {imgs.shape[0]}")  # 应为 1
        print(f"单个图像形状: {imgs[0].shape}")  # [C, D, W, H]
        print(f"标签形状: {labels.shape}")  # [1, 1, D, W, H]
        break
    print(f"测试集样本总数: {len(test_dataset)}")
    for imgs, labels in val_loader:
        print("测试批次信息：")
        print(f"图像数量: {imgs.shape[0]}")  # 应为 1
        print(f"单个图像形状: {imgs[0].shape}")  # [C, D, W, H]
        print(f"标签形状: {labels.shape}")  # [1, 1, D, W, H]
        break
