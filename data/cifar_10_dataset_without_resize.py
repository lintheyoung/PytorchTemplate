# cifar10数据集的处理
# 数据集的获取
# 数据集的数据增强

import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms

''' 自定义数据集的书写格式
class CustomDataset(data.Dataset):# 需要继承data.Dataset
    def __init__(self):
        # TODO
        # 1. Initialize file path or list of file names.
        pass
    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        #这里需要注意的是，第一步：read one data，是一个data
        pass
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return 0
'''


class Cifar10DatasetWithoutResize(Dataset):
    def __init__(self, args, is_train):
        # usually we need args rather than single datalist to init the dataset
        # 修正
        super(Cifar10DatasetWithoutResize, self).__init__()

        # 图像的获取
        # 训练集和测试集的获取，要是is_train为true就下载的是训练集，要是为false下载的就是测试集
        self.dataset = torchvision.datasets.CIFAR10(root='./data', train=is_train,
                                                    download=True, transform=self.dataset_transform(is_train))

    # 对于dataloader这两个函数是最主要的
    def __len__(self):
        return self.dataset.__len__()

    # 主要函数
    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)

    # 数据增强
    def dataset_transform(self, is_train):
        train_transforms = transforms.Compose([transforms.ToTensor()])

        test_transforms = transforms.Compose([transforms.ToTensor()])

        if is_train:
            return train_transforms
        else:
            return test_transforms

    '''
    def datasize(self):
        print('Training Set:\n')
        for images, labels in train_loader:
            print('Image batch dimensions:', images.size())
            print('Image label dimensions:', labels.size())
            print(labels[:10])
            break

        # Checking the dataset
        print('\nValidation Set:')
        for images, labels in valid_loader:
            print('Image batch dimensions:', images.size())
            print('Image label dimensions:', labels.size())
            print(labels[:10])
            break

        # Checking the dataset
        print('\nTesting Set:')
        for images, labels in train_loader:
            print('Image batch dimensions:', images.size())
            print('Image label dimensions:', labels.size())
            print(labels[:10])
            break
    '''
