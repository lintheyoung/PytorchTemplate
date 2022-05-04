# 文件 导入 函数
from data.cifar_10_dataset import Cifar10Dataset
from data.cifar_10_dataset_without_resize import Cifar10DatasetWithoutResize
from torch.utils.data import DataLoader

# 选择数据的类型
# 数据集的选择
def get_dataset_by_type(args, is_train=False):
    type2data = {
        'cifar_10': Cifar10Dataset(args, is_train),
        'cifar_10_without_resize': Cifar10DatasetWithoutResize(args, is_train)
    }
    dataset = type2data[args.data_type]
    return dataset


def select_train_loader(args):
    # usually we need loader in training, and dataset in eval/test
    # 用是否是用于训练的输出是否用于训练，为True时，就是用于训练的
    train_dataset = get_dataset_by_type(args, True)
    print('{} samples found in train'.format(len(train_dataset)))
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)
    return train_loader


def select_eval_loader(args):
    eval_dataset = get_dataset_by_type(args)
    print('{} samples found in val'.format(len(eval_dataset)))
    val_loader = DataLoader(eval_dataset, args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    return val_loader


