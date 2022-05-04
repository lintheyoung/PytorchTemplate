# from model.base.fcn import CustomFcn
# from model.best.fcn import DeepLabv3Fcn
# from model.better.fcn import Resnet101Fcn
# from model.sota.fcn import LightFcn
from model.alexnet.alexnet_model import AlexNet
from model.lenet5.lenet_5_model import LeNet5
from model.vggnet.vggnet16 import VGG16
from model.densenet.densenet_model import DenseNet121
from model.resnet.resnet34_model import resnet34
from model.resnet.resnet101_model import resnet101, resnet50
from model.cotnet.cotnet_model import cotnet50
import torch.nn as nn


def select_model(args):
    type2model = {
        'alexnet_fcn': AlexNet(args),
        'lenet5_fcn': LeNet5(args),
        'vggnet16_fcn': VGG16(args),
        'densenet121_fcn': DenseNet121(num_classes=args.classes_num, grayscale=False),
        'resnet34_fcn': resnet34(num_classes=args.classes_num),
        'resnet101_fcn': resnet101(num_classes=args.classes_num),
        'resnet50_fcn': resnet50(num_classes=args.classes_num),
        'cotnet50_fcn': cotnet50(num_classes=args.classes_num)
    }
    model = type2model[args.model_type]
    return model


def equip_multi_gpu(model, args):
    model = nn.DataParallel(model, device_ids=args.gpus)
    return model
