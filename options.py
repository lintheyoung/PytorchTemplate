import argparse
import os


def parse_common_args(parser):
    # 模型的选择
    parser.add_argument('--model_type', type=str, default='cotnet50_fcn', help='used in model_entry.py')
    # 数据集的选择
    parser.add_argument('--data_type', type=str, default='cifar_10_without_resize', help='used in data_entry.py')
    # 体现在名字里面
    parser.add_argument('--save_prefix', type=str, default='pref', help='some comment for model or test result dir')
    # 选择pth文件，checkpoints/alexnet_fcn_pref/alexnet_best.pth
    parser.add_argument('--load_model_path', type=str, default='checkpoints/cotnet50_fcn_pref/19_000000.pth',
                        help='model path for pretrain or test')
    # 是否采用有缺损的pth文件（可能有些层数用不了）
    parser.add_argument('--load_not_strict', type=bool, default=False, help='allow to load only common state dicts')
    # 不是很懂
    parser.add_argument('--val_list', type=str, default='/data/dataset1/list/base/val.txt',
                        help='val list in train, test list path in test')
    # 选择GPU
    parser.add_argument('--gpus', nargs='+', type=int)
    # 随机种子
    parser.add_argument('--seed', type=int, default=1234)
    return parser


def parse_train_args(parser):
    parser = parse_common_args(parser)
    parser.add_argument('--classes_num', type=int, default=10, help='num of object to identify')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum for sgd, alpha parameter for adam')
    parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                        help='beta parameters for adam')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay')
    parser.add_argument('--model_dir', type=str, default='', help='leave blank, auto generated')
    parser.add_argument('--train_list', type=str, default='/data/dataset1/list/base/train.txt')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--val_acc_freq', type=int, default=5)
    return parser


def parse_test_args(parser):
    parser = parse_common_args(parser)
    parser.add_argument('--classes_num', type=int, default=10, help='num of object to identify')
    parser.add_argument('--save_viz', action='store_true', help='save viz result in eval or not')
    parser.add_argument('--result_dir', type=str, default='', help='leave blank, auto generated')
    parser.add_argument('--viz_freq', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    return parser


def get_train_args():
    parser = argparse.ArgumentParser()
    parser = parse_train_args(parser)
    args = parser.parse_args()
    return args


def get_test_args():
    parser = argparse.ArgumentParser()
    parser = parse_test_args(parser)
    args = parser.parse_args()
    return args


def get_train_model_dir(args):
    model_dir = os.path.join('checkpoints', args.model_type + '_' + args.save_prefix)
    if not os.path.exists(model_dir):
        os.system('mkdir -p ' + model_dir)
    args.model_dir = model_dir


def get_test_result_dir(args):
    print("here")
    ext = os.path.basename(args.load_model_path).split('.')[-1]
    print("path = ", ext)
    model_dir = args.load_model_path.replace(ext, '')
    val_info = os.path.basename(os.path.dirname(args.val_list)) + '_' + os.path.basename(args.val_list.replace('.txt', ''))
    result_dir = os.path.join(model_dir, val_info + '_' + args.save_prefix)
    if not os.path.exists(result_dir):
        os.system('mkdir -p ' + result_dir)
    args.result_dir = result_dir


def save_args(args, save_dir):
    args_path = os.path.join(save_dir, 'args.txt')
    with open(args_path, 'w') as fd:
        fd.write(str(args).replace(', ', ',\n'))


def prepare_train_args():
    args = get_train_args()
    get_train_model_dir(args)
    save_args(args, args.model_dir)
    return args


def prepare_test_args():
    args = get_test_args()
    get_test_result_dir(args)
    save_args(args, args.result_dir)
    return args


if __name__ == '__main__':
    train_args = get_train_args()
    test_args = get_test_args()
