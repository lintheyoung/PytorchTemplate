import torch
import torch.nn.functional as F
from torch.autograd import Variable
from data.data_entry import select_eval_loader
from model.model_entry import select_model
from options import prepare_test_args
from utils.logger import Recoder
from utils.tools_train import compute_accuracy
from utils.tools_train import compute_epoch_loss
from utils.tools_train import compute_confusion_matrix
from utils.tools_train import plot_confusion_matrix
import numpy as np
import cv2
import os

from utils.viz import label2rgb

class Evaluator:
    def __init__(self):
        args = prepare_test_args()
        print(args)
        self.args = args
        self.model = select_model(args)
        self.model = torch.nn.DataParallel(self.model)
        self.model.load_state_dict(torch.load(args.load_model_path))
        self.model.eval()
        self.val_loader = select_eval_loader(args)
        self.recoder = Recoder()

    def eval(self):
        eval_acc = compute_accuracy(self.model, self.val_loader, 'cuda:0')
        print("eval accuracy = ", eval_acc)

        eval_loss = compute_epoch_loss(self.model, self.val_loader, 'cuda:0')
        print("eval_loss = ", eval_loss)

        compute_matrix = compute_confusion_matrix(self.model, self.val_loader, 'cuda:0')
        print("compute_matrix = ", compute_matrix)

        class_dict = {0: 'airplane',
                      1: 'automobile',
                      2: 'bird',
                      3: 'cat',
                      4: 'deer',
                      5: 'dog',
                      6: 'frog',
                      7: 'horse',
                      8: 'ship',
                      9: 'truck'}

        plot_confusion_matrix(compute_matrix, class_names=class_dict)

        # for i, data in enumerate(self.val_loader):
        #     img, pred, label = self.step(data)
        #     # accuracy
        #     eval_acc = compute_accuracy(self.model, self.val_loader, 'cuda:0')
        #     print("eval accuracy = ", eval_acc)
        #     # loss
        #     metrics = self.compute_metrics(pred, label)
        #
        #     for key in metrics.keys():
        #         self.recoder.record(key, metrics[key])
        #     # if i % self.args.viz_freq:
        #     #     self.viz_per_batch(img, pred, label, i)
        #
        # metrics = self.recoder.summary()
        # print(metrics)
        # result_txt_path = os.path.join(self.args.result_dir, 'result.txt')
        #
        # # write metrics to result dir,
        # # you can also use pandas or other methods for better stats
        # with open(result_txt_path, 'w') as fd:
        #     fd.write(str(metrics))

    def compute_metrics(self, pred, gt, is_train = False, loss_fn=None):
        if loss_fn is None:
            loss_fn = F.cross_entropy

        # you can call functions in metrics.py
        l1 = loss_fn(pred, gt)
        prefix = 'train/' if is_train else 'val/'
        metrics = {
            prefix + 'l1': l1
        }
        return metrics

    def viz_per_batch(self, img, pred, gt, step):
        # call functions in viz.py
        # here is an example about segmentation
        img_np = img[0].cpu().numpy().transpose((1, 2, 0))
        pred_np = label2rgb(pred[0].cpu().detach().numpy())
        gt_np = label2rgb(gt[0].cpu().numpy())
        viz = np.concatenate([img_np, pred_np, gt_np], axis=1)
        viz_path = os.path.join(self.args.result_dir, "%04d.jpg" % step)
        cv2.imwrite(viz_path, viz)
    
    def step(self, data):
        img, label = data
        # warp input
        img = Variable(img).cuda()
        label = Variable(label).cuda()

        # compute output
        pred = self.model(img)
        return img, pred, label


def eval_main():
    evaluator = Evaluator()
    evaluator.eval()


if __name__ == '__main__':
    eval_main()
