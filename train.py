import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from torch.autograd import Variable

from data.data_entry import select_train_loader, select_eval_loader
from model.model_entry import select_model
from options import prepare_train_args
from utils.logger import Logger
from utils.tools_train import compute_accuracy
from utils.torch_utils import load_match_dict


class Trainer:
    def __init__(self):
        args = prepare_train_args()
        self.args = args
        torch.manual_seed(args.seed)
        self.logger = Logger(args)

        self.train_loader = select_train_loader(args)
        self.val_loader = select_eval_loader(args)

        self.model = select_model(args)
        self.model = torch.nn.DataParallel(self.model)

        if args.load_model_path != '':
            print("=> using pre-trained weights for DPSNet")
            self.model.load_state_dict(torch.load(args.load_model_path), strict=args.load_not_strict)


        self.optimizer = torch.optim.Adam(self.model.parameters(), self.args.lr,
                                          betas=(self.args.momentum, self.args.beta),
                                          weight_decay=self.args.weight_decay)

    def train(self):
        for epoch in range(self.args.epochs):
            # train for one epoch
            self.train_per_epoch(epoch)
            self.val_per_epoch(epoch)
            self.logger.save_curves(epoch)
            self.logger.save_check_point(self.model, epoch)

    # the train code need to be rewrite
    def train_per_epoch(self, epoch):
        # switch to train mode
        self.model.train()

        for i, data in enumerate(self.train_loader):

            # the image to show
            # the pred data
            # the label
            img, pred, label = self.step(data)

            # compute loss
            metrics = self.compute_metrics(pred, label, is_train=True)

            # get the item for backward
            loss = metrics['train/l1']

            # compute gradient and do Adam step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # logger record
            for key in metrics.keys():
                self.logger.record_scalar(key, metrics[key])

            # only save img at first step
            if i == len(self.train_loader) - 1:
                self.logger.save_imgs(self.gen_imgs_to_write(img, pred, label, True), epoch)

            # monitor training progress
            if i % self.args.print_freq == 0:
                print('Train: Epoch {} batch {} Loss {}'.format(epoch, i, loss))

    def val_per_epoch(self, epoch):
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                img, pred, label = self.step(data)
                metrics = self.compute_metrics(pred, label, is_train=False)

                loss = metrics['val/l1']

                for key in metrics.keys():
                    self.logger.record_scalar(key, metrics[key])

                if i == len(self.val_loader) - 1:
                    self.logger.save_imgs(self.gen_imgs_to_write(img, pred, label, False), epoch)

            if epoch % self.args.val_acc_freq == 0:
                with torch.set_grad_enabled(False):  # save memory during inference
                    eval_acc = compute_accuracy(self.model, self.val_loader, 'cuda:0')
                    print('Val: Epoch {} Acc {}'.format(epoch, eval_acc))


    def step(self, data):
        img, label = data
        # warp input
        img = Variable(img).cuda()
        label = Variable(label).cuda()

        # compute output
        pred = self.model(img)
        return img, pred, label

    def compute_metrics(self, pred, gt, is_train, loss_fn=None):
        if loss_fn is None:
            loss_fn = F.cross_entropy

        # you can call functions in metrics.py
        l1 = loss_fn(pred, gt)
        prefix = 'train/' if is_train else 'val/'
        metrics = {
            prefix + 'l1': l1
        }
        return metrics

    def gen_imgs_to_write(self, img, pred, label, is_train):
        # override this method according to your visualization
        prefix = 'train/' if is_train else 'val/'
        return {
            prefix + 'img': img[0]
            # prefix + 'pred': pred[0],
            # prefix + 'label': label[0]
        }

    def compute_loss(self, pred, gt):
        if self.args.loss == 'l1':
            loss = (pred - gt).abs().mean()
        elif self.args.loss == 'ce':
            loss = torch.nn.functional.cross_entropy(pred, gt)
        else:
            loss = torch.nn.functional.mse_loss(pred, gt)
        return loss


def main():
    trainer = Trainer()
    trainer.train()


if __name__ == '__main__':
    main()
