import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy, accuracy

from timm.scheduler import CosineLRScheduler

class calibratedCE(nn.Module):
    def __init__(self, cls_num_list, device):
        super().__init__()
        # beta = num_old * [cls_num_list[0] / sum(cls_num_list)] + (num_all - num_old) * [1]
        # all_num = sum(cls_num_list)
        alpha = [cls_num_list[-1] / cls_num for cls_num in cls_num_list]
        alpha = np.array(alpha) ** 0.3
        alpha_classwise = alpha
        
        beta = [cls_num / sum(cls_num_list) for cls_num in cls_num_list]
        beta_classwise = np.log(beta)
        
        self.alpha_classwise = torch.tensor(alpha_classwise, dtype=torch.float32).to(device)
        self.beta_classwise = torch.tensor(beta_classwise, dtype=torch.float32).to(device)

    def forward(self, x, target):
        # output = x / self.alpha_classwise.to(device)
        output = x + self.beta_classwise
        return F.cross_entropy(output, target)

class SMC(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)
        self.batch_size = self.args["batch_size"]
        self.lr = self.args["lr"]
        self.lr_min = self.args["lr_min"]
        self.weight_decay = self.args["weight_decay"]
        self.epochs = self.args["epochs"]
        self.task_ncls = self.args["task_ncls"]
        self.num_workers = self.args["num_workers"]
        self.ncols = self.args["ncols"]
        self.last_epoch = 0

    def after_task(self):
        self._known_classes = self._total_classes
        
    def eval_task(self, test_loader):
        y_pred, y_true = self._eval_cnn(test_loader)
        total_acc, class_wise_acc = accuracy(y_pred.T[0], y_true)
        return total_acc, class_wise_acc
    
    def eval_task_seperate(self): # 分别创建loader，然后分别测试
        cls_begin = 0
        cls_end = 0
        task_acc_dict = {}
        for task in range(self._cur_task + 1): # 之前的每一个task在当前epoch的性能
            cls_begin = cls_end
            cls_end += self.args["task_ncls"][task]
            test_dataset = self.data_manager.get_dataset(np.arange(cls_begin, cls_end), source="test", mode="test")
            test_loader = DataLoader(
                test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
            )
            y_pred, y_true = self._eval_cnn(test_loader, class_index=np.arange(cls_begin, cls_end))
            total_acc, _ = accuracy(y_pred.T[0], y_true, class_wise=False)
            task_acc_dict[task] = total_acc
            
        # 对于还没有开始的task直接给0
        for task in range(self._cur_task + 1, len(self.task_ncls)):
            task_acc_dict[task] = 0
            
        return task_acc_dict

    def incremental_train(self):
        self._cur_task += 1
        self._total_classes = self._known_classes + self.data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        
        if self._cur_task not in self.args["train_tasks"]+self.args["load_tasks"]:
            logging.info(f"Task {self._cur_task} Skip: No training or Loading.")
            return True

        test_dataset = self.data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=self.num_workers
        )

        # 下面就是正常训练
        if self._cur_task in self.args["train_tasks"]:
            both_dataset = self.data_manager.get_dataset(
                # np.arange(self._known_classes, self._total_classes),
                np.arange(0, self._total_classes), # 先改成用全部的
                source="mem",
                mode="train",
            )
            # 暂时用同一个dataset就行了
            self.both_loader = DataLoader(
                both_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=self.num_workers
            )
            
            self.criterion = nn.CrossEntropyLoss()

            if len(self._multiple_gpus) > 1:
                self._network = nn.DataParallel(self._network, self._multiple_gpus)
            self._train(self.both_loader, self.test_loader)
            if len(self._multiple_gpus) > 1:
                self._network = self._network.module
                
            if self.args["save_checkpoints"] == True:
                self.save_or_load("save")
        
        elif self._cur_task in self.args["load_tasks"]: # 之前的都load，为了测试。或者从中断处训练也需要之前的结果
            self.save_or_load("load")
            
        return False
   
    # def _wake_train(self, train_loader, test_loader): # 生成分类器
    #     self._network.eval()
    #     features_list = []
    #     labels_list = []
    #     for i, (_, inputs, targets) in enumerate(train_loader):
    #         inputs, targets = inputs.to(self._device), targets.to(self._device)
    #         with torch.no_grad():
    #             features = self._network.convnet(inputs)["features"]

    #         features_list.append(features.cpu())
    #         labels_list.append(targets.cpu())

    #     features_all = torch.cat(features_list, dim=0) # [N, feat_dim]
    #     labels_all = torch.cat(labels_list, dim=0)     # [N]
    #     num_classes = len(torch.unique(labels_all))

    #     # 分类别求均值
    #     prototypes = [] # shape [num_classes, feat_dim]
    #     for cls in range(num_classes):
    #         cls_feats = features_all[labels_all == cls]
    #         prototype = cls_feats.mean(dim=0)
    #         prototypes.append(prototype)
    #     prototypes = torch.stack(prototypes, dim=0)  # [num_classes, feat_dim]

    #     with torch.no_grad():
    #         self._network.fc.weight.copy_(prototypes)
    #         self._network.fc.bias.zero_()
            
    #     # 完成之后进行一轮测试看看
    #     total_acc, class_wise_acc = self.eval_task(test_loader)
    #     task_acc_dict = self.eval_task_seperate()
    #     logging.info(f"Wake Acc: {total_acc}")
    #     logging.info(f"Wake Task Acc: {task_acc_dict}")
    
        
    def _train(self, both_loader, test_loader):
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))
        
        self._network.to(self._device)
        
        if self._cur_task == 0:
            # init的时候不需要，先弄一下作为debug
            # self._wake_train(both_loader, test_loader)
            sleep_optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=1e-1,
                weight_decay=self.args["weight_decay"],
            )
            
            warmup_scheduler = CosineLRScheduler(
                optimizer=sleep_optimizer, t_initial=self.epochs[self._cur_task]-self.last_epoch, lr_min=self.args["lr_min"],\
                warmup_t=5, warmup_lr_init=1e-3, cycle_limit=1
            )
            self._sleep_train(self.epochs[self._cur_task], both_loader, test_loader, sleep_optimizer, warmup_scheduler)
        else:
            # wake阶段生成classifier
            self._wake_train(both_loader, test_loader)
            
            sleep_optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=self.args["lr"],
                weight_decay=self.args["weight_decay"],
            )
            warmup_scheduler = CosineLRScheduler(
                optimizer=sleep_optimizer, t_initial=self.epochs[self._cur_task]-self.last_epoch, lr_min=self.args["lr_min"],\
                warmup_t=self.args["warmup_t"], warmup_lr_init=1e-3, cycle_limit=1
            )
            self._sleep_train(self.epochs[self._cur_task], both_loader, test_loader, sleep_optimizer, warmup_scheduler)
            
    def _wake_train(self, train_loader, test_loader): # 生成分类器
        # self._network.eval()
        wake_optimizer = optim.SGD(
            self._network.fc.parameters(),
            momentum=0.9,
            lr=self.args["lr"], # 0.1 不确定哪个更好  小lr看起来挺好的
            # lr=0.1,
            weight_decay=self.args["weight_decay"],
        )
        for epoch in range(1):
            self._network.train()
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                with torch.no_grad():
                    features = self._network.convnet(inputs)["features"]
                logits = self._network.fc(features)["logits"]

                loss = self.criterion(logits, targets)

                wake_optimizer.zero_grad()
                loss.backward()
                wake_optimizer.step()
        
            # 完成之后进行一轮测试看看
            self._network.eval()
            total_acc, class_wise_acc = self.eval_task(test_loader)
            task_acc_dict = self.eval_task_seperate()
            logging.info(f"Wake Acc: {total_acc}")
            logging.info(f"Wake Task Acc: {task_acc_dict}")

    def _sleep_train(self, epoch_all, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(epoch_all), ncols=self.ncols)
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss = self.criterion(logits, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
                
            scheduler.step(epoch)
    
            loss = round(losses/len(train_loader),3)
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc, class_wise_acc = self.eval_task(test_loader)
            task_acc_dict = self.eval_task_seperate()
            info = self._epoch_logging(
                epoch, self.epochs[self._cur_task], prog_bar, loss, train_acc, test_acc, task_acc_dict)
        self._task_logging(info)
        
