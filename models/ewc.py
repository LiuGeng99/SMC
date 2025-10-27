import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.toolkit import target2onehot, tensor2numpy, accuracy

# T = 2
lamda = 1000
fishermax = 0.0001


class EWC(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.fisher = None
        self._network = IncrementalNet(args, False)
        self.batch_size = self.args["batch_size"]
        self.lr = self.args["lr"]
        self.lr_min = self.args["lr_min"]
        self.weight_decay = self.args["weight_decay"]
        self.epochs = self.args["epochs"]
        self.task_ncls = self.args["task_ncls"]
        self.num_workers = self.args["num_workers"]
        self.ncols = self.args["ncols"]

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
            test_dataset = self.data_manager.get_dataset(
                np.arange(cls_begin, cls_end), source="test", mode="test"
            )
            test_loader = DataLoader(
                test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
            )
            y_pred, y_true = self._eval_cnn(test_loader, class_index=np.arange(cls_begin, cls_end))
            total_acc, class_wise_acc = accuracy(y_pred.T[0], y_true)
            task_acc_dict[task] = total_acc
            
        for task in range(self._cur_task + 1, len(self.task_ncls)):
            task_acc_dict[task] = 0
            
        return task_acc_dict

    def incremental_train(self):
        self._cur_task += 1
        self._total_classes = self._known_classes + self.task_ncls[self._cur_task]
        self._network.update_fc(self._total_classes)
        
        if self._cur_task not in self.args["train_tasks"]+self.args["load_tasks"]:
            logging.info(f"Task {self._cur_task} Skip: No training or Loading.")
            return True
        
        test_dataset = self.data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
        
        if self._cur_task in self.args["train_tasks"]:
            train_dataset = self.data_manager.get_dataset(
                np.arange(self._known_classes, self._total_classes),
                source="train",
                mode="train",
                appendent=self._get_memory(),
            )
            self.train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
            )

            if len(self._multiple_gpus) > 1:
                self._network = nn.DataParallel(self._network, self._multiple_gpus)
                
            self._train(self.train_loader, self.test_loader)
            
            if len(self._multiple_gpus) > 1:
                self._network = self._network.module
            
            if self.args["save_checkpoints"] == True:
                self.save_or_load("save")
                
        elif self._cur_task in self.args["load_tasks"]:
            self.save_or_load("load")


        if self.fisher is None:
            self.fisher = self.getFisherDiagonal(self.train_loader)
        else:
            alpha = self._known_classes / self._total_classes
            new_finsher = self.getFisherDiagonal(self.train_loader)
            for n, p in new_finsher.items():
                new_finsher[n][: len(self.fisher[n])] = (
                    alpha * self.fisher[n]
                    + (1 - alpha) * new_finsher[n][: len(self.fisher[n])]
                )
            self.fisher = new_finsher
        self.mean = {
            n: p.clone().detach()
            for n, p in self._network.named_parameters()
            if p.requires_grad
        }

    def _train(self, train_loader, test_loader):
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))
        self._network.to(self._device)
        optimizer = optim.SGD(
            self._network.parameters(),
            momentum=0.9,
            lr=self.args["lr"],
            weight_decay=self.args["weight_decay"],
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=self.epochs[self._cur_task], eta_min=self.args["lr_min"])
        if self._cur_task == 0:
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.epochs[self._cur_task]), ncols=self.ncols)
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]
                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            
            loss = round(losses/len(train_loader),3)
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc, class_wise_acc = self.eval_task(test_loader)
            task_acc_dict = self.eval_task_seperate()
            info = self._epoch_logging(
                epoch, self.epochs[self._cur_task], prog_bar, loss, train_acc, test_acc, task_acc_dict)
        self._task_logging(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.epochs[self._cur_task]), ncols=self.ncols)
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss_clf = F.cross_entropy(
                    logits[:, self._known_classes :], targets - self._known_classes
                )
                loss_ewc = self.compute_ewc()
                loss = loss_clf + lamda * loss_ewc

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            
            loss = round(losses/len(train_loader),3)
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc, class_wise_acc = self.eval_task(test_loader)
            task_acc_dict = self.eval_task_seperate()
            info = self._epoch_logging(
                epoch, self.epochs[self._cur_task], prog_bar, loss, train_acc, test_acc, task_acc_dict)
        self._task_logging(info)

    def compute_ewc(self):
        loss = 0
        if len(self._multiple_gpus) > 1:
            for n, p in self._network.module.named_parameters():
                if n in self.fisher.keys():
                    loss += (
                        torch.sum(
                            (self.fisher[n])
                            * (p[: len(self.mean[n])] - self.mean[n]).pow(2)
                        )
                        / 2
                    )
        else:
            for n, p in self._network.named_parameters():
                if n in self.fisher.keys():
                    loss += (
                        torch.sum(
                            (self.fisher[n])
                            * (p[: len(self.mean[n])] - self.mean[n]).pow(2)
                        )
                        / 2
                    )
        return loss

    def getFisherDiagonal(self, train_loader):
        fisher = {
            n: torch.zeros(p.shape).to(self._device)
            for n, p in self._network.named_parameters()
            if p.requires_grad
        }
        self._network.train()
        optimizer = optim.SGD(self._network.parameters(), lr=self.args["lr"])
        for i, (_, inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            logits = self._network(inputs)["logits"]
            loss = torch.nn.functional.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            for n, p in self._network.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2).clone()
        for n, p in fisher.items():
            fisher[n] = p / len(train_loader)
            fisher[n] = torch.min(fisher[n], torch.tensor(fishermax))
        return fisher
