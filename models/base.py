import copy
import logging
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.toolkit import tensor2numpy, accuracy
from scipy.spatial.distance import cdist
import os
import swanlab


EPSILON = 1e-8
batch_size = 64


class BaseLearner(object):
    def __init__(self, args):
        self.args = args
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._old_network = None
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self.topk = 1 # 默认改小，防止初始类别不足
        
        self.test_curve = [] # 记录每个task的测试acc  其实也不是必要

        self._memory_size = args["memory_size"]
        self._memory_per_class = args.get("memory_per_class", None)
        self._fixed_memory = args.get("fixed_memory", False)
        self._device = args["device"][0]
        self._multiple_gpus = args["device"]

        model_name = self.args["model_name"]
        random_seed = args["random_seed"]
        prefix = args["prefix"]
        dataset = self.args["dataset"]
        
        if self.args["swanlab"]:
            self.swanlab_run = swanlab.init(
                project=f"CIL_{dataset}_{model_name}",
                experiment_name=f"{prefix}_random_{random_seed}",
                config=self.args["config"],
                # resume=True,
                # id="vtlqwp2wzbh5nokwnw8ih",
            )
        
    def _set_data_manager(self, data_manager):
        self.data_manager = data_manager

    @property
    def exemplar_size(self):
        assert len(self._data_memory) == len(
            self._targets_memory
        ), "Exemplar size error."
        return len(self._targets_memory)

    @property
    def samples_per_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, "Total classes is 0"
            return self._memory_size // self._total_classes

    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim

    def build_rehearsal_memory(self, data_manager, per_class):
        if self._fixed_memory:
            self._construct_exemplar_unified(data_manager, per_class)
        else:
            self._reduce_exemplar(data_manager, per_class)
            self._construct_exemplar(data_manager, per_class)
            
    def _epoch_logging(self, current_epoch, epoch_all, prog_bar, loss, train_acc, test_acc, task_acc_dict, **kwargs):
        log_dict = {"Loss": loss, "Train_accy": train_acc, "Test_accy": test_acc}
        for task in task_acc_dict:
            log_dict[f"Task{task}_accy"] = task_acc_dict[task]
        self.swanlab_run.log(log_dict) # 考虑要记录哪些内容
        
        self.test_curve.append(test_acc)
            
        info = "Task {}, Epoch {}/{}: Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
            self._cur_task,
            current_epoch + 1,
            epoch_all,
            loss,
            train_acc,
            test_acc,
        )
        # prog_bar.set_description(info)
        return info
    
    def _task_logging(self, info): # 一整个task结束以后需要的信息  包括 训练测试acc，class wise acc等等
        logging.info(info)
        logging.info(f"Test Curve: {self.test_curve}")


    def save_checkpoint(self, filename):
        self._network.cpu()
        save_dict = {
            "tasks": self._cur_task,
            "model_state_dict": self._network.state_dict(),
        }
        torch.save(save_dict, "{}_{}.pkl".format(filename, self._cur_task))

    def load_checkpoint(self, filename, cur_task):
        if "load_path" in self.args:
            self._network.load_state_dict(torch.load(self.args["load_path"], map_location=torch.device("cpu"))["model_state_dict"])
        else:
            self._network.load_state_dict(torch.load("{}_{}.pkl".format(filename, cur_task), map_location=torch.device("cpu"))["model_state_dict"])
        self._network.to(self._device)
    
    def save_or_load(self, mode):
        dataset_name = self.args["dataset"]
        model_name = self.args["model_name"]
        random_seed = self.args["random_seed"]
        prefix = self.args["prefix"]
        nsamples = self.args["memory_per_class"]
        inc = self.args["increment"]

        checkpoint_dir = f"/home/geng_liu/CL/checkpoints/{dataset_name}/{model_name}/{prefix}_inc{inc}_nsamples_{nsamples}/random_{random_seed}"

        if mode == "save":
            if not os.path.isdir(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            self.save_checkpoint(f"{checkpoint_dir}/task")
            self._network.to(self._device) # save的时候给拿出来了，重新放回device
        elif mode == "load":
            self.load_checkpoint(f"{checkpoint_dir}/task", self._cur_task)

    def after_task(self):
        pass
    
    def confusion_evaluate(self, loader):
        self._network.eval()
        all_preds, all_labels = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
                prob, _pred = F.softmax(outputs, dim=1).max(1)
                all_preds.append(_pred.cpu().numpy())
                all_labels.append(targets.cpu().numpy())
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        return all_preds, all_labels
    
    def ood_evaluate(self, loader):
        self._network.eval()
        conf, correct = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
                prob, _pred = F.softmax(outputs, dim=1).max(1)
                conf.append(prob.detach().cpu().view(-1).numpy())
        conf_ood = np.concatenate(conf, axis=0)
        return conf_ood

    def tsne_evaluate(self, loader):
        self._network.eval()
        all_features = []
        all_labels = []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                features = self._network(inputs)["features"]
                all_features.append(features.detach().cpu().numpy())
                all_labels.append(targets.detach().cpu().numpy())
        features_array = np.concatenate(all_features, axis=0)
        labels_array = np.concatenate(all_labels, axis=0)
        return features_array, labels_array
    
    def svd_evaluate(self, loader):
        self._network.eval()
        features_dict = {}
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                feats = self._network(inputs)["features"]
                feats = feats.detach().cpu().numpy()
                targets = targets.detach().cpu().numpy()
                for feat, label in zip(feats, targets):
                    if label not in features_dict:
                        features_dict[label] = []
                    features_dict[label].append(feat)
        
        return features_dict
    
    # def _get_logits(self, loader, topk=5): # 这里用test loader
    #     self._network.eval()
    #     label_list, logit_list, feature_list = [],[],[]
    #     for _, (_, inputs, targets) in enumerate(loader):
    #         inputs = inputs.to(self._device)
    #         with torch.no_grad():
    #             outputs = self._network(inputs)["logits"]
    #             features = self._network(inputs)["features"]
            
    #         label_list.append(targets.cpu().numpy())
    #         logit_list.append(outputs.cpu().numpy())
    #         feature_list.append(features.cpu().numpy())
            
    #     return [np.concatenate(label_list), np.concatenate(logit_list), np.concatenate(feature_list)]

    # def _logits_eval(self, loader, topk=5): # 这里用test loader
    #     self._network.eval()
    #     correct_top1, correct_topk, wrong_all = [], [], []
    #     feature_correct_top1, feature_correct_topk, feature_wrong_all = [], [], []
    #     for _, (_, inputs, targets) in enumerate(loader):
    #         inputs = inputs.to(self._device)
    #         with torch.no_grad():
    #             outputs = self._network(inputs)["logits"]
    #             features = self._network(inputs)["features"]

    #         _, preds_top1 = torch.max(outputs, 1)
    #         _, preds_top5 = outputs.topk(topk, dim=1)
            
    #         outputs = outputs.cpu()
    #         labels = targets.cpu()
    #         preds_top1 = preds_top1.cpu()
    #         preds_top5 = preds_top5.cpu()

    #         for i in range(len(labels)):
    #             logits = outputs[i].cpu().numpy()
    #             label = labels[i].cpu().numpy()
                
    #             # print(preds_top1[i].cpu().numpy().tolist(), label)
    #             # assert 0
    #             if preds_top1[i].cpu().numpy().tolist() == label:
    #                 correct_top1.append(logits)
    #                 feature_correct_top1.append(features[i].cpu().numpy())
    #             else:
    #                 top5_correct = label in preds_top5[i].tolist()
    #                 if top5_correct:
    #                     correct_topk.append(logits)
    #                     feature_correct_topk.append(features[i].cpu().numpy())
    #                 else:
    #                     wrong_all.append(logits)
    #                     feature_wrong_all.append(features[i].cpu().numpy())
    #     # 根据各个列表的长度就可以判断acc

    #     return [correct_top1, correct_topk, wrong_all, feature_correct_top1, feature_correct_topk, feature_wrong_all]


    # def eval_task(self, save_conf=False):
    #     y_pred, y_true = self._eval_cnn(self.test_loader)
    #     cnn_accy = self._evaluate(y_pred, y_true)

    #     if hasattr(self, "_class_means"):
    #         y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
    #         nme_accy = self._evaluate(y_pred, y_true)
    #     else:
    #         nme_accy = None

    #     if save_conf:
    #         _pred = y_pred.T[0]
    #         _pred_path = os.path.join(self.args['logfilename'], "pred.npy")
    #         _target_path = os.path.join(self.args['logfilename'], "target.npy")
    #         np.save(_pred_path, _pred)
    #         np.save(_target_path, y_true)

    #         _save_dir = os.path.join(f"./results/conf_matrix/{self.args['prefix']}")
    #         os.makedirs(_save_dir, exist_ok=True)
    #         _save_path = os.path.join(_save_dir, f"{self.args['csv_name']}.csv")
    #         with open(_save_path, "a+") as f:
    #             f.write(f"{self.args['time_str']},{self.args['model_name']},{_pred_path},{_target_path} \n")

    #     return cnn_accy, nme_accy

    def incremental_train(self):
        pass

    def _train(self):
        pass

    def _get_memory(self):
        if len(self._data_memory) == 0:
            return None
        else:
            return (self._data_memory, self._targets_memory)

    # def _compute_accuracy(self, model, loader):
    #     model.eval()
    #     correct, total = 0, 0
    #     for i, (_, inputs, targets) in enumerate(loader):
    #         inputs = inputs.to(self._device)
    #         with torch.no_grad():
    #             outputs = model(inputs)["logits"]
    #         predicts = torch.max(outputs, dim=1)[1]
    #         correct += (predicts.cpu() == targets).sum()
    #         total += len(targets)

    #     return np.around(tensor2numpy(correct) * 100 / total, decimals=2)
    
    # # 不算topk就用不到这个
    # def _evaluate(self, y_pred, y_true): # 根据输入的预测结果和label得到acc 统一都会用的
    #     # ret = {}
    #     total_acc, class_wise_acc = accuracy(y_pred.T[0], y_true) # 相当于取top1
    #     # ret["grouped"] = grouped
    #     # ret["top1"] = grouped["total"]
    #     # ret["top{}".format(self.topk)] = np.around(
    #     #     (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
    #     #     decimals=2,
    #     # )

    #     return total_acc, class_wise_acc
    
    def _eval_cnn(self, loader, class_index=None): # 输出预测结果和label  预测结果取self.topk
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
                if isinstance(class_index, list):
                    outputs = outputs[:,class_index]
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1].cpu().numpy()
            # [bs, topk] 输出元组(value, index)，只取indexes
            if isinstance(class_index, list): # 还原为原类别编号
                predicts = np.array(class_index)[predicts]
            y_pred.append(predicts)
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    # def _eval_nme(self, loader, class_means): # 这两个选择性调用一个就好了
    #     self._network.eval()
    #     vectors, y_true = self._extract_vectors(loader)
    #     vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

    #     dists = cdist(class_means, vectors, "sqeuclidean")  # [nb_classes, N]
    #     scores = dists.T  # [N, nb_classes], choose the one with the smallest distance

    #     return np.argsort(scores, axis=1)[:, : self.topk], y_true  # [N, topk]
    
    def _eval_nme(self, loader, class_means, class_index=None):
        self._network.eval()
        vectors, y_true = self._extract_vectors(loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

        # 选择性的只用部分类别
        if class_index is not None:
            class_means = class_means[class_index]  # shape: [len(class_index), feature_dim]

        dists = cdist(class_means, vectors, "sqeuclidean")  # [selected_classes, N]
        scores = dists.T  # [N, selected_classes]

        # 得到topk index，注意index是指在class_index列表的下标
        preds = np.argsort(scores, axis=1)[:, : self.topk]  # shape: [N, topk]
        if class_index is not None:
            # 还原为原类别编号
            preds = np.array(class_index)[preds]

        return preds, y_true  # [N, topk], [N]

    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            _targets = _targets.numpy()
            if isinstance(self._network, nn.DataParallel):
                _vectors = tensor2numpy(
                    self._network.module.extract_vector(_inputs.to(self._device))
                )
            else:
                _vectors = tensor2numpy(
                    self._network.extract_vector(_inputs.to(self._device))
                )

            vectors.append(_vectors)
            targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)

    def _reduce_exemplar(self, data_manager, m):
        logging.info("Reducing exemplars...({} per classes)".format(m))
        dummy_data, dummy_targets = copy.deepcopy(self._data_memory), copy.deepcopy(
            self._targets_memory
        )
        self._class_means = np.zeros((self._total_classes, self.feature_dim))
        self._data_memory, self._targets_memory = np.array([]), np.array([])

        for class_idx in range(self._known_classes):
            mask = np.where(dummy_targets == class_idx)[0]
            dd, dt = dummy_data[mask][:m], dummy_targets[mask][:m]
            self._data_memory = (
                np.concatenate((self._data_memory, dd))
                if len(self._data_memory) != 0
                else dd
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, dt))
                if len(self._targets_memory) != 0
                else dt
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(dd, dt)
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar(self, data_manager, m):
        logging.info("Constructing exemplars...({} per classes)".format(m))
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []  # [n, feature_dim]
            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection

            # uniques = np.unique(selected_exemplars, axis=0)
            # print('Unique elements: {}'.format(len(uniques)))
            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar_unified(self, data_manager, m):
        logging.info(
            "Constructing exemplars for new classes...({} per classes)".format(m)
        )
        _class_means = np.zeros((self._total_classes, self.feature_dim))

        # Calculate the means of old classes with newly trained network
        for class_idx in range(self._known_classes):
            mask = np.where(self._targets_memory == class_idx)[0]
            class_data, class_targets = (
                self._data_memory[mask],
                self._targets_memory[mask],
            )

            class_dset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(class_data, class_targets)
            )
            class_loader = DataLoader(
                class_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        # Construct exemplars for new classes and calculate the means
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, class_dset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            class_loader = DataLoader(
                class_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )

            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []
            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))

                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection

            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # Exemplar mean
            exemplar_dset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            exemplar_loader = DataLoader(
                exemplar_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(exemplar_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        self._class_means = _class_means
