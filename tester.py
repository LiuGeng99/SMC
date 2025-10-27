import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager, DummyDataset
from utils.toolkit import count_parameters
import os
import numpy as np
import pickle

from utils.data import iImageNet_OOD, iImageNet100
from torch.utils.data import DataLoader
import seaborn as sns
from torchvision import transforms
import matplotlib.pyplot as plt

from utils import calibration_tools
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap

import json

def train(args):
    random_seeds = copy.deepcopy(args["random_seeds"]) # 固定使用1，2，3
    device = copy.deepcopy(args["device"])
    
    args["increment"] = args["task_ncls"][-1]
    args ["init_cls"] = args["task_ncls"][0]

    for random_seed in random_seeds:
        args["device"] = device
        args["random_seed"] = random_seed
        _test_corr(args)

# def _train(args):
#     init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
#     logs_name = "logs/{}/{}/".format(args["dataset"], args["model_name"])
    
#     if not os.path.exists(logs_name):
#         os.makedirs(logs_name)

#     logfilename = "logs/{}/{}/base{}_inc{}_{}_exemplar{}".format(
#         args["dataset"],
#         args["model_name"],
#         init_cls,
#         args["increment"],
#         args["prefix"],
#         args["memory_per_class"],
#     )
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s [%(filename)s] => %(message)s",
#         handlers=[
#             logging.FileHandler(filename=logfilename + ".log"),
#             logging.StreamHandler(sys.stdout),
#         ],
#     )

#     _set_random(args["random_seed"])
#     _set_device(args)
#     print_args(args)
#     data_manager = DataManager( # 看看这里还需要做什么改动
#         args["dataset"],
#         args["shuffle"],
#         args["seed"],
#         args["init_cls"],
#         args["increment"],
#         args["aug"] if "aug" in args else 1
#     )
#     args["num_workers"] = 4
#     args["ncols"] = 80
#     args["swanlab"] = False
    
#     args["train_tasks"] = []
#     args["load_tasks"] = [9] # 只load最后一个模型来做测试
#     # args["epochs"] = [1]*10
    
#     model = factory.get_model(args["model_name"], args)
#     model._set_data_manager(data_manager)

#     acc_curve = []
#     assert len(args["task_ncls"]) == len(args["epochs"]), "task ncls != epochs!"
#     n_tasks = len(args["task_ncls"])

#     for task in range(n_tasks):  
#         skip = model.incremental_train()
#         if skip:
#             model.after_task()
#             continue
#         total_acc, class_wise_acc = model.eval_task(model.test_loader) # 这个让模型自己选择一个固定的即可
#         model.after_task()
        
#         acc_curve.append(round(total_acc,1))

#         logging.info("Class Acc: {}".format(class_wise_acc))
#         logging.info("Acc: {}".format(round(total_acc,1)))
#         logging.info("CNN top1 curve: {}".format(acc_curve))
#         logging.info("Average Acc: {:.1f}".format(round(sum(acc_curve)/len(acc_curve),1)))
        
    # model.swanlab_run.finish()
    
def _test_confusion(args):
    _set_random()
    _set_device(args)
    
    args["num_workers"] = 4
    args["ncols"] = 80
    args["swanlab"] = False
    
    model = factory.get_model(args["model_name"], args)

    dataset_name = args["dataset"]
    model_name = args["model_name"]
    random_seed = args["random_seed"]
    prefix = args["prefix"]

    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
    )
    ind_dataset = data_manager.get_dataset(
            np.arange(0, 100), source="test", mode="test"
    )
    ind_loader = DataLoader(
        ind_dataset, batch_size=256, shuffle=False, num_workers=args["num_workers"]
    )

    model._network.update_fc(100)
    
    nsamples = args["memory_per_class"]
    inc = args["increment"]
    checkpoint_dir = f"/home/geng_liu/CL/checkpoints/{dataset_name}/{model_name}/{prefix}_inc{inc}_nsamples_{nsamples}/random_{random_seed}"
    model.load_checkpoint(f"{checkpoint_dir}/task", 9)

    all_preds, all_labels = model.confusion_evaluate(ind_loader)
    cm = confusion_matrix(all_labels, all_preds, labels=np.arange(100))

    with open(f'/home/geng_liu/CL/scripts/SMC_fig/pkls/{model_name}_confusion.pkl', 'wb') as file:
        pickle.dump(cm, file)

def _test_ood(args):
    _set_random()
    _set_device(args)
    
    args["num_workers"] = 4
    args["ncols"] = 80
    args["swanlab"] = False
    
    model = factory.get_model(args["model_name"], args)

    ood_idata = iImageNet_OOD()
    ood_idata.download_data()
    trsf = transforms.Compose([*ood_idata.test_trsf, *ood_idata.common_trsf])
    ood_dataset = DummyDataset(ood_idata.test_data, ood_idata.test_targets, trsf, ood_idata.use_path)
    ood_loader = DataLoader(ood_dataset, batch_size=128, shuffle=False, num_workers=4)

    ind_idata = iImageNet100()
    ind_idata.download_data()
    trsf = transforms.Compose([*ind_idata.test_trsf, *ind_idata.common_trsf])
    ind_dataset = DummyDataset(ind_idata.test_data, ind_idata.test_targets, trsf, ind_idata.use_path)
    ind_loader = DataLoader(ind_dataset, batch_size=128, shuffle=False, num_workers=4)

    dataset_name = args["dataset"]
    model_name = args["model_name"]
    random_seed = args["random_seed"]
    prefix = args["prefix"]

    model._network.update_fc(100)

    nsamples = args["memory_per_class"]
    inc = args["increment"]
    checkpoint_dir = f"/home/geng_liu/CL/checkpoints/{dataset_name}/{model_name}/{prefix}_inc{inc}_nsamples_{nsamples}/random_{random_seed}"
    model.load_checkpoint(f"{checkpoint_dir}/task", 9)
    
    conf_ood = model.ood_evaluate(ood_loader)
    conf_ind = model.ood_evaluate(ind_loader)
    # print(conf_ood)

    measures = calibration_tools.get_measures(-conf_ood, -conf_ind)
    aurocs = measures[0]; auprs = measures[1] ; fprs = measures[2]
    auroc = np.mean(aurocs); aupr = np.mean(auprs) ; fpr = np.mean(fprs)
    print(auroc, aupr, fpr)
    assert 0

    with open(f'/home/geng_liu/CL/scripts/SMC_fig/pkls/{model_name}_ood.pkl', 'wb') as file:
        pickle.dump((conf_ood, conf_ind), file)
        
def _test_tsne(args):
    _set_random()
    _set_device(args)
    
    args["num_workers"] = 4
    args["ncols"] = 80
    args["swanlab"] = False
    
    model = factory.get_model(args["model_name"], args)

    dataset_name = args["dataset"]
    model_name = args["model_name"]
    random_seed = args["random_seed"]
    prefix = args["prefix"]

    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
    )
    ind_dataset = data_manager.get_dataset(np.arange(0, 100), source="test", mode="test")
    ind_loader = DataLoader(ind_dataset, batch_size=256, shuffle=False, num_workers=4)

    model._network.update_fc(100)
    
    nsamples = args["memory_per_class"]
    inc = args["increment"]
    checkpoint_dir = f"/home/geng_liu/CL/checkpoints/{dataset_name}/{model_name}/{prefix}_inc{inc}_nsamples_{nsamples}/random_{random_seed}"
    model.load_checkpoint(f"{checkpoint_dir}/task", 9)

    features_array, labels_array = model.tsne_evaluate(ind_loader)

    with open(f'/home/geng_liu/CL/scripts/SMC_fig/pkls/{model_name}_tsne.pkl', 'wb') as file:
        pickle.dump((features_array, labels_array), file)

def _test_corr(args):
    _set_random()
    _set_device(args)
    
    args["num_workers"] = 4
    args["ncols"] = 80
    args["swanlab"] = False
    
    model = factory.get_model(args["model_name"], args)
    model._network.update_fc(100)
        
    nsamples = args["memory_per_class"]
    inc = args["increment"]
    dataset_name = args["dataset"]
    model_name = args["model_name"]
    random_seed = args["random_seed"]
    prefix = args["prefix"]
    
    checkpoint_dir = f"/home/geng_liu/CL/checkpoints/{dataset_name}/{model_name}/{prefix}_inc{inc}_nsamples_{nsamples}/random_{random_seed}"
    model.load_checkpoint(f"{checkpoint_dir}/task", 9)
    
    level = 1
    
    for corr_type in os.listdir(f"/home/geng_liu/CL/dataset/ImageNet-C/level_{level}"):
        # type = "brightness"
        test_dir = f"/home/geng_liu/CL/dataset/ImageNet-C/level_{level}/{corr_type}"

        data_manager = DataManager(
            args["dataset"],
            args["shuffle"],
            args["seed"],
            args["init_cls"],
            args["increment"],
            test_dir=test_dir
        )
        ind_dataset = data_manager.get_dataset(np.arange(0, 100), source="test", mode="test")
        ind_loader = DataLoader(ind_dataset, batch_size=256, shuffle=False, num_workers=4)
        total_acc, class_wise_acc = model.eval_task(ind_loader)
        json_data = {"type":corr_type, "total":total_acc, "class":class_wise_acc}
        with open(f"/home/geng_liu/CL/scripts/SMC_fig/jsonls/{model_name}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(json_data, ensure_ascii=False) + "\n")

def _test_svd(args):
    _set_random()
    _set_device(args)
    model = factory.get_model(args["model_name"], args)

    ind_idata = iImageNet100()
    ind_idata.download_data()
    trsf = transforms.Compose([*ind_idata.test_trsf, *ind_idata.common_trsf])
    ind_dataset = DummyDataset(ind_idata.test_data, ind_idata.test_targets, trsf, ind_idata.use_path)
    ind_loader = DataLoader(ind_dataset, batch_size=256, shuffle=False, num_workers=8)

    dataset_name = args["dataset"]
    model_name = args["model_name"]
    random_seed = args["random"]

    # data_manager = DataManager(
    #     args["dataset"],
    #     args["shuffle"],
    #     args["seed"],
    #     args["init_cls"],
    #     args["increment"],
    # )
    # for i in range(10):
    #     model.incremental_train(data_manager)
    #     # cnn_accy, nme_accy = model.eval_task()
    #     model.after_task()

    model._network.update_fc(100)
    # model.load_checkpoint(f"/data3/geng_liu/checkpoints_{dataset_name}/{model_name}/random_{random_seed}/task", 9)
    # model.load_checkpoint(f"/data3/geng_liu/checkpoints_{dataset_name}/{model_name}/task", 9)

    model.load_checkpoint(f"/data3/geng_liu/checkpoints_imagenet100/joint/joint_random_1/task", 9)

    features_dict = model.svd_evaluate(ind_loader)

    centered_features = []
    for label, feats in features_dict.items():
        feats = np.array(feats)  # shape: (num_samples, feature_dim)
        mean_feat = np.mean(feats, axis=0, keepdims=True)
        feats_centered = feats - mean_feat
        centered_features.append(feats_centered)

    # 拼接所有类别的中心化特征
    all_features = np.concatenate(centered_features, axis=0)

    # 对中心化后的特征矩阵进行 SVD 分解
    # 注意：这里计算的是完整 SVD，如果特征数较多，计算量可能较大
    U, S, Vt = np.linalg.svd(all_features, full_matrices=False)

    # 绘制奇异值分布图
    print(S.tolist())
    assert 0

    # ours_array = np.array(ours_s)[:100]/np.array(ours_s)[0]
    # replay_array = np.array(replay_s)[:100]/np.array(replay_s)[0]
    # ewc_array = np.array(ewc_s)[:100]/np.array(ewc_s)[0]
    # S_array = S[:100]/S[0]
    fig = plt.figure(figsize=(7,6))
    plt.plot(ewc_array, '-', linewidth=2, c="blue")
    plt.plot(ours_array, '-', linewidth=2, c="green")
    plt.plot(replay_array, '-', linewidth=2, c="red")
    plt.title("Singular Value Distribution")
    plt.xlabel("Singular Value Index")
    plt.ylabel("Singular Value")
    # plt.grid(True)

    plt.tight_layout()
    plt.show()
    fig.savefig(f"/data3/geng_liu/scripts/outputs/svd_{model_name}.svg", format="svg")

def _set_device(args):
    device_type = args["device"]
    gpus = []
    for device in device_type:
        if device == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))
        gpus.append(device)
    args["device"] = gpus

def _set_random(random_seed=1):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
