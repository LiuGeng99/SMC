import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import numpy as np

def train(args):
    random_seeds = copy.deepcopy(args["random_seeds"]) # 固定使用1，2，3
    device = copy.deepcopy(args["device"])
    
    args["increment"] = args["task_ncls"][-1]
    args ["init_cls"] = args["task_ncls"][0]

    for random_seed in random_seeds:
        args["device"] = device
        args["random_seed"] = random_seed
        _train(args)

def _train(args):
    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/".format(args["dataset"], args["model_name"])
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}/base{}_inc{}_{}_exemplar{}".format(
        args["dataset"],
        args["model_name"],
        init_cls,
        args["increment"],
        args["prefix"],
        args["memory_per_class"],
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random(args["random_seed"])
    _set_device(args)
    print_args(args)
    data_manager = DataManager( # 看看这里还需要做什么改动
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args["aug"] if "aug" in args else 1
    )
    args["num_workers"] = 4
    args["ncols"] = 80
    args["swanlab"] = True
    
    # args["epochs"] = [1]*10
    
    model = factory.get_model(args["model_name"], args)
    model._set_data_manager(data_manager)

    acc_curve = []
    assert len(args["task_ncls"]) == len(args["epochs"]), "task ncls != epochs!"
    n_tasks = len(args["task_ncls"])

    for task in range(n_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info("Trainable params: {}".format(count_parameters(model._network, True)))
            
        skip = model.incremental_train()
        if skip:
            model.after_task()
            continue
        total_acc, class_wise_acc = model.eval_task(model.test_loader) # 这个让模型自己选择一个固定的即可
        model.after_task()
        
        acc_curve.append(round(total_acc,1))

        logging.info("Class Acc: {}".format(class_wise_acc))
        logging.info("Acc: {}".format(round(total_acc,1)))
        logging.info("CNN top1 curve: {}".format(acc_curve))
        logging.info("Average Acc: {:.1f}".format(round(sum(acc_curve)/len(acc_curve),1)))
        
    model.swanlab_run.finish()


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
