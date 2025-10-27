import logging
import copy
import torch
import torch.nn as nn
from utils import factory
from utils.data_manager import DataManager
import os
import numpy as np
import json
import matplotlib.pyplot as plt
from PIL import Image

import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm

from compressai.losses import RateDistortionLoss
from compressai.zoo import bmshj2018_factorized_relu 
from compressai.optimizers import net_aux_optimizer

from utils.smc_utils.sm_model import SM_Model


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def configure_optimizers(net, lr, aux_lr):
    conf = {
        "net": {"type": "Adam", "lr": lr},
        "aux": {"type": "Adam", "lr": aux_lr},
    }
    optimizer = net_aux_optimizer(net, conf)
    return optimizer["net"], optimizer["aux"]

def train_one_epoch(epoch, model, criterion, train_dataloader, optimizer, device, aux_optimizer=None, clip_max_norm=1.0):
    """Train one epoch, supporting main optimizer and auxiliary optimizer"""
    model.train()

    loss_meter = AverageMeter()
    bpp_meter = AverageMeter()
    mse_meter = AverageMeter()
    
    for images, targets, batch_indices in train_dataloader:
        images = images.to(device)
        batch_size = images.size(0)
        
        if hasattr(train_dataloader.dataset, 'get_indexes'):
            indexes = train_dataloader.dataset.get_indexes(batch_indices)
        else:
            indexes = batch_indices
        
        latents = model.learnable_latents[indexes].to(device)
        optimizer.zero_grad()
        aux_optimizer.zero_grad()
        
        latent_output = model.latent_codec.forward(latents)
        y_hat = latent_output["y_hat"]
        likelihoods = latent_output["likelihoods"]
        x_hat = model.g_s(y_hat).clamp_(0, 1)
        
        # Prepare output for RateDistortionLoss
        out_net = {
            "y_hat": y_hat,
            "likelihoods": likelihoods,
            "x_hat": x_hat  # Add x_hat to the output dictionary
        }

        out_criterion = criterion(out_net, images)
        loss = out_criterion["loss"]
        
        loss.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        
        optimizer.step()
        aux_optimizer.step()
        
        loss_meter.update(loss.item(), batch_size)
        if "bpp_loss" in out_criterion:
            bpp_meter.update(out_criterion["bpp_loss"].item(), batch_size)
        if "mse_loss" in out_criterion:
            mse_meter.update(out_criterion["mse_loss"].item(), batch_size)
    
    return {
        "loss": loss_meter.avg,
        "bpp": bpp_meter.avg,
        "mse": mse_meter.avg,
    }

def calculate_model_size(model):
    """Calculate model size (MB)"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    total_size = (param_size + buffer_size) / 1024 / 1024
    
    return total_size


def calculate_entropy_encoded_size(latents, entropy_model, device):
    """Calculate entropy-encoded latents size (MB)"""
    with torch.no_grad():
        latents = latents.to(device)
        strings = entropy_model.compress(latents)
        
        total_bytes = 0
        for s in strings:
            if isinstance(s, bytes):
                total_bytes += len(s)
            elif isinstance(s, list):
                for sub_s in s:
                    total_bytes += len(sub_s) if isinstance(sub_s, bytes) else 0
    total_mb = total_bytes / 1024 / 1024
    
    return total_mb

def compute_psnr(img1, img2):
    """Compute PSNR (Peak Signal-to-Noise Ratio)"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def compute_ssim(img1, img2):
    """Compute SSIM (Structural Similarity Index)"""
    # Ensure inputs are in the correct range
    img1 = img1.clamp(0, 1)
    img2 = img2.clamp(0, 1)
    
    # SSIM implementation
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    img1 = img1 * 255
    img2 = img2 * 255
    
    mu1 = torch.nn.AvgPool2d(kernel_size=11, stride=1, padding=5)(img1)
    mu2 = torch.nn.AvgPool2d(kernel_size=11, stride=1, padding=5)(img2)
    
    sigma1 = torch.nn.AvgPool2d(kernel_size=11, stride=1, padding=5)(img1 ** 2) - mu1 ** 2
    sigma2 = torch.nn.AvgPool2d(kernel_size=11, stride=1, padding=5)(img2 ** 2) - mu2 ** 2
    sigma12 = torch.nn.AvgPool2d(kernel_size=11, stride=1, padding=5)(img1 * img2) - mu1 * mu2
    
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2))
    return ssim_map.mean().item()


def test_model(model, test_latents, test_dataset, device, batch_size):
    model.eval()
    model = model.to(device)
    
    # Create data loader
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Calculate sizes of different model components
    decoder_size = calculate_model_size(model.g_s)
    entropy_model_size = calculate_model_size(model.latent_codec.entropy_bottleneck)
    
    # Calculate total entropy-encoded latents size
    total_latent_size = calculate_entropy_encoded_size(
        test_latents, model.latent_codec.entropy_bottleneck, device
    )
    
    total_compressed_size = decoder_size + entropy_model_size + total_latent_size

    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    
    print("Starting reconstruction quality evaluation...")
    test_iterator = tqdm(test_dataloader, desc="Evaluation Progress", ncols=100)
    
    with torch.no_grad():
        for images, targets, batch_indices in test_iterator:
            images = images.to(device)
            batch_size = images.size(0)
            
            if hasattr(test_dataloader.dataset, 'get_indexes'):
                indexes = test_dataloader.dataset.get_indexes(batch_indices)
            else:
                indexes = batch_indices

            latents = test_latents[indexes].to(device)

            latent_output = model.latent_codec.forward(latents)
            y_hat = latent_output["y_hat"]
            x_hat = model.g_s(y_hat).clamp_(0, 1)

            for i in range(batch_size):
                psnr = compute_psnr(images[i], x_hat[i])
                ssim = compute_ssim(images[i], x_hat[i])
                psnr_meter.update(psnr)
                ssim_meter.update(ssim)

    return {
        "total_memoriy_size_mb": total_compressed_size,
        "avg_psnr": psnr_meter.avg,
    }

def generate_memorized_dataset(model, test_latents, test_dataset, device, data_manager, output_dir, batch_size):
    model.eval()
    model = model.to(device)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    target_to_class_name = {}
    
    if hasattr(data_manager, 'class_names') and data_manager.class_names:
        for idx, class_name in enumerate(data_manager.class_names):
            target_to_class_name[idx] = class_name
    else:
        original_dataset = None
        if hasattr(test_dataset, 'dataset'):
            original_dataset = test_dataset.dataset
        else:
            original_dataset = test_dataset
        
        if hasattr(original_dataset, 'class_to_idx'):
            original_class_to_idx = original_dataset.class_to_idx
            if hasattr(data_manager, '_class_order'):
                class_order = data_manager._class_order
                for internal_idx, original_idx in enumerate(class_order):
                    for folder_name, idx in original_class_to_idx.items():
                        if idx == original_idx:
                            target_to_class_name[internal_idx] = folder_name
                            break
    
    if not target_to_class_name:
        unique_targets = set([target.item() for _, target, _ in test_dataset])
        for target in unique_targets:
            if target not in target_to_class_name:
                target_to_class_name[target] = f"class_{target}"
    
    for target, class_name in target_to_class_name.items():
        class_dir = os.path.join(output_dir, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
    
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    image_counters = {target: 0 for target in target_to_class_name.keys()}
    
    print(f"Generating memorized dataset to {output_dir}...")
    print(f"Found {len(target_to_class_name)} classes with original folder names")
    test_iterator = tqdm(test_dataloader, desc="Generating Dataset", ncols=100)
    
    with torch.no_grad():
        for images, targets, batch_indices in test_iterator:
            images = images.to(device)
            batch_size = images.size(0)
            
            if hasattr(test_dataloader.dataset, 'get_indexes'):
                indexes = test_dataloader.dataset.get_indexes(batch_indices)
            else:
                indexes = batch_indices
            
            latents = test_latents[indexes].to(device)
            latent_output = model.latent_codec.forward(latents)
            y_hat = latent_output["y_hat"]
            x_hat = model.g_s(y_hat).clamp_(0, 1)
            
            for i in range(batch_size):
                target_class = targets[i].item()
                class_name = target_to_class_name.get(target_class, f"unknown_class_{target_class}")
                
                class_dir = os.path.join(output_dir, class_name)
                if not os.path.exists(class_dir):
                    os.makedirs(class_dir)
                
                filename = f"{class_name}_{image_counters.get(target_class, 0)}.png"
                filepath = os.path.join(class_dir, filename)
                
                img_tensor = x_hat[i]
                img = img_tensor.permute(1, 2, 0).cpu().numpy()
                img = (img * 255).astype(np.uint8)
                pil_img = Image.fromarray(img)

                pil_img.save(filepath)
                
                if target_class in image_counters:
                    image_counters[target_class] += 1
                else:
                    image_counters[target_class] = 1

def generate_latents(model, dataset, device, batch_size):
    """Generate latent representations for all images in the dataset"""
    model.eval()
    model = model.to(device)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    all_latents = []
    all_indexes = []
    
    with torch.no_grad():
        for batch_idx, (images, targets, batch_indices) in enumerate(tqdm(dataloader, desc="Generating latents", ncols=100)):
            images = images.to(device)
            latents = model.g_a(images)
            print(latents.shape)
            assert 0
            all_latents.append(latents)
            all_indexes.append(batch_indices)
    all_latents = torch.cat(all_latents)
    all_indexes = torch.cat(all_indexes)
    all_latents = all_latents.to(device)
    
    return all_latents, all_indexes


CHECKPOINT_DIR = "/home/geng_liu/CL/checkpoints/SM"

class sm_args_class():
    sm_arg_path = "./utils/smc_utils/sm_args.json"
    with open(sm_arg_path) as data_file:
        param = json.load(data_file)
    architecture = param["architecture"]
    quality = param["quality"]
    metric = param["metric"]
    latent_lr = param["latent_lr"]
    decoder_lr = param["decoder_lr"]
    lmbda = param["lmbda"]
    epochs = param["epochs"]
    batch_size = param["batch_size"]
    clip_max_norm = param["clip-max-norm"]
    workers = param["workers"]
    checkpoint_name = "checkpoint_sm"

def train(args):
    device = copy.deepcopy(args["device"])
    
    args["increment"] = args["task_ncls"][-1]
    args["init_cls"] = args["task_ncls"][0]

    args["device"] = device
    args["random_seed"] = 1
    _train(args)

def _train(args): # Use for loop to train memory module
    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]

    _set_random(args["random_seed"])
    _set_device(args)
    print_args(args)
    
    device = args["device"][0]
    
   
    print(f"Using device: {device}")

    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args["aug"] if "aug" in args else 1,
        max_samples_per_class=args["memory_per_class"]
    )
    args["num_workers"] = 4
    args["ncols"] = 80
    args["swanlab"] = False

    assert len(args["task_ncls"]) == len(args["epochs"]), "task ncls != epochs!"
    n_tasks = len(args["task_ncls"])

    _known_classes = 0
    all_test_results = []
    
    for task in range(n_tasks):
        _total_classes = _known_classes + data_manager.get_task_size(task)
        print(f"Loading dataset for task {task+1}")
        
        if hasattr(data_manager, 'class_names') and data_manager.class_names:
            valid_indices = [i for i in range(_known_classes, _total_classes) if i < len(data_manager.class_names)]
            current_class_names = [data_manager.class_names[i] for i in valid_indices]
        
        sm_dataset = data_manager.get_dataset(
                np.arange(_known_classes, _total_classes),  
                source="train",
                mode="train",
                sm_data=True
            )

        sm_args = sm_args_class()
        
        sm_base_model = bmshj2018_factorized_relu(
            quality=sm_args.quality,
            metric=sm_args.metric,
            pretrained=True,
            progress=True,
            device=device
        ).to(device)

        print("Generating latents...")
        train_latents, _ = generate_latents(sm_base_model, sm_dataset, device, sm_args.batch_size)
        
        sm_dataset.set_latents(train_latents)
        
        train_dataloader = DataLoader(
            sm_dataset, 
            batch_size=sm_args.batch_size, 
            shuffle=True, 
            num_workers=args["num_workers"],
            pin_memory=True 
        )
        
        sm_model = SM_Model(sm_base_model, train_latents).to(device)
        optimizer, aux_optimizer = configure_optimizers(sm_model, sm_args.latent_lr, sm_args.decoder_lr)
        criterion = RateDistortionLoss(lmbda=sm_args.lmbda)
        best_loss = float("inf")

        for epoch in tqdm(range(sm_args.epochs), total=sm_args.epochs, desc="Training", ncols=80):
            train_metrics = train_one_epoch(
                epoch,
                sm_model,
                criterion,
                train_dataloader,
                optimizer,
                device,
                aux_optimizer,
                sm_args.clip_max_norm
            )
        
        if args.get('test_after_train', True):
            print(f"Testing Task {task+1} Model")
            task_output_dir = args["output_dir"]
            os.makedirs(task_output_dir, exist_ok=True)
            
            test_results = test_model(
                sm_model,
                train_latents,
                sm_dataset,
                device,
                sm_args.batch_size
            )
            
            all_test_results.append({
                'task': task + 1,
                'results': test_results
            })
            
            if args.get('generate_dataset', True):
                print(f"Generating Memorized Dataset for Task {task+1}")
                generate_memorized_dataset(
                    sm_model,
                    train_latents,
                    sm_dataset,
                    device,
                    data_manager,
                    task_output_dir,
                    sm_args.batch_size
                )
        _known_classes = _total_classes
    

    if args.get('test_after_train', True) and all_test_results:
        total_memory_size = 0
        total_psnr = 0
        for result in all_test_results:
            task_results = result['results']
            total_memory_size += task_results['total_memoriy_size_mb']
            total_psnr += task_results['avg_psnr']
        avg_psnr = total_psnr / len(all_test_results)
        print(f"memory size: {total_memory_size:.2f}")
        print(f"avg psnr: {avg_psnr:.2f}")

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
        
def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)
    args.update(param)

    train(args)

def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param

def setup_parser():
    parser = argparse.ArgumentParser(description='Training script for SM model.')
    parser.add_argument('--config', type=str, default='/home/geng_liu/CL/SMC/exps/med40/smc.json',
                        help='Json file of settings.')
    parser.add_argument('--test-after-train', action='store_true', default=True,
                        help='Run tests after training each task')
    parser.add_argument('--generate-dataset', action='store_true', default=True,
                        help='Generate memorized images dataset after training each task')
    parser.add_argument('--output-dir', type=str, default='/home/geng_liu/CL/dataset/Med/data/memorized',
                        help='Output directory for generated memorized dataset')

    return parser

if __name__ == '__main__':
    main()