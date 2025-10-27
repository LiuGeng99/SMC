import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import iCIFAR10, iCIFAR100, iImageNet100, iImageNet1000, iCore50, iMed40, iImageNet100_LT, iLaryngo29
from tqdm import tqdm
import torch

class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment, aug=1, test_dir=None, max_samples_per_class=None):
        self.dataset_name = dataset_name
        self.aug = aug
        self.max_samples_per_class = max_samples_per_class 
        self._setup_data(dataset_name, shuffle, seed, test_dir)
        assert init_cls <= len(self._class_order), "No enough classes."
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)

    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]
    
    def get_accumulate_tasksize(self,task):
        return sum(self._increments[:task+1])
    
    def get_total_classnum(self):
        return len(self._class_order)

    def get_dataset(self, indices, source, mode, appendent=None, ret_data=False, m_rate=None, sm_data=False):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "mem":
            x, y = self._mem_data, self._mem_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        data, targets = [], []
        for idx in indices:
            if m_rate is None:
                class_data, class_targets = self._select(
                    x, y, low_range=idx, high_range=idx + 1,
                    max_samples=self.max_samples_per_class
                )
            else:
                class_data, class_targets = self._select_rmm(
                    x, y, low_range=idx, high_range=idx + 1, m_rate=m_rate,
                    max_samples=self.max_samples_per_class
                )
            data.append(class_data)
            targets.append(class_targets)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        if sm_data:
            return LatentDataset(data, targets, self.use_path, self.class_names)
        else:
            if ret_data:
                return data, targets, DummyDataset(data, targets, trsf, self.use_path, self.aug if mode == "train" else 1)
            else:
                return DummyDataset(data, targets, trsf, self.use_path, self.aug if mode == "train" else 1)

    def _setup_data(self, dataset_name, shuffle, seed, test_dir=None):
        self._seed = seed
        idata = _get_idata(dataset_name)
        idata.download_data(test_dir)

        # Data
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._mem_data, self._mem_targets = idata.mem_data, idata.mem_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path

        # Transforms
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        original_class_names = idata.class_names

        # Order
        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order
        self._class_order = order
        logging.info(self._class_order)

        # Map indices
        self._train_targets = _map_new_class_index(self._train_targets, self._class_order)
        self._mem_targets = _map_new_class_index(self._mem_targets, self._class_order)
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)
        

        if original_class_names and len(original_class_names) > 0:
            valid_order = [idx for idx in order if 0 <= idx < len(original_class_names)]
            self.class_names = [original_class_names[idx] for idx in valid_order]
            for idx in order:
                if idx >= len(original_class_names):
                    self.class_names.append(f"class_{idx}")


    def _select(self, x, y, low_range, high_range, max_samples=None):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        
        if max_samples is not None and max_samples > 0 and len(idxes) > max_samples:
            np.random.seed(self._seed if hasattr(self, '_seed') else 42)
            idxes = np.random.choice(idxes, max_samples, replace=False)
        
        if isinstance(x, np.ndarray):
            x_return = x[idxes]
        else:
            x_return = []
            for id in idxes:
                x_return.append(x[id])
        return x_return, y[idxes]

    def _select_rmm(self, x, y, low_range, high_range, m_rate, max_samples=None):
        assert m_rate is not None
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        
        if m_rate != 0:
            selected_idxes = np.random.randint(
                0, len(idxes), size=int((1 - m_rate) * len(idxes))
            )
            new_idxes = idxes[selected_idxes]
            new_idxes = np.sort(new_idxes)
        else:
            new_idxes = idxes

        if max_samples is not None and max_samples > 0 and len(new_idxes) > max_samples:
            np.random.seed(self._seed if hasattr(self, '_seed') else 42)
            new_idxes = np.random.choice(new_idxes, max_samples, replace=False)
        
        return x[new_idxes], y[new_idxes]

    def getlen(self, index):
        y = self._train_targets
        return np.sum(np.where(y == index))

class LatentDataset(Dataset):
    def __init__(self, images, labels, use_path=False, class_names=None):
        assert len(images) == len(labels), "Data size error!"
        if use_path:
            self.images = images
        else:
            print("loading dataset")
            images_list = []
            for img_path in images:
                images_list.append(np.array(Image.open(img_path).convert('RGB')))
            self.images = images_list

        self.labels = labels
        self.trsf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.use_path = use_path
        self.class_names = class_names

        self.latents = None
        self.indices = torch.arange(len(self.images))
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        if self.use_path:
            image = self.trsf(pil_loader(self.images[idx]))
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]
        return image, label, idx
    
    def get_class_name(self, target):
        if self.class_names is None:
            return f"class_{target}"
        if 0 <= target < len(self.class_names):
            return self.class_names[target]
        return f"unknown_class_{target}"
    
    def get_all_class_names(self):
        if self.class_names is None:
            max_label = int(max(self.labels)) if len(self.labels) > 0 else 0
            return [f"class_{i}" for i in range(max_label + 1)]
        return self.class_names
        
    def set_latents(self, latents):
        self.latents = latents
        
    def get_indexes(self, batch_indices):
        return self.indices[batch_indices]

class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, use_path=False, aug=1):
        assert len(images) == len(labels), "Data size error!"
        if use_path:
            self.images = images
        else:
            print("loading dataset")
            images_list = []
            for img_path in images:
                images_list.append(np.array(Image.open(img_path).convert('RGB')))
            self.images = images_list
        
        self.aug = aug
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

    def _get_nsamples_list(self):
        nsamples_dict = {}
        for label in self.labels:
            int_label = int(label)
            if int_label not in nsamples_dict:
                nsamples_dict[int_label] = 1
            else:
                nsamples_dict[int_label] += 1
                
        nsamples_list = []
        for cls in range(len(nsamples_dict)):
            nsamples_list.append(nsamples_dict[cls])

        return nsamples_list

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.aug == 1:
            if self.use_path:
                image = self.trsf(pil_loader(self.images[idx]))
            else:
                image = self.trsf(Image.fromarray(self.images[idx]))
            label = self.labels[idx]
            return idx, image, label
        else:
            if self.use_path:
                images = [self.trsf(pil_loader(self.images[idx])) for _ in range(self.aug)]
            else:
                images = [self.trsf(Image.fromarray(self.images[idx])) for _ in range(self.aug)]
            label = self.labels[idx]
            return idx, *images, label
        

def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))

def _get_idata(dataset_name):
    name = dataset_name.lower()
    if name == "cifar10":
        return iCIFAR10()
    elif name == "cifar100":
        return iCIFAR100()
    elif name == "imagenet1000":
        return iImageNet1000()
    elif name == "imagenet100":
        return iImageNet100()
    elif name == "med40":
        return iMed40()
    elif name == "core50":
        return iCore50()
    elif name == "imagenet100_lt":
        return iImageNet100_LT()
    elif name == "laryngo":
        return iLaryngo29()
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


def pil_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

def accimage_loader(path):
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)
