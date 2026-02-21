import torch
from torch.utils.data import DataLoader
from .datasets import Market1501, DukeMTMC_reID, MSMT17, VeRi776, CUB200, InShop, SOP
from .datasets.bases import ImageDataset
from .sampler import RandomIdentitySampler
import numpy as np
import random
import torchvision.transforms as T




class DataLoaderFactory:
    def __init__(self, cfg):

        self.cfg = cfg
        self.factory = {"Market1501": Market1501, "DukeMTMC_reID": DukeMTMC_reID, 'MSMT17': MSMT17, "VeRi776":VeRi776,
                         'CUB200': CUB200, 'InShop': InShop, 'SOP': SOP}
        
        # Define transforms
        self.train_transforms = self._build_train_transforms(self.cfg.INPUT.SIZE_TRAIN, self.cfg.INPUT.PADDING)
        self.test_transforms = self._build_test_transforms(self.cfg.INPUT.SIZE_TEST)

    def _build_train_transforms(self, size, padding):
        transforms = [
            T.Resize(size),
            T.RandomHorizontalFlip(p=self.cfg.INPUT.PROB),
            T.Pad(padding),
            T.RandomCrop(size),
            T.ToTensor(),
            T.Normalize(mean=self.cfg.INPUT.PIXEL_MEAN, std=self.cfg.INPUT.PIXEL_STD)
        ]
        if self.cfg.INPUT.RE_PROB > 0:
            transforms.append(T.RandomErasing(p=self.cfg.INPUT.RE_PROB, value=self.cfg.INPUT.PIXEL_MEAN))
        return T.Compose(transforms)

    def _build_test_transforms(self, size):
        return T.Compose([
            T.Resize(size),
            T.ToTensor(),
            T.Normalize(mean=self.cfg.INPUT.PIXEL_MEAN, std=self.cfg.INPUT.PIXEL_STD)
        ])

   
    def _worker_init_fn(self, worker_id):
        """Worker init function to set random seeds."""
        np.random.seed(self.cfg.SOLVER.SEED + worker_id)
        random.seed(self.cfg.SOLVER.SEED + worker_id)

    @staticmethod
    def train_collate_fn(batch):
        img, pids, _, _ = zip(*batch)
        pids = torch.tensor(pids, dtype=torch.int64)
        return torch.stack(img, dim=0), pids


    @staticmethod
    def val_collate_fn(batch):
        imgs, pids, camids, _ = zip(*batch)
        pids = torch.tensor(pids, dtype=torch.int64)
        camids = torch.tensor(camids, dtype=torch.int64)
        return torch.stack(imgs, dim=0), pids, camids

    def create_dataloaders(self):
        if self.cfg.DATASETS.NAMES not in self.factory:
            raise ValueError(
                f"Dataset {self.cfg.DATASETS.NAMES} is not supported. "
                f"Available datasets: {list(self.factory.keys())}"
            )
        
        self.dataset = self.factory[self.cfg.DATASETS.NAMES](root=self.cfg.DATASETS.ROOT_DIR)
        self.num_classes = self.dataset.num_train_pids

        # Student train loader
        student_train_set = ImageDataset(self.dataset.train, self.train_transforms)
        if self.cfg.DATALOADER.SAMPLER == 'triplet':
            print('Using triplet sampler')
            # Ensure IMS_PER_BATCH is divisible by NUM_INSTANCE
            if self.cfg.SOLVER.IMS_PER_BATCH % self.cfg.DATALOADER.NUM_INSTANCE != 0:
                raise ValueError(
                    f"cfg.SOLVER.IMS_PER_BATCH ({self.cfg.SOLVER.IMS_PER_BATCH}) must be divisible by "
                    f"cfg.DATALOADER.NUM_INSTANCE ({self.cfg.DATALOADER.NUM_INSTANCE}). Please adjust your configuration."
                )
            student_train_loader = DataLoader(
                student_train_set,
                batch_size=self.cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(
                    self.dataset.train,
                    self.cfg.SOLVER.IMS_PER_BATCH,
                    self.cfg.DATALOADER.NUM_INSTANCE
                ),
                num_workers=self.cfg.DATALOADER.NUM_WORKERS,
                collate_fn=self.train_collate_fn,
                pin_memory=True,
                worker_init_fn=self._worker_init_fn
            )

        elif self.cfg.DATALOADER.SAMPLER == 'random':
            print('Using random sampler')
            student_train_loader = DataLoader(
                student_train_set,
                batch_size=self.cfg.SOLVER.IMS_PER_BATCH,
                shuffle=True,
                num_workers=self.cfg.DATALOADER.NUM_WORKERS,
                collate_fn=self.train_collate_fn,
                pin_memory=True,
                drop_last=True,
                worker_init_fn=self._worker_init_fn
            )
        else:
            raise ValueError(
                f"Unsupported sampler: expected 'random' or 'triplet' but got {self.cfg.DATALOADER.SAMPLER}"
            )

        if self.cfg.DISTILLER.TYPE == "NONE":
            distillation_loader = None
            
            
        else:
            if  self.cfg.DATALOADER.DISTILLATION_SAMPLER == 'triplet':
                distillation_loader = DataLoader(
                    student_train_set,
                    batch_size=self.cfg.SOLVER.IMS_DISTILLATION_PER_BATCH,
                    sampler=RandomIdentitySampler(
                    self.dataset.train,
                    self.cfg.SOLVER.IMS_DISTILLATION_PER_BATCH,
                    self.cfg.DATALOADER.DISTILLATION_NUM_INSTANCE),
                    num_workers=self.cfg.DATALOADER.NUM_WORKERS,
                    collate_fn=self.train_collate_fn,
                    pin_memory=True,
                    drop_last=True,
                    worker_init_fn=self._worker_init_fn)
                
            elif self.cfg.DATALOADER.DISTILLATION_SAMPLER == 'random':
                # Distillation loader
                distillation_loader = DataLoader(
                    student_train_set,
                    batch_size=self.cfg.SOLVER.IMS_DISTILLATION_PER_BATCH,
                    shuffle=True,
                    num_workers=self.cfg.DATALOADER.NUM_WORKERS,
                    collate_fn=self.train_collate_fn,
                    pin_memory=True,
                    drop_last=True,
                    worker_init_fn=self._worker_init_fn
                )

            else:
                raise ValueError(
                    f"Unsupported sampler: expected 'random' or 'triplet' but got {self.cfg.DATALOADER.DISTILLATION_SAMPLER}"
                )

       
        # Query loader
        query_set = ImageDataset(self.dataset.query, self.test_transforms)
        query_loader = DataLoader(
            query_set,
            batch_size=self.cfg.TEST.IMS_PER_BATCH,
            shuffle=False,
            num_workers=self.cfg.DATALOADER.NUM_WORKERS,
            collate_fn=self.val_collate_fn
        )

        # Gallery loader
        gallery_set = ImageDataset(self.dataset.gallery, self.test_transforms)
        gallery_loader = DataLoader(
            gallery_set,
            batch_size=self.cfg.TEST.IMS_PER_BATCH,
            shuffle=False,
            num_workers=self.cfg.DATALOADER.NUM_WORKERS,
            collate_fn=self.val_collate_fn
        )
        
        return student_train_loader, distillation_loader, query_loader, gallery_loader, self.num_classes
