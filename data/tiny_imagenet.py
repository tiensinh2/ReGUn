import os
import random
from typing import Optional

import lightning.pytorch as pl
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import datasets

from utils.utils import build_transforms
from .split_logic import class_split, stratified_split, stratified_split_random


class _TinyImageNetVal(Dataset):
    """Tiny ImageNet validation dataset wrapper."""

    def __init__(self, val_dir: str, class_to_idx: dict, transform=None):
        """Index validation images and labels."""
        self.transform = transform
        ann_path = os.path.join(val_dir, "val_annotations.txt")
        images_dir = os.path.join(val_dir, "images")

        samples = []
        with open(ann_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                image_name, wnid = parts[0], parts[1]
                if wnid not in class_to_idx:
                    raise ValueError(f"Unknown Tiny ImageNet class '{wnid}' in {ann_path}.")
                target = class_to_idx[wnid]
                samples.append((os.path.join(images_dir, image_name), target))

        self.samples = samples
        self.targets = [target for _, target in samples]

    def __len__(self) -> int:
        """Return the number of validation samples."""
        return len(self.samples)

    def __getitem__(self, idx: int):
        """Load a validation image and label by index."""
        path, target = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target


class TinyImageNetDataModule(pl.LightningDataModule):
    """Tiny ImageNet data module with unlearning splits."""

    def __init__(self, cfg):
        """Initialize the datamodule from config."""
        super().__init__()
        self.cfg = cfg
        self.name = cfg.data.name.lower()
        self.data_dir = cfg.data.data_dir
        self.download = bool(cfg.data.download)
        self.bs = int(cfg.data.batch_size)
        self.num_workers = int(cfg.data.num_workers)
        self.pin_memory = bool(cfg.data.pin_memory)
        self.persistent_workers = bool(cfg.data.persistent_workers)
        self.prefetch_factor = int(cfg.data.prefetch_factor)

        self.train_tf, self.eval_tf = build_transforms(cfg)
        self.ds_train = None   # full train with train_tf
        self.ds_val = None     # view with eval_tf
        self.ds_test = None

        self.train_full = None
        self.train_retain = None
        self.train_retain_subset = None
        self.train_forget = None
        self.train_heldout = None

        self.eval_train_full = None
        self.eval_retain = None
        self.eval_retain_subset = None
        self.eval_forget = None
        self.eval_heldout = None
        self.val = None

    def prepare_data(self) -> None:
        """No-op since data is expected to exist locally."""
        return

    def setup(self, stage: Optional[str] = None) -> None:
        """Create datasets and split indices."""
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        root = os.path.join(repo_root, self.data_dir)
        train_dir = os.path.join(root, "train")
        val_dir = os.path.join(root, "val")

        self.ds_train = datasets.ImageFolder(train_dir, transform=self.train_tf)
        self.ds_val = datasets.ImageFolder(train_dir, transform=self.eval_tf)
        self.ds_test = _TinyImageNetVal(val_dir, self.ds_train.class_to_idx, transform=self.eval_tf)
        raw_labels = datasets.ImageFolder(train_dir, transform=None).targets

        # Optional local-only downsampling to speed up debugging and testing
        debug_subset_size = getattr(self.cfg.data, "debug_subset_size", None)
        if debug_subset_size:
            self.ds_train = Subset(self.ds_train, range(debug_subset_size))
            self.ds_val = Subset(self.ds_val, range(debug_subset_size))
            self.ds_test = Subset(self.ds_test, range(debug_subset_size))
            raw_labels = raw_labels[:debug_subset_size]

        sp = self.cfg.split
        scheme = str(sp.scheme)
        if scheme == "stratified":
            idx = stratified_split(
                labels=raw_labels,
                seed=int(sp.seed),
                val_frac=float(sp.validation_frac),
                retain_frac=float(sp.retain_frac),
                forget_frac=float(sp.forget_frac),
                heldout_frac=float(sp.heldout_frac),
            )
        elif scheme == "random":
            idx = stratified_split_random(
                labels=raw_labels,
                seed=int(sp.seed),
                val_frac=float(sp.validation_frac),
                retain_frac=float(sp.retain_frac),
                forget_frac=float(sp.forget_frac),
                heldout_frac=float(sp.heldout_frac),
            )
        elif scheme == "class":
            idx = class_split(
                labels=raw_labels,
                seed=int(sp.seed),
                val_frac=float(sp.validation_frac),
                retain_frac=float(sp.retain_frac),
                heldout_frac=float(sp.heldout_frac),
                forget_classes=list(sp.forget_classes),
                forget_frac=float(sp.forget_frac),
            )

        retain_idx = sorted(idx.retain)
        forget_idx = sorted(idx.forget)
        heldout_idx = sorted(idx.heldout)
        val_idx = sorted(idx.val)

        train_full_idx = sorted(set(retain_idx + forget_idx))

        self.train_full = Subset(self.ds_train, train_full_idx)
        self.train_retain = Subset(self.ds_train, retain_idx)
        self.train_forget = Subset(self.ds_train, forget_idx)
        self.train_heldout = Subset(self.ds_train, heldout_idx)
        rng = random.Random(int(sp.seed))
        retain_subset_idx = retain_idx[:]
        rng.shuffle(retain_subset_idx)
        retain_subset_idx = sorted(retain_subset_idx[:len(forget_idx)])
        self.train_retain_subset = Subset(self.ds_train, retain_subset_idx)

        self.eval_train_full = Subset(self.ds_val, train_full_idx)
        self.eval_retain = Subset(self.ds_val, retain_idx)
        self.eval_retain_subset = Subset(self.ds_val, retain_subset_idx)
        self.eval_forget = Subset(self.ds_val, forget_idx)
        self.eval_heldout = Subset(self.ds_val, heldout_idx)
        self.val = Subset(self.ds_val, val_idx)

    def _dl(self, ds, shuffle):
        """Build a DataLoader for the given dataset."""
        return DataLoader(
            ds,
            batch_size=self.bs,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def train_dataloader(self):              return self._dl(self.train_full, True)
    def val_dataloader(self):                return self._dl(self.val, False)
    def test_dataloader(self):               return self._dl(self.ds_test, False)

    def retain_dataloader(self):             return self._dl(self.train_retain, True)
    def retain_dataloader_subset(self):      return self._dl(self.train_retain_subset, True)
    def forget_dataloader(self):             return self._dl(self.train_forget, True)
    def heldout_dataloader(self):            return self._dl(self.train_heldout, True)

    def train_eval_dataloader(self):         return self._dl(self.eval_train_full, False)
    def retain_eval_dataloader(self):        return self._dl(self.eval_retain, False)
    def retain_eval_dataloader_subset(self): return self._dl(self.eval_retain_subset, False)
    def forget_eval_dataloader(self):        return self._dl(self.eval_forget, False)
    def heldout_eval_dataloader(self):       return self._dl(self.eval_heldout, True)

    def combined_retain_heldout_dataloader(self):
        ds = ConcatDataset([self.train_retain, self.train_heldout])
        return self._dl(ds, True)
