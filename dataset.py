import json
import random
import torch
from torch.utils.data import Dataset


class ZeroShotDataset(Dataset):

    def __init__(self,
                 samples,
                 all_labels=None,
                 max_num_negatives=10,
                 is_train=True):
        """
        Args:
            samples: list of dicts, each with 'text' and 'labels'
            all_labels: global label set (for negative sampling). If None, computed from samples.
            max_num_negatives: maximum number of negative samples per instance (only used if is_train=True)
            is_train: if True, apply negative sampling
        """
        self.samples = samples
        self.is_train = is_train
        self.max_num_negatives = max_num_negatives

        if all_labels is None:
            self.all_labels = list(
                set(lbl for s in self.samples for lbl in s["labels"]))
        else:
            self.all_labels = all_labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        text = item["text"]
        pos_labels = item["labels"]  # list of strings
        if self.is_train and self.max_num_negatives > 0:
            num_neg = random.randint(1, self.max_num_negatives)
            neg_candidates = [
                l for l in self.all_labels if l not in pos_labels
            ]
            neg_labels = random.sample(neg_candidates,
                                       min(num_neg, len(neg_candidates)))
            all_labels = pos_labels + neg_labels
            targets = [1.0] * len(pos_labels) + [0.0] * len(neg_labels)
        else:
            all_labels = pos_labels
            targets = [1.0] * len(pos_labels)
        return {"text": text, "labels": all_labels, "targets": targets}
