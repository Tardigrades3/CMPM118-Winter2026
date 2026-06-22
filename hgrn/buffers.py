import torch
import random
import numpy as np


class SimpleMemoryBuffer:
    """Random-replacement bounded buffer for Experience Replay."""
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []

    def add_data(self, sequences, labels, masks):
        for i in range(sequences.size(0)):
            if len(self.buffer) < self.capacity:
                self.buffer.append((sequences[i].cpu(), labels[i].cpu(), masks[i].cpu()))
            else:
                idx = random.randint(0, self.capacity)
                if idx < self.capacity:
                    self.buffer[idx] = (sequences[i].cpu(), labels[i].cpu(), masks[i].cpu())

    def sample(self, batch_size):
        samples = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        seqs = torch.stack([s[0] for s in samples])
        labels = torch.stack([s[1] for s in samples])
        masks = torch.stack([s[2] for s in samples])
        return seqs, labels, masks

    def __len__(self):
        return len(self.buffer)


class HerdingBuffer:
    """iCaRL-inspired herding buffer: stores prototypical exemplars per class."""
    def __init__(self, capacity_per_class=20):
        self.capacity_per_class = capacity_per_class
        self.buffer = {}  # {class_id: [(seq, label, mask), ...]}

    def __len__(self):
        return sum(len(exemplars) for exemplars in self.buffer.values())

    def select_exemplars(self, model, loader, device, num_classes=17):
        model.eval()
        all_data = {cls: [] for cls in range(num_classes)}
        all_feats = {cls: [] for cls in range(num_classes)}

        with torch.no_grad():
            for x, y, m in loader:
                feats = model.get_features(x.to(device)).cpu()
                for i in range(x.size(0)):
                    cls = y[i].item()
                    all_data[cls].append((x[i].cpu(), y[i].cpu(), m[i].cpu()))
                    all_feats[cls].append(feats[i])

        for cls in range(num_classes):
            if not all_feats[cls]:
                continue

            feats = torch.stack(all_feats[cls])
            centroid = feats.mean(dim=0)

            selected_indices = []
            curr_sum = torch.zeros_like(centroid)

            for i in range(min(self.capacity_per_class, len(feats))):
                dists = torch.norm((curr_sum + feats) / (i + 1) - centroid, dim=1)
                dists[selected_indices] = float('inf')
                best_idx = torch.argmin(dists).item()
                selected_indices.append(best_idx)
                curr_sum += feats[best_idx]

            self.buffer[cls] = [all_data[cls][idx] for idx in selected_indices]

    def sample(self, batch_size):
        all_samples = []
        for cls in self.buffer:
            all_samples.extend(self.buffer[cls])

        if not all_samples:
            return None, None, None

        chosen = random.sample(all_samples, min(batch_size, len(all_samples)))
        seqs = torch.stack([x[0] for x in chosen])
        labels = torch.stack([x[1] for x in chosen])
        masks = torch.stack([x[2] for x in chosen])

        return seqs, labels, masks
