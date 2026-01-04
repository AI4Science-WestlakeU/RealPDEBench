import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

class IdentityNormalizer(object):
    def __init__(self, device):
        super(IdentityNormalizer, self).__init__()
        self.device = device

    def preprocess(self, x, y):
        x, y = x.to(self.device), y.to(self.device)
        return x, y

    def postprocess(self, x, y):
        x, y = x.to(self.device), y.to(self.device)
        return x, y


class GaussianNormalizer(object):
    def __init__(self, dataset, device, batch_size=512, is_save=True):
        super(GaussianNormalizer, self).__init__()
        self.device = device
        dataset_dir = dataset.dataset_dir

        if is_save:
            try:
                self.mean_inputs, self.mean_targets, self.std_inputs, self.std_targets \
                                                    = torch.load(os.path.join(dataset_dir, "mean_std.pt"))
            except:
                self.mean_inputs, self.mean_targets, self.std_inputs, self.std_targets \
                                                    = self.compute_mean_std(dataset, batch_size)
                torch.save((self.mean_inputs, self.mean_targets, self.std_inputs, self.std_targets), 
                        os.path.join(dataset_dir, "mean_std.pt"))
        else:
            self.mean_inputs, self.mean_targets, self.std_inputs, self.std_targets \
                                                = self.compute_mean_std(dataset, batch_size)
        # print(self.mean_inputs)
        # print(self.mean_targets)
        # print(self.std_inputs)
        # print(self.std_targets)

        self.mean_inputs = self.mean_inputs.to(device)
        self.mean_targets = self.mean_targets.to(device)
        self.std_inputs = self.std_inputs.to(device)
        self.std_targets = self.std_targets.to(device)
        self.std_inputs = torch.where(self.std_inputs==0, torch.ones_like(self.std_inputs), self.std_inputs)
        self.std_targets = torch.where(self.std_targets==0, torch.ones_like(self.std_targets), self.std_targets)

    def preprocess(self, x, y):
        c1, c2 = x.shape[-1], y.shape[-1]
        x, y = x.to(self.device), y.to(self.device)
        x = (x - self.mean_inputs[..., :c1]) / self.std_inputs[..., :c1]
        y = (y - self.mean_targets[..., :c2]) / self.std_targets[..., :c2]
        return x, y

    def postprocess(self, x, y):
        c1, c2 = x.shape[-1], y.shape[-1]
        x, y = x.to(self.device), y.to(self.device)
        x = x * self.std_inputs[..., :c1] + self.mean_inputs[..., :c1]
        y = y * self.std_targets[..., :c2] + self.mean_targets[..., :c2]
        return x, y

    def compute_mean_std(self, dataset, batch_size):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=12)

        n_samples = 0
        mean_inputs, mean_targets = 0., 0.
        var_inputs, var_targets = 0., 0. 
        for inputs, targets in tqdm(loader, desc="Computing mean and std"):
            b, c1, c2 = inputs.size(0), inputs.size(-1), targets.size(-1)
            inputs = inputs.view(b, -1, c1)
            targets = targets.view(b, -1, c2)

            batch_mean = inputs.mean(dim=1)
            batch_var = inputs.var(dim=(0,1), unbiased=False)
            mean_inputs += batch_mean.sum(0)
            var_inputs += batch_var * b

            batch_mean = targets.mean(dim=1)
            batch_var = targets.var(dim=(0,1), unbiased=False)
            mean_targets += batch_mean.sum(0)
            var_targets += batch_var * b

            n_samples += b

        mean_inputs = mean_inputs / n_samples
        mean_targets = mean_targets / n_samples
        var_inputs = var_inputs / n_samples
        var_targets = var_targets / n_samples

        std_inputs = var_inputs ** 0.5
        std_targets = var_targets ** 0.5

        return mean_inputs, mean_targets, std_inputs, std_targets


class RangeNormalizer(object):
    def __init__(self, dataset, device, batch_size=512, is_save=True):
        super(RangeNormalizer, self).__init__()
        self.device = device
        dataset_dir = dataset.dataset_dir

        if is_save:
            try:
                self.max_inputs, self.max_targets = torch.load(os.path.join(dataset_dir, "max.pt"))
            except:
                self.max_inputs, self.max_targets = self.compute_max(dataset, batch_size)
                torch.save((self.max_inputs, self.max_targets), os.path.join(dataset_dir, "max.pt"))
        else:
            self.max_inputs, self.max_targets = self.compute_max(dataset, batch_size)

        self.max_inputs = self.max_inputs.to(device)
        self.max_targets = self.max_targets.to(device)
        self.max_inputs = torch.where(self.max_inputs==0, torch.ones_like(self.max_inputs), self.max_inputs)
        self.max_targets = torch.where(self.max_targets==0, torch.ones_like(self.max_targets), self.max_targets)

    def preprocess(self, x, y):
        c1, c2 = x.shape[-1], y.shape[-1]
        x, y = x.to(self.device), y.to(self.device)
        x = x / self.max_inputs[..., :c1]
        y = y / self.max_targets[..., :c2]
        return x, y

    def postprocess(self, x, y):
        c1, c2 = x.shape[-1], y.shape[-1]
        x, y = x.to(self.device), y.to(self.device)
        x = x * self.max_inputs[..., :c1]
        y = y * self.max_targets[..., :c2]
        return x, y

    def compute_max(self, dataset, batch_size):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=12)

        max_inputs, max_targets = None, None
        for inputs, targets in tqdm(loader, desc="Computing max"):
            c1, c2 = inputs.size(-1), targets.size(-1)
            inputs = inputs.view(-1, c1)
            targets = targets.view(-1, c2)

            batch_max_inputs = inputs.abs().max(dim=0)[0]
            batch_max_targets = targets.abs().max(dim=0)[0]

            if max_inputs is None:
                max_inputs = batch_max_inputs
                max_targets = batch_max_targets
            else:
                max_inputs = torch.max(max_inputs, batch_max_inputs)
                max_targets = torch.max(max_targets, batch_max_targets)

        return max_inputs, max_targets
