import numpy as np

from torch.utils.data import SubsetRandomSampler, Sampler

class SubsetSampler(Sampler):

    def __init__(self, indices):

        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in np.arange(len(self.indices)))

    def __len__(self):
        return len(self.indices)
