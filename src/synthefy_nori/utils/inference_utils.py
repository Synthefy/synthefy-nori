from __future__ import annotations

from torch.utils.data import DistributedSampler



class NonPaddingDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.num_samples = len(range(rank, len(dataset), num_replicas))
        self.total_size = len(dataset)

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)

def swap_rows_back(tensor, indices):
    """

    Args:
        tensor (torch.Tensor):
        indices (list|torch.Tensor):

    Returns:
        torch.Tensor:
    """
    inverse_indices = [0] * len(indices)
    for i, idx in enumerate(indices):
        inverse_indices[idx] = i
    return tensor[inverse_indices]
