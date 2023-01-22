import torch


def one_hot(index: torch.Tensor, n_cat: int) -> torch.Tensor:
    """One hot a tensor of categories."""
    loc = (torch.gt(index, n_cat-1) | torch.lt(index, 0))
    loc = loc.reshape(-1).tolist()
    index2 = index.clone()
    index2[loc,:] = 0
    onehot = torch.zeros(index2.size(0), n_cat, device=index2.device)
    onehot.scatter_(1, index2.type(torch.long), 1)
    onehot.type(torch.float32)
    onehot[loc,:] = 0
    return onehot
