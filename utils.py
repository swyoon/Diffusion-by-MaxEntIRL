import os
import errno
import numpy as np
import torch


# Recursive mkdir
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


from sklearn.metrics import roc_auc_score
def roc_btw_arr(arr1, arr2):
    true_label = np.concatenate([np.ones_like(arr1),
                                 np.zeros_like(arr2)])
    score = np.concatenate([arr1, arr2])
    return roc_auc_score(true_label, score)


def batch_run(m, dl, device, flatten=False, method='predict', input_type='first', no_grad=True,
              submodule=None, show_tqdm=False,
              **kwargs):
    """
    m: model
    dl: dataloader
    device: device
    method: the name of a function to be called
    no_grad: use torch.no_grad if True.
    kwargs: additional argument for the method being called
    submodule: if not None, use m.submodule instead of m
    show_tqdm: if True, display tqdm progress bar
    """
    if submodule is not None:
        m = getattr(m, submodule)
    method = getattr(m, method)
    l_result = []
    dl = tqdm(dl) if show_tqdm else dl 
    for batch in dl:
        if input_type == 'first':
            x = batch[0]

        if no_grad:
            with torch.no_grad():
                if flatten:
                    x = x.view(len(x), -1)
                pred = method(x.to(device), **kwargs).detach().cpu()
        else:
            if flatten:
                x = x.view(len(x), -1)
            pred = method(x.to(device), **kwargs).detach().cpu()

        l_result.append(pred)
    return torch.cat(l_result)


def get_rank():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


def print0(*args, **kwargs):
    if get_rank() == 0:
        print(*args, **kwargs)