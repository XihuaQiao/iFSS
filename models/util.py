import numpy as np
import torch

def try_index(scalar_or_list, i):
    try:
        return scalar_or_list[i]
    except TypeError:
        return scalar_or_list
    

def get_lambda(alpha=1.0):
    '''Return lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    return lam


def mixup_data(x, y, alpha=1.0, lam=None, new_classes=None):
    with torch.no_grad():
        if not lam:
            lam = get_lambda(alpha)
            
        indices = np.random.permutation(x.size(0))
        if new_classes is not None:
            adaptive_lam = torch.zeros(x.shape[0], dtype=torch.float32).to(x.device)
            for i, j in enumerate(indices):
                cls_i = bool(set(np.unique(y[i].cpu().numpy())) & set(new_classes))
                cls_j = bool(set(np.unique(y[j].cpu().numpy())) & set(new_classes))
                if cls_i and cls_j:
                    adaptive_lam[i] = 0.5
                elif cls_i and not cls_j:
                    adaptive_lam[i] = max(lam, 1-lam)
                elif cls_j and not cls_i:
                    adaptive_lam[i] = min(lam, 1-lam)
                else:
                    adaptive_lam[i] = lam
            lam = adaptive_lam
        else:
            lam = torch.tensor(lam).repeat(x.shape[0]).to(x.device)
    # for i, j in enumerate(indices):
    #     print(f"{np.unique(y[i].cpu().numpy())} -- {np.unique(y[j].cpu().numpy())}")
    # print(f"lam - {lam}")
    # mixed_x = lam * x + (1 - lam) * x[indices, :]
    mixed_x = torch.einsum('b,bdhw->bdhw', lam, x) + torch.einsum('b,bdhw->bdhw', 1-lam, x[indices, :])

    y_a, y_b = y, y[indices]

    return mixed_x, y_a, y_b, lam
