import torch 
import numpy as np
from scipy.stats import pearsonr
from dataloaders import Dataset_visual

params = {'batch_size': 4,
          'shuffle': True,
          'num_workers': 6}

def masked_MSEloss(output, target):
    vec = (output - target)**2
    mask = target > -900.0
    loss = torch.sum(vec[mask])/torch.sum(mask)
    return loss

def masked_MSEloss_vec(output, target):
    vec = (output - target)**2
    mask = target < -900.0
    vec[mask] = 0
    loss = vec.sum(1)/(mask.sum(1).type(torch.cuda.FloatTensor))
    return loss


def full_objective(model, inputs, targets, criterion = masked_MSEloss):
    outputs = model(inputs)
    return criterion(outputs, targets)


def compute_scores(y, y_hat):
    corrs = []
    for i in range(y.shape[1]):
        val_idx = y[:,i] != -999.0
        corrs.append(pearsonr(y[val_idx, i], y_hat[val_idx, i])[0])
    return np.nanmean(corrs)

def compute_scores_dist(y, y_hat):
    corrs = []
    for i in range(y.shape[1]):
        val_idx = y[:,i] != -999.0
        corrs.append(pearsonr(y[val_idx, i], y_hat[val_idx, i])[0])
    return np.nanmean(corrs), np.asarray(corrs)


def compute_predictions(loader, model, reshape=True, stack=True, return_lag=False):
    y, y_hat = [], []
    for x_val, y_val in loader:
        neurons = y_val.size(-1)

        y_mod = model(x_val.cuda().float()).data.cpu().numpy()
        y.append(y_val.numpy())
        y_hat.append(y_mod)
    if stack:
        y, y_hat = np.vstack(y), np.vstack(y_hat) 
    return y, y_hat

def compute_predictions_input(loader, model, reshape=True, stack=True, return_lag=False):
    y, y_hat, x = [], [], []
    for x_val, y_val in loader:
        neurons = y_val.size(-1)

        y_mod = model(x_val.cuda().float()).data.cpu().numpy()
        y.append(y_val.numpy())
        y_hat.append(y_mod)
        x.append(x_val.cpu())
    if stack:
        y, y_hat = np.vstack(y), np.vstack(y_hat) 
    return y, y_hat, x

def get_responses(roi_type, roi, reshape=True, stack=True, return_lag=False):
    test_set = Dataset_visual(mode = 'test', roi_type = roi_type, roi = roi)
    test_generator = torch.utils.data.DataLoader(test_set,  **params)
    y = []
    for x_val, y_val in test_generator:
        y.append(y_val.numpy())
    if stack:
        y = np.vstack(y)
    return y