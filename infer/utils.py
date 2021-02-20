import torch
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

from PIL import Image


def gkern(kernlen=21, nsig=2.5):
    """Returns a 2D Gaussian kernel."""
    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d / kern2d[kernlen // 2, kernlen // 2]

def disp_occ_grid(occ_grid, name):
    fig, axs = plt.subplots(1, 1)
    occ_grid = occ_grid.detach().numpy()[0]
    print(occ_grid.shape)
    axs.imshow(Image.fromarray(np.uint8(occ_grid * 255), 'L'), cmap='gray')
    plt.savefig(name)
    plt.close()

def disp_tensor(occ_grid):
    batch_size = occ_grid.size()[0]
    fig, axs = plt.subplots(batch_size, 1, figsize=(batch_size * 20, 40))
    for i in range(batch_size):
        axs[i].imshow(Image.fromarray(np.uint8(occ_grid[i].detach().numpy()[0] * 255), 'L'), cmap='gray')
    plt.savefig('./tmp.png')
    plt.close()

# Some of the code has been partially taken from here (https://stackoverflow.com/a/53215649)
def heatmap_accuracy(pred, gt, device):
    batch_size, num_channels, h, w = pred.size()
    # Indices of highest probability points in pred
    pred_indices = pred.view(batch_size, -1).argmax(1).view(-1, 1)
    pred_indices = torch.cat((pred_indices // w, pred_indices % w), dim=1)

    # Indices of highest probability points in gt
    gt_indices = gt.view(batch_size, -1).argmax(1).view(-1, 1)
    gt_indices = torch.cat((gt_indices // w, gt_indices % w), dim=1)

    # Compute Distance be gt and prediction
    scale = torch.tensor([0.1, 0.5])
    scale = scale.to(device)
    dist = torch.abs(pred_indices - gt_indices) * scale
    dist = torch.sqrt(torch.sum(dist ** 2, axis=1))
    return dist