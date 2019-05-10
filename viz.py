import torch
import numpy as np
from torch.autograd import Variable
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.pylab as P

epsilon = 1e-10
def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def plot_cam(attr, xi, cmap='jet', alpha=0.5):
    attr -= attr.min()
    attr /= (attr.max() + 1e-20)

    plt.imshow(xi)
    plt.imshow(attr, alpha=alpha, cmap=cmap)

def plot_bbox(bboxes, xi, linewidth=1):
    ax = plt.gca()
    ax.imshow(xi)

    if not isinstance(bboxes[0], list):
        bboxes = [bboxes]

    for bbox in bboxes:
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                 linewidth=linewidth, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    ax.axis('off')

def ShowGrayscaleImage(im, title='', ax=None):
    if ax is None:
        P.figure()
        P.axis('off')
    P.imshow(im, cmap=P.cm.gray, vmin=0, vmax=1)
    P.title(title)

def VisualizeImageGrayscale(imgs):
    batch_size, n_chs, height, width = imgs.shape
    if isinstance(imgs, Variable):
        imgs = imgs.data
    imgs = torch.abs(imgs).sum(dim=1)
    imgs = imgs.view(batch_size, -1)
    vmax = torch.max(imgs, dim=1, keepdim=True)[0]
    vmin = torch.min(imgs, dim=1, keepdim=True)[0]
    imgs = torch.clamp((imgs - vmin) / (vmax - vmin), 0, 1)
    imgs = imgs.view(batch_size, height, width)
    return imgs

def agg_default(x):
    if x.ndim == 4:
        return np.abs(x).sum(1)
    elif x.ndim == 3:
        return np.abs(x).sum(0)

def jaccard(x1, x2, frac=0.1):
    topk1 = np.argsort(-x1)[:int(x1.size * frac)]
    topk2 = np.argsort(-x2)[:int(x2.size * frac)]
    intersect = np.intersect1d(topk1, topk2)
    union = np.union1d(topk1, topk2)
    return len(intersect)/len(union)

def clip(x, top_clip=True):
    if x.ndim == 3:
        batch_size, height, width = x.shape
        x = x.reshape(batch_size, -1)
        if top_clip:
            vmax = np.percentile(x, 99, axis=1, keepdims=True)
        else:
            vmax = np.max(x, axis=1, keepdims=True)
        vmin = np.min(x, axis=1, keepdims=True)
        vdiff = vmax - vmin
        for i, v in enumerate(vdiff):
            v = max(0, np.abs(v))
            if np.abs(v) < epsilon:
                x[i] = np.zeros_like(x[i])
            else:
                x[i] = np.clip((x[i] - vmin[i]) / v, 0, 1)
        x = x.reshape(batch_size, height, width)
    elif x.ndim == 2:
        height, width = x.shape
        x = x.ravel()
        x = np.nan_to_num(x)
        vmax = np.percentile(x, 99) if top_clip else np.max(x)
        vmin = np.min(x)
        vdiff = max(0, np.abs(vmax - vmin))
        if np.abs(vdiff) < epsilon:
            x = np.zeros_like(x)
        else:
            x = np.clip((x - vmin) / (vmax - vmin), 0, 1)
        x = x.reshape(height, width)
    return x

def agg_clip(x, top_clip=True):
    return clip(agg_default(x), top_clip=top_clip)

def get_median_difference(saliency):
    # compute median difference
    # s = agg_clip(saliency).ravel()
    s = saliency.ravel()
    topk = np.argsort(-s)[:int(s.size * 0.02)]
    botk = np.argsort(s)[:int(s.size * 0.98)]

    top_median = np.median(s[topk])
    bot_median = np.median(s[botk])
    return top_median - bot_median

def batch_median_difference(saliency):
    # compute median difference
    # s = agg_clip(saliency).ravel()
    saliency = np.reshape(saliency, (saliency.shape[0], -1))
    batch_size, num_pixels = saliency.shape
    topk = -np.sort(-saliency, axis=1)[:, :int(num_pixels * 0.02)]
    botk = np.sort(saliency, axis=1)[:, :int(num_pixels * 0.98)]

    top_median = np.median(topk, axis=1)
    bot_median = np.median(botk, axis=1)
    return top_median - bot_median

def sparsity(saliency):
    saliency = np.reshape(saliency, (saliency.shape[0], -1))
    batch_size, num_pixels = saliency.shape

    sparse_count = np.sum((np.abs(saliency) == 0), axis=1)
#    sparse_count = np.sum((np.abs(saliency) < epsilon), axis=1)
    sparse_count = sparse_count/num_pixels
    return sparse_count