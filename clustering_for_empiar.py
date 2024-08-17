from __future__ import print_function, division

import numpy as np
import pandas as pd
import mrcfile
import sys
import os
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
from matplotlib import cm
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data
import torchvision

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from sklearn import metrics

from astropy.stats import circcorrcoef
from astropy import units as u

import src.models as models


def load_images(path):
    images = []
    skipped_files = 0

    if os.path.isdir(path):
        for file_name in os.listdir(path):
            file_path = os.path.join(path, file_name)
            if file_path.endswith('.mrc'):
                try:
                    with mrcfile.open(file_path, permissive=True) as mrc:
                        img_data = mrc.data
                        if not np.isnan(img_data).any():
                            images.append(img_data)
                        else:
                            print(f"Skipping {file_path} due to NaN values")
                            skipped_files += 1
                except Exception as e:
                    print(f"Error reading {file_path}: {str(e)}")
                    skipped_files += 1

    print(f"Loaded {len(images)} images, skipped {skipped_files} files")
    return np.array(images) if images else np.array([])

# get_latent() extracts latent representations (z_content), rotation (theta_mu), and translation (dx) from the input images using a trained encoder model. This is based on the selected inference types (t_inf and r_inf), such as unimodal or attention-based inferences.
def get_latent(x, y, encoder_model, t_inf, r_inf, device):
    """
    Arguments
        x: base coordinates of the pixels, not rotated or translated
        y: input
        encoder_model: the encoder model
        t_inf: translation inference which can be 'unimodal' or 'attention'
        r_inf: rotation inference which can be 'unimodal' or 'attention' or 'attention+offsets'
        device: int
    Return
        z_content: rotation-translation-invariant representations
        theta_mu: predicted rotation for the object
        dx: predicted translation for the object
    """
    b = y.size(0)
    btw_pixels_space = (x[1, 0] - x[0, 0]).cpu().numpy()
    x = x.expand(b, x.size(0), x.size(1)).to(device)
    y = y.to(device)

    if t_inf == 'unimodal' and r_inf == 'unimodal':
        with torch.no_grad():
            y = y.view(b, -1)
            z_mu, z_logstd = encoder_model(y)
            z_std = torch.exp(z_logstd)
            z_dim = z_mu.size(1)

            # z[0] is the rotation
            theta_mu = z_mu[:, 0].unsqueeze(1)

            dx_mu = z_mu[:, 1:3]
            dx = dx_mu

            z_content = torch.cat((z_mu[:, 3:], z_std[:, 3:]), dim=1)

    elif t_inf == 'attention' and r_inf == 'unimodal':
        with torch.no_grad():
            attn, sampled_attn, theta_vals, z_vals = encoder_model(y, device)

            # getting most probable t
            val, ind1 = attn.view(attn.shape[0], -1).max(1)
            ind0 = torch.arange(ind1.shape[0])

            z_vals = z_vals.view(z_vals.shape[0], z_vals.shape[1], -1)
            theta_vals = theta_vals.view(theta_vals.shape[0], theta_vals.shape[1], -1)

            z_dim = z_vals.size(1) // 2
            z_mu = z_vals[:, :z_dim, ]
            z_logstd = z_vals[:, z_dim:, ]
            z_std = torch.exp(z_logstd)

            # selecting z_values from the most probable t
            z_mu = z_mu[ind0, :, ind1]
            z_std = z_std[ind0, :, ind1]
            z_content = torch.cat((z_mu, z_std), dim=1)

            attn_softmax = F.softmax(attn.view(b, -1), dim=1).unsqueeze(2)

            attn_dim = attn.shape[3]
            if attn_dim % 2:
                x_grid = np.arange(-btw_pixels_space * (attn_dim // 2), btw_pixels_space * (attn_dim // 2 + 1),
                                   btw_pixels_space)
                y_grid = np.arange(-btw_pixels_space * (attn_dim // 2), btw_pixels_space * (attn_dim // 2 + 1),
                                   btw_pixels_space)[::-1]
            else:
                x_grid = np.arange(-btw_pixels_space * (attn_dim // 2), btw_pixels_space * (attn_dim // 2),
                                   btw_pixels_space)
                y_grid = np.arange(-btw_pixels_space * (attn_dim // 2), btw_pixels_space * (attn_dim // 2),
                                   btw_pixels_space)[::-1]
            x_0, x_1 = np.meshgrid(x_grid, y_grid)
            x_coord_translate = np.stack([x_0.ravel(), x_1.ravel()], 1)
            x_coord_translate = torch.from_numpy(x_coord_translate).float().to(device)
            x_coord_translate = x_coord_translate.expand(b, x_coord_translate.size(0), x_coord_translate.size(1))
            x_coord_translate = x_coord_translate.transpose(1, 2)
            dx = torch.bmm(x_coord_translate, attn_softmax).squeeze(2)

            # selecting theta_means from the most probable t
            theta_mu = theta_vals[ind0, 0:1, ind1]

    else:
        with torch.no_grad():
            attn, _, _, _, _, theta_vals, z_vals = encoder_model(y, device)

            # getting most probable t_r
            val, ind1 = attn.view(attn.shape[0], -1).max(1)
            ind0 = torch.arange(ind1.shape[0])

            z_vals = z_vals.view(z_vals.shape[0], z_vals.shape[1], -1)
            theta_vals = theta_vals.view(theta_vals.shape[0], theta_vals.shape[1], -1)

            z_dim = z_vals.size(1) // 2
            z_mu = z_vals[:, :z_dim, ]
            z_logstd = z_vals[:, z_dim:, ]
            z_std = torch.exp(z_logstd)

            # selecting z_values from the most probable t_r
            z_mu = z_mu[ind0, :, ind1]
            z_std = z_std[ind0, :, ind1]
            z_content = torch.cat((z_mu, z_std), dim=1)

            attn_softmax = F.softmax(attn.view(b, -1), dim=1).view(attn.shape).sum(1).view(b, -1).unsqueeze(2)

            attn_dim = attn.shape[3]
            if attn_dim % 2:
                x_grid = np.arange(-btw_pixels_space * (attn_dim // 2), btw_pixels_space * (attn_dim // 2 + 1),
                                   btw_pixels_space)
                y_grid = np.arange(-btw_pixels_space * (attn_dim // 2), btw_pixels_space * (attn_dim // 2 + 1),
                                   btw_pixels_space)[::-1]
            else:
                x_grid = np.arange(-btw_pixels_space * (attn_dim // 2), btw_pixels_space * (attn_dim // 2),
                                   btw_pixels_space)
                y_grid = np.arange(-btw_pixels_space * (attn_dim // 2), btw_pixels_space * (attn_dim // 2),
                                   btw_pixels_space)[::-1]
            x_0, x_1 = np.meshgrid(x_grid, y_grid)
            x_coord_translate = np.stack([x_0.ravel(), x_1.ravel()], 1)
            x_coord_translate = torch.from_numpy(x_coord_translate).to(device)
            x_coord_translate = x_coord_translate.expand(b, x_coord_translate.size(0), x_coord_translate.size(1))
            x_coord_translate = x_coord_translate.transpose(1, 2)
            dx = torch.bmm(x_coord_translate.type(torch.float), attn_softmax).squeeze(2)

            # selecting theta_means from the most probable t_r
            theta_mu = theta_vals[ind0, 0:1, ind1]

    # Replace NaN values with zeros
    z_content = torch.nan_to_num(z_content, nan=0.0)
    theta_mu = torch.nan_to_num(theta_mu, nan=0.0)
    dx = torch.nan_to_num(dx, nan=0.0)

    print(f"Any NaN in z_content after replacement: {torch.isnan(z_content).any()}")
    print(f"Any NaN in theta_mu after replacement: {torch.isnan(theta_mu).any()}")
    print(f"Any NaN in dx after replacement: {torch.isnan(dx).any()}")

    return z_content, theta_mu, dx


def measure_correlations(path_to_transformations, r_pred, t_pred):
    """
    Arguments
        path_to_transformation: path to the transformations file
        r_pred:predicted rotation angles
        t_pred: predicted translation values
    Return
        r_corr: circular rotatation correlation
        t_corr: Pearson correaltion coefficient for translations over x and y
    """
    test_transforms = np.load(path_to_transformations)
    rot_val = test_transforms[:, 0].reshape(test_transforms.shape[0], 1)
    t_val = test_transforms[:, 1:].reshape(test_transforms.shape[0], 2)

    r_corr = circcorrcoef(rot_val, r_pred.numpy())
    x_corr = np.corrcoef(t_val[:, 0], t_pred.numpy()[:, 0])[0][1]
    y_corr = np.corrcoef(t_val[:, 1], t_pred.numpy()[:, 1])[0][1]
    t_corr = [x_corr, y_corr]

    return r_corr, t_corr


def main():
    import argparse

    parser = argparse.ArgumentParser('Clustering particles...')

    parser.add_argument('-z', '--z-dim', type=int, default=2, help='latent variable dimension (default: 2)')
    parser.add_argument('--test-path', help='path to the whole data; or path to testing data')
    parser.add_argument('--path-to-encoder', help='path to the saved encoder model')

    parser.add_argument('--path-to-transformations',
                        help='path to a single file that contains the ground-truth rotation in the first column, and the ground-truth translation values for x and y, in the second and third columns (for calculating correlations)')

    parser.add_argument('--t-inf', default='attention', choices=['unimodal', 'attention'],
                        help='unimodal | attention (default:attention)')
    parser.add_argument('--r-inf', default='attention+offsets', choices=['unimodal', 'attention', 'attention+offsets'],
                        help='unimodal | attention | attention+offsets (default:attention+offsets)')

    parser.add_argument('--clustering', default='agglomerative', choices=['agglomerative', 'k-means'],
                        help='agglomerative | k-means (default:agglomerative)')
    parser.add_argument('--n-clusters', default=10, type=int, help='Number of clusters (default:10)')

    parser.add_argument('--normalize', action='store_true', help='normalize the images before training')
    parser.add_argument('--crop', default=0, type=int, help='size of the cropped images (default:0)')

    parser.add_argument('--in-channels', type=int, default=1, help='number of channels in the images (default:0)')

    parser.add_argument('--activation', choices=['tanh', 'leakyrelu'], default='leakyrelu',
                        help='activation function (default: leakyrelu)')
    parser.add_argument('--minibatch-size', type=int, default=16, help='minibatch size (default:100)')
    parser.add_argument('-d', '--device', type=int, default=0, help='compute device to use (default:0)')

    args = parser.parse_args()

    ## load the images
    images_test = load_images(args.test_path)
    print(f"Initial shape of images_test: {images_test.shape}")
    print(f"Any NaN in images_test: {np.isnan(images_test).any()}")
    print(f"Number of NaN in images_test: {np.isnan(images_test).sum()}")

    # Remove any images that contain NaN values
    images_test = images_test[~np.isnan(images_test).any(axis=(1, 2))]
    print(f"Shape of images_test after removing NaNs: {images_test.shape}")

    if images_test.shape[0] == 0:
        print("Error: All images contain NaN values. Please check your data.")
        return

    crop = args.crop
    if crop > 0:
        images_test = image_utils.crop(images_test, crop)
        print('# cropped to:', crop, file=sys.stderr)

    n, m = images_test.shape[1:]

    # normalize the images using edges to estimate background
    if args.normalize:
        print('# normalizing particles', file=sys.stderr)
        mu = images_test.reshape(-1, m * n).mean(1)
        std = images_test.reshape(-1, m * n).std(1)
        images_test = (images_test - mu[:, np.newaxis, np.newaxis]) / std[:, np.newaxis, np.newaxis]

    # x coordinate array
    xgrid = np.linspace(-1, 1, m)
    ygrid = np.linspace(1, -1, n)
    x0, x1 = np.meshgrid(xgrid, ygrid)
    x_coord = np.stack([x0.ravel(), x1.ravel()], 1)
    x_coord = torch.from_numpy(x_coord).float()

    images_test = torch.from_numpy(images_test).float()
    in_channels = 1
    y_test = images_test.view(-1, in_channels, n, m)

    ## set the device
    d = args.device
    use_cuda = (d != -1) and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(d)
        print('# using CUDA device:', d, file=sys.stderr)
        device = torch.device("cuda:" + str(d) if use_cuda else "cpu")
    else:
        device = torch.device("cpu")

    y_test = y_test.to(device)
    x_coord = x_coord.to(device)

    data_test = torch.utils.data.TensorDataset(y_test)

    z_dim = args.z_dim
    print('# clustering with z-dim:', z_dim, file=sys.stderr)

    # defining encoder model
    t_inf = args.t_inf
    r_inf = args.r_inf

    print('# translation inference is {}'.format(t_inf), file=sys.stderr)
    print('# rotation inference is {}'.format(r_inf), file=sys.stderr)

    path_to_encoder = args.path_to_encoder
    encoder = torch.load(path_to_encoder)
    encoder = encoder.to(device)

    minibatch_size = args.minibatch_size

    # folder for writing log files
    path_prefix = '/'.join(path_to_encoder.split('/')[:-1])

    z_values = torch.empty(len(data_test), 2 * z_dim)
    print(f"Shape of z_values before NaN removal: {z_values.shape}")
    print(f"Any NaN in z_values: {torch.isnan(z_values).any()}")
    print(f"Number of NaN in z_values: {torch.isnan(z_values).sum()}")
    z_values = z_values[~torch.isnan(z_values).any(dim=1)]
    print(f"Shape of z_values after NaN removal: {z_values.shape}")

    if z_values.shape[0] == 0:
        print("Error: All processed data contains NaN values. Please check your processing steps.")
        return

    # Convert to numpy array for clustering
    # Convert to numpy array for clustering
    z_values_np = z_values.detach().cpu().numpy()
    print(f"Any NaN in z_values_np: {np.isnan(z_values_np).any()}")
    print(f"Number of NaN in z_values_np: {np.isnan(z_values_np).sum()}")
    # If there are still NaN values, replace them with the mean of the column
    if np.isnan(z_values_np).any():
        column_mean = np.nanmean(z_values_np, axis=0)
        inds = np.where(np.isnan(z_values_np))
        z_values_np[inds] = np.take(column_mean, inds[1])
        print(f"NaN values replaced with column means")
        print(f"Any NaN in z_values_np after replacement: {np.isnan(z_values_np).any()}")
    tr_pred = torch.empty(len(data_test), 2)
    rot_pred = torch.empty(len(data_test), 1)

    # getting predicted z, rotation, and translation for the data
    for i in range(0, len(data_test), minibatch_size):
        try:
            y = data_test[i:i + minibatch_size]
            y = torch.stack(y, dim=0).squeeze(0).to(device)

            a, b, c = get_latent(x_coord, y, encoder, t_inf, r_inf, device)

            z_values[i:i + minibatch_size] = a.cpu()
            rot_pred[i:i + minibatch_size] = b.cpu()
            tr_pred[i:i + minibatch_size] = c.cpu()
        except Exception as e:
            print(f"Error processing batch {i}: {str(e)}")
            continue

    n_clusters = args.n_clusters
    if args.clustering == 'agglomerative':
        ac = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward', compute_full_tree=True)
        cluster = ac.fit_predict(z_values_np)
    elif args.clustering == 'k-means':
        km = KMeans(n_clusters, n_init=100).fit(z_values_np)
        cluster = km.predict(z_values_np)

    if args.path_to_transformations:
        rot_corr, tr_corr = measure_correlations(args.path_to_transformations, rot_pred, tr_pred)
        rot_corr, tr_corr = measure_correlations(args.path_to_transformations, rot_pred, tr_pred)

    # saving tsne figure
    print('# saving tsne figure ... ', file=sys.stderr)
    n_samples = z_values.shape[0]
    perplexity = min(30, n_samples - 1)  # Ensure perplexity is less than n_samples
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=200.0, init='random', n_iter=1000)
    tsne_result = tsne.fit_transform(z_values.detach().cpu().numpy())

    plt.figure(figsize=(10, 10))

    cmap = plt.cm.rainbow
    norm = colors.BoundaryNorm(np.arange(0, n_clusters + 1, 1), cmap.N)

    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=cluster, cmap=cmap, norm=norm, s=50)

    # to modify size of the colorbar
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.2)

    # to make the number on the colorbar centered
    cb = plt.colorbar(cax=cax)
    labels = np.arange(0, n_clusters, 1)
    loc = labels + .5
    cb.set_ticks(loc)
    cb.set_ticklabels(labels)

    plt.title(f't-SNE visualization of {n_samples} samples')
    plt.savefig(path_prefix + "/tsne.jpg")


if __name__ == '__main__':
    main()
