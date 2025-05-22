import argparse
import os
import torch
import torch.nn as nn
from typing import Tuple, Optional
import numpy as np
import math
import matplotlib.pyplot as plt


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class DataSaverHook:
    def __init__(self, store_input=True, store_output=True) -> None:
        self.store_input = store_input
        self.store_output = store_output

        self.input = None
        self.output = None
    
    def __call__(self, module, input_batch, output_batch):
        if self.store_input:
            self.input = input_batch
        
        if self.store_output:
            self.output = output_batch


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
class CenterCrop2d(nn.Module):
    """Crop the input tensor."""
    def __init__(self, crop_px: int):
        super().__init__()
        self.crop = crop_px

    def forward(self, x):
        if self.crop == 0:
            return x
        return x[:, :, self.crop:-self.crop, self.crop:-self.crop]

def deconv_to_pixelshuffle(
        deconv: nn.ConvTranspose2d
) -> Tuple[nn.Sequential, nn.Conv2d, Optional[int]]:
    """Convert ConvTranspose2d ➜ Conv2d + PixelShuffle + CenterCrop."""
    s   = deconv.stride[0]
    kT  = deconv.kernel_size[0]
    pT  = deconv.padding[0]
    print(f"stride={s}, kernel={kT}, padding={pT}")
    k_l = kT // s                      # low‑res kernel
    p_l = k_l - pT                 # derived low‑res padding
    if p_l < 0:
        raise ValueError("PixelShuffle padding < 0! ")


    conv_lr = nn.Conv2d(
        deconv.in_channels, deconv.out_channels * s**2,
        kernel_size=k_l, stride=1, padding=p_l,
        bias=deconv.bias is not None)
    pix = nn.PixelShuffle(s)
    crop_px = p_l                   
    crop = CenterCrop2d(crop_px) if crop_px else nn.Identity()

    seq = nn.Sequential(conv_lr, crop, pix)
    # weight rearrangement
    with torch.no_grad():
        for cout in range(deconv.out_channels):
            for cin in range(deconv.in_channels):
                for i0 in range(s):
                    for j0 in range(s):
                        for u in range(k_l):
                            for v in range(k_l):
                                kH = (k_l - 1 - u) * s + i0
                                kW = (k_l - 1 - v) * s + j0
                                q = cout * s**2 + i0 * s + j0
                                conv_lr.weight[q, cin, u, v] = (
                                    deconv.weight[cin, cout, kH, kW])
        if deconv.bias is not None:
            conv_lr.bias.view(deconv.out_channels, s**2)\
                   .copy_(deconv.bias.view(-1, 1).expand(-1, s**2))

    return seq, conv_lr, (crop_px if crop_px else None)


def plot_masks(masks: torch.Tensor, path: str):
    n = masks.size(0)
    # choose layout
    if n > 1:
        cols = 4
        rows = (n + cols - 1) // cols
        fig, axs = plt.subplots(rows, cols, figsize=(15, 6), facecolor='w', edgecolor='k')
        axs = axs.flatten()
        fig.subplots_adjust(hspace=.5, wspace=.4)

        for i, ax in enumerate(axs):
            ax.axis('off')
            if i < n:
                # show the mask, capture the AxesImage
                im = ax.imshow(masks[i].detach().cpu().numpy(), cmap='viridis')
                # add a small colorbar next to this subplot
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # hide any extra axes
        for j in range(n, len(axs)):
            axs[j].set_visible(False)

        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close(fig)

    else:
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(masks[0, 0].cpu().numpy(), cmap='viridis')
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close(fig)



def plot_activation_outliers_per_channel(
    activations,
    threshold=None,
    path='activation_outliers.png',
    bar_color='steelblue',
    outlier_color='tomato',
    threshold_color='k'
):
    """
    Plots per-channel maximum activation values for each sample in the batch,
    highlighting channels considered outliers.

    activations: torch.Tensor or np.ndarray of shape (batch, channels, h, w)
    threshold: float or sequence of floats, optional
               If None, computed as global mean+2*std over all batch+channels.
    """
    # to numpy
    if isinstance(activations, torch.Tensor):
        activations = activations.detach().cpu().numpy()

    b, c, h, w = activations.shape
    flat = activations.reshape(b, c, -1)       # (batch, channels, spatial)
    max_per = np.max(np.abs(flat), axis=2)     # (batch, channels)

    # compute thresholds
    if threshold is None:
        vals = max_per.ravel()
        th_val = vals.mean() + 2 * vals.std()
        thresholds = np.full(b, th_val)
    else:
        arr = np.atleast_1d(threshold).astype(float)
        if arr.size == 1:
            thresholds = np.full(b, arr.item())
        elif arr.size == b:
            thresholds = arr
        else:
            raise ValueError("`threshold` must be None, a float, or length==batch")

    # plot
    fig, axes = plt.subplots(1, b, figsize=(6*b, 4), squeeze=False)
    axes = axes[0]

    for i, ax in enumerate(axes):
        ch_max = max_per[i]
        th = thresholds[i]

        # bar plot
        ax.bar(np.arange(c), ch_max, color=bar_color, label='max activation', alpha=0.8)

        # threshold line
        # ax.axhline(th, color=threshold_color, linestyle='--', linewidth=1.2,
                #    label=f'Th={th:.2f}')

        # outlier points
        mask = ch_max > th
        # ax.scatter(np.where(mask)[0], ch_max[mask],
        #            color=outlier_color, marker='*', s=100, label='outlier')

        # y-axis extend 10%
        y_max =  th * 1.4
        ax.set_ylim(0, y_max)

        # styling
        ticks = [50,100,150,200, 250]
        ax.set_xticks(ticks)            
        ax.set_xticklabels(ticks,       
                   fontsize=18)
        ax.tick_params(axis="both",labelsize=18)
        ax.set_xlabel('Channels',fontsize=20)
        ax.set_ylabel('Max |Activation|',fontsize=20)
        ax.set_title(f'Batch {i}')
        ax.grid(axis='y', linestyle='--', alpha=0.6)

        # legend
        # ax.legend(loc='upper right', fontsize=18)

    plt.tight_layout()
    plt.savefig(path, dpi=400)
    plt.show()
    plt.close(fig)



def plot_weight_3d(weight, path='weight_3d.png', elev=30, azim=45, cmap='viridis'):
    """
    Plots a 3D surface of a layer's weights.
    - For Conv layers (4D): weight shape (out_channels, in_channels, kH, kW). 
      We average over the spatial dims, yielding shape (out_channels, in_channels).
    - For Linear layers (2D): weight shape (out_features, in_features).
    
    X axis: input channel index
    Y axis: output channel index
    Z axis: weight value (averaged for conv, direct for linear)
    """
    # Convert PyTorch tensor to numpy
    if isinstance(weight, torch.Tensor):
        weight = weight.detach().cpu().numpy()
    
    # Handle dimensions
    if weight.ndim == 4:
        weight_mat = weight.mean(axis=(2,3))
    elif weight.ndim == 2:
        weight_mat = weight
    else:
        raise ValueError("weight must be 2D (linear) or 4D (conv).")

    out_c, in_c = weight_mat.shape
    x = np.arange(in_c)
    y = np.arange(out_c)
    X, Y = np.meshgrid(x, y)
    Z = weight_mat

    # Create 3D surface plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cmap, edgecolor='none')
    ax.set_xlabel('Input channel')
    ax.set_ylabel('Output channel')
    ax.set_zlabel('Weight value')
    ax.view_init(elev=elev, azim=azim)
    fig.colorbar(surf, shrink=0.5, aspect=10, label='Weight magnitude')
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.show()
    plt.close(fig)


def plot_masks(masks: torch.Tensor, path: str):
    n = masks.size(0)
    # choose layout
    if n > 1:
        cols = 4
        rows = (n + cols - 1) // cols
        fig, axs = plt.subplots(rows, cols, figsize=(15, 6), facecolor='w', edgecolor='k')
        axs = axs.flatten()
        fig.subplots_adjust(hspace=.5, wspace=.4)

        for i, ax in enumerate(axs):
            ax.axis('off')
            if i < n:
                # show the mask, capture the AxesImage
                plot_im = np.rot90(masks[i].detach().cpu().numpy(), k=2)
                plot_im = np.power(plot_im, 0.5)  # adjust brightness
                im = ax.imshow(plot_im, cmap='viridis')
                # add a small colorbar next to this subplot
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # hide any extra axes
        for j in range(n, len(axs)):
            axs[j].set_visible(False)

        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close(fig)

    else:
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_im = np.rot90(masks[0].detach().cpu().numpy(), k=2)
        plot_im = np.power(plot_im, 0.5)
        im = ax.imshow(plot_im, cmap='viridis')
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close(fig)

def plot_image_distribution(images: torch.Tensor, path: str):


    B, C, H, W = images.shape

    data = images.view(B, C, -1)        
    per_channel = data.permute(1, 0, 2)     
    per_channel = per_channel.reshape(C, -1).cpu().numpy()  

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    ax = axes[0]
    bins = 100
    for ch in range(C):
        ax.hist(per_channel[ch], bins=bins, alpha=0.4, label=f'channel {ch}')
    ax.set_title('Per-Channel Histogram')
    ax.set_xlabel('Pixel value')
    ax.set_ylabel('Frequency')
    ax.legend()

    ax = axes[1]
    ax.boxplot([per_channel[ch] for ch in range(C)],
               labels=[f'c{ch}' for ch in range(C)],
               showfliers=True)  
    ax.set_title('Per-Channel Boxplot (Outliers shown)')
    ax.set_ylabel('Pixel value')

    plt.tight_layout()
    plt.savefig(path, dpi=300)

    plt.show()
    plt.close(fig)


def analyze_and_save_distribution(img: torch.Tensor,
                                   path: str,
                                   bins: int = 50,
                                   max_cols: int = 8,
                                   iqr_factor: float = 1.5,
                                   top_k: int = 8,
                                   clip_ratio: float = 0.0):
    """
    Compute per-channel outlier statistics for a batch of images and save a grid of histograms
    for the top_k most “outlier-heavy” channels to the given path. Optionally clip extremes before plotting.

    Args:
        img (torch.Tensor): Input tensor of shape (B, C, H, W).
        path (str): File path for saving the figure (e.g., "output/distribution.png").
        bins (int): Number of histogram bins per subplot.
        max_cols (int): Max number of columns in the grid.
        iqr_factor (float): Multiplier for IQR when defining outlier thresholds.
        top_k (int): Number of channels with the most outliers to plot.
        clip_ratio (float): Fraction in [0, 0.5] to clip from each tail (e.g., 0.05 → clip below 5% and above 95%).

    Returns:
        List[Tuple[int, int, float, float, float]]: Sorted list of tuples
            (channel_index, outlier_count, min, median, max), descending by outlier_count.
    """
    # Ensure output directory exists
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)

    B, C, H, W = img.shape
    # Flatten to (C, B*H*W)
    data = img.view(B, C, -1).permute(1, 0, 2).reshape(C, -1).cpu().detach().numpy()

    # Summarize outliers per channel
    summaries = []
    for ch in range(C):
        arr = data[ch]
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        low, high = q1 - iqr_factor * iqr, q3 + iqr_factor * iqr
        outlier_count = int(((arr < low) | (arr > high)).sum())
        summaries.append((ch, outlier_count, float(arr.min()), float(np.median(arr)), float(arr.max())))
    summaries.sort(key=lambda x: x[1], reverse=True)

    # Select top_k channels
    top_channels = [ch for ch, *_ in summaries[:min(top_k, C)]]

    # Prepare grid for histograms
    n_cols = min(max_cols, len(top_channels))
    n_rows = math.ceil(len(top_channels) / n_cols)
    # fig, axes = plt.subplots(n_rows, n_cols,
    #                          figsize=(n_cols * 2, n_rows * 1.8),
    #                          squeeze=False)
    fig, axes = plt.subplots(1, B, figsize=(6*B, 4), squeeze=False)
    for idx, ch in enumerate(top_channels):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]

        arr = data[ch]
        if clip_ratio > 0:
            # Compute symmetric quantile bounds
            lower_pct = clip_ratio * 100
            upper_pct = (1 - clip_ratio) * 100
            low_clip, high_clip = np.percentile(arr, [lower_pct, upper_pct])
            arr = np.clip(arr, low_clip, high_clip)

        ax.hist(arr, bins=bins,  color='steelblue', edgecolor='lightgray',  linewidth=0.2, alpha=0.8)
        ax.set_title(f'No. {ch}' + (f' (clipped @ {clip_ratio:.2f})' if clip_ratio > 0 else ''), fontsize=6)
        ax.tick_params(axis='both', labelsize=18)
        ax.set_ylabel("Frequency",fontsize=20)
        ax.set_xlabel("Activation Value",fontsize=20)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
    # Remove empty subplots
    total_plots = n_rows * n_cols
    for empty in range(len(top_channels), total_plots):
        r, c = divmod(empty, n_cols)
        fig.delaxes(axes[r][c])

    plt.tight_layout()
    fig.savefig(path, dpi=400)
    plt.close(fig)

    return summaries



def plot_outlier_3d(batch: torch.Tensor, k: int, path: str, iqr_factor: float = 1.5):
    """
    For a batch tensor of shape (B, C, H, W), plot 3D surface plots for the first k channels
    for each sample in the batch highlighting outliers after absolute value transformation.
    The plots are saved as PNG files under the specified directory path.

    Args:
        batch (torch.Tensor): Input tensor with shape (B, C, H, W).
        k (int): Number of channels to plot (plots channels [0, k-1]).
        path (str): Directory path where plots will be saved.
        iqr_factor (float): Multiplier for IQR to define outlier threshold.
    """
    # Ensure output directory exists
    os.makedirs(path, exist_ok=True)

    # Convert to numpy and take absolute value
    data = batch.abs().cpu().detach().numpy()

    B, C, H, W = data.shape
    k = min(k, C)

    # Prepare grid coordinates
    X, Y = np.meshgrid(np.arange(W), np.arange(H))

    for c in range(k):
        for b in range(B):
            Z = data[b, c]
            # Compute IQR-based threshold
            q1, q3 = np.percentile(Z, [25, 75])
            iqr = q3 - q1
            threshold = q3 + iqr_factor * iqr
            # Identify outliers
            outlier_mask = Z > threshold
            Xo = X[outlier_mask]
            Yo = Y[outlier_mask]
            Zo = Z[outlier_mask]

            # Create 3D plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, Z, cmap='viridis')
            ax.scatter(Xo, Yo, Zo, marker='x', s=20, c='red')
            ax.set_title(f'Channel {c} Sample {b} (Outliers > {threshold:.2f})')
            ax.set_xlabel('Width Index (x)')
            ax.set_ylabel('Height Index (y)')
            ax.set_zlabel('Absolute Value')

            # Save figure
            filename = os.path.join(path, f"channel{c}_sample{b}.png")
            fig.savefig(filename, dpi=200, bbox_inches='tight')
            plt.close(fig)



def plot_outlier_3d_bar(batch: torch.Tensor, k: int, path: str, iqr_factor: float = 1.5, th = 0.1, z_lim = None):
    """
    For a batch tensor of shape (B, C, H, W), create 3D bar charts for the first k channels
    of each sample, highlighting outliers after absolute value transformation.
    The charts are saved as PNG files under the specified directory path.

    Args:
        batch (torch.Tensor): Input tensor with shape (B, C, H, W).
        k (int): Number of channels to plot (channels [0, k-1]).
        path (str): Directory path where charts will be saved.
        iqr_factor (float): Multiplier for IQR to define outlier threshold.
    """
    os.makedirs(path, exist_ok=True)

    # Convert to numpy and take absolute values
    data = batch.abs().cpu().detach().numpy()
    B, C, H, W = data.shape
    k = min(k, C)

    # Prepare grid coordinates
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    zeros = np.zeros_like(X_flat)

    for c in range(k):
        for b in range(B):
            Z = data[b, c].flatten()
            # IQR-based threshold
            q1, q3 = np.percentile(Z, [25, 75])
            threshold = q3 + iqr_factor * (q3 - q1)
            threshold = th
            mask_out = Z > threshold

            # Separate normal and outlier values
            X_norm = X_flat[~mask_out]; Y_norm = Y_flat[~mask_out]; Z_norm = Z[~mask_out]
            X_out = X_flat[mask_out]; Y_out = Y_flat[mask_out]; Z_out = Z[mask_out]

            # print(X_norm.shape, X_out.shape)
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            # Plot normal bars with lower opacity
            if X_norm.size > 0:
                ax.bar3d(X_norm, Y_norm, zeros[~mask_out], 1, 1, Z_norm, alpha=0.6)
            # Plot outlier bars with full opacity
            if X_out.size > 0:
                ax.bar3d(X_out, Y_out, zeros[mask_out], 1, 1, Z_out, alpha=1.0)

            if z_lim is not None:
                ax.set_zlim(z_lim)
            ax.set_title(f'3D Bar: Channel {c} Sample {b} (Outliers > {threshold:.2f})')
            ax.set_xlabel('Width Index (x)')
            ax.set_ylabel('Height Index (y)')
            ax.set_zlabel('Absolute Value')

            filename = os.path.join(path, f"3dbar_channel{c}_sample{b}.png")
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)

def plot_batch_channel_abs_max_outlier(batch: torch.Tensor, path: str, th: float = 1.5, y_lim: tuple = None):
    """
    For a batch tensor of shape (B, C, H, W), plot per-sample channel-wise absolute max values
    and highlight outlier channels. Saves one bar chart per sample.

    Args:
        batch (torch.Tensor): Input tensor of shape (B, C, H, W).
        path (str): Directory where plots will be saved.
        iqr_factor (float): Multiplier for IQR to define outlier threshold.
    """
    os.makedirs(path, exist_ok=True)
    B, C, H, W = batch.shape

    # Compute per-sample, per-channel absolute max
    # Shape: (B, C)
    abs_max = batch.abs().view(B, C, -1).max(dim=2).values.cpu().detach().numpy()

    for b in range(B):
        vals = abs_max[b]
        # Compute IQR threshold on this sample
        # q1, q3 = np.percentile(vals, [25, 75])
        threshold = th

        indices = np.arange(C)
        mask_out = vals > threshold

        fig, ax = plt.subplots(figsize=(8, 4))
        # Normal channels
        ax.bar(indices[~mask_out], vals[~mask_out], alpha=0.7, label='Normal')
        # Outlier channels
        ax.bar(indices[mask_out],  vals[mask_out],  alpha=1.0, label='Outlier')
        # Threshold line
        ax.axhline(threshold, linestyle='--', label=f'Threshold = {threshold:.2f}')

        if y_lim is not None:
            ax.set_ylim(y_lim)

        # ax.set_xticks(indices)
        ax.set_xlabel('Channel Index')
        ax.set_ylabel('Abs Max Value')
        ax.set_title(f'Sample {b}: Channel-wise Abs Max with Outliers')
        ax.legend()

        save_path = os.path.join(path, f"sample{b}_channel_abs_max_outlier.png")
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)


