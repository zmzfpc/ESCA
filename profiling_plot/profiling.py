import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os
import numpy as np
from scipy.ndimage import maximum_filter
import gc
from concurrent.futures import ThreadPoolExecutor

# =============================================================
# Input File Structure
# The script expects the input data to be stored in Numpy `.npy` files within a specific directory structure:
#
# Base Directory:
# /root/autodl-fs/mae/output_activations_gradients/
# └── epoch{epoch_number}/
#     ├── mae_input_activation_epoch{epoch_number}_{fixed_iteration}.npy
#     ├── mae_output_activation_epoch{epoch_number}_{fixed_iteration}.npy
#     ├── mae_input_grad_epoch{epoch_number}_{fixed_iteration}.npy
#     ├── mae_output_grad_epoch{epoch_number}_{fixed_iteration}.npy
#     └── mae_weight_epoch{epoch_number}_{fixed_iteration}.npy
#
# File Descriptions:
# 1. mae_input_activation_epoch{epoch}_{itr}.npy
#    - Contains input activations for various layers of the model (e.g., norm1, attn.qkv).
#
# 2. mae_output_activation_epoch{epoch}_{itr}.npy
#    - Contains output activations for layers like Q, K, V, softmax, and FC layers.
#
# 3. mae_input_grad_epoch{epoch}_{itr}.npy
#    - Contains input gradients for specific layers of the model.
#
# 4. mae_output_grad_epoch{epoch}_{itr}.npy
#    - Contains output gradients for layers such as Q, K, V, and FC layers.
#
# 5. mae_weight_epoch{epoch}_{itr}.npy
#    - Contains weights for layers such as QKV, attention projection, and FC layers.
#
# Naming Conventions:
# - {epoch_number}: The epoch index (e.g., 1, 2, 3, ...).
# - {fixed_iteration}: Fixed iteration value, currently set to "itr16".
#
# Example Structure for epoch 5:
# /root/autodl-fs/mae/output_activations_gradients/
# └── epoch5/
#     ├── mae_input_activation_epoch5_itr16.npy
#     ├── mae_output_activation_epoch5_itr16.npy
#     ├── mae_input_grad_epoch5_itr16.npy
#     ├── mae_output_grad_epoch5_itr16.npy
#     └── mae_weight_epoch5_itr16.npy
# =============================================================


# =============================================================
# - profiling.py: Main script for profiling QKV, FC1, and FC2 blocks.
# - Dependencies:
#   * Numpy: For numerical computations and data handling.
#   * Matplotlib: For visualization of 3D bar charts and distributions.
#   * Scipy: For applying maximum pooling.
#   * Argparse: For parsing command-line arguments.
#   * Concurrent.Futures: For parallelizing 3D plotting.
# - Input:
#   * Numpy files containing activation and gradient data for QKV, FC1, and FC2 blocks.
# - Output:
#   * PNG images visualizing activations, gradients, weights, and weight gradients for each block.

# Usage Instructions
# Run the script with the following command:
# python profiling.py --epoch <epoch_number> --batchid <batch_index>
# Example:
# python profiling.py --epoch 5 --batchid 2
# This will profile the specified epoch and batch index, generating output images in the current directory.
# =============================================================

# Plotting functions from plot.py
# Plotting function for individual grid of distributions
# Function to plot a single 3D graph
def plot_single_3d(ax, data, title):
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    #data = max_pool(data)  # Apply max pooling before plotting
    abs_data = np.abs(data)
    max_value = abs_data.max()
    max_idx = np.unravel_index(np.argmax(abs_data), abs_data.shape)
    tokens, channels = data.shape
    x, y = np.meshgrid(np.arange(tokens), np.arange(channels), indexing='ij')
    x = x.flatten()
    y = y.flatten()
    z = np.zeros_like(x)
    dz = abs_data.flatten()
    norm = plt.Normalize(vmin=dz.min(), vmax=dz.max() * 1.1)
    cmap = cm.Reds
    colors = cmap(norm(dz))

    ax.bar3d(x, y, z, dx=0.8, dy=0.8, dz=dz, color=colors, alpha=0.8)
    ax.set_title(title)
    ax.text(max_idx[0], max_idx[1], max_value * 1.1,
            f"Max: {data[max_idx]:.2f}\n({max_idx[0]}, {max_idx[1]})",
            color='blue', fontsize=10, ha='center')

# Function to plot multiple 3D graphs in a single figure
def plot_grid_combined(data_list, titles, rows, cols, output_path):
    fig = plt.figure(figsize=(20, 15))
    axes = [fig.add_subplot(rows, cols, i + 1, projection='3d') for i in range(len(data_list))]

    with ThreadPoolExecutor() as executor:
        executor.map(lambda args: plot_single_3d(*args), zip(axes, data_list, titles))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_distribution(data, title="Data Distribution", save_name='', factor=1.1):
    data = np.array(data)
    abs_data = np.abs(data)
    max_value = abs_data.max()
    max_idx = np.unravel_index(np.argmax(abs_data), abs_data.shape)
    original_max_value = data[max_idx]
    tokens, channels = data.shape
    abs_values = abs_data.flatten()
    x, y = np.meshgrid(np.arange(tokens), np.arange(channels), indexing='ij')
    x = x.flatten()
    y = y.flatten()
    z = np.zeros_like(x)
    dz = abs_values
    norm = plt.Normalize(vmin=dz.min(), vmax=dz.max() * factor)
    cmap = cm.Reds
    colors = cmap(norm(dz))
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.bar3d(x, y, z, dx=0.8, dy=0.8, dz=dz, color=colors, alpha=0.8)
    ax.set_zlim(0, dz.max() * factor)
    ax.text(max_idx[0], max_idx[1], max_value * 1.1,
            f"Max: {original_max_value:.6g}\n({max_idx[0]}, {max_idx[1]})",
            color='blue', fontsize=10, ha='center')
    ax.scatter(max_idx[0], max_idx[1], max_value, color='blue', s=50, label="Max Value")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Value")
    ax.set_title(f"{title}")
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def plot_qkv_distribution(data, title="QKV Distribution", save_name='', factor=1.1):
    data = np.array(data)
    if 2304 in data.shape:
        split_dim = data.shape.index(2304)
        if split_dim == 0:
            q_data, k_data, v_data = np.split(data, 3, axis=0)
        elif split_dim == 1:
            q_data, k_data, v_data = np.split(data, 3, axis=1)
        else:
            raise ValueError("Data dimensions are not compatible with QKV splitting.")
    else:
        raise ValueError("Input data must have one dimension equal to 2304.")
    qkv_data = [q_data, k_data, v_data]
    titles = ["Q Distribution", "K Distribution", "V Distribution"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': '3d'})
    for ax, sub_data, sub_title in zip(axes, qkv_data, titles):
        abs_data = np.abs(sub_data)
        max_value = abs_data.max()
        max_idx = np.unravel_index(np.argmax(abs_data), abs_data.shape)
        original_max_value = sub_data[max_idx]
        tokens, channels = sub_data.shape
        abs_values = abs_data.flatten()
        x, y = np.meshgrid(np.arange(tokens), np.arange(channels), indexing='ij')
        x = x.flatten()
        y = y.flatten()
        z = np.zeros_like(x)
        dz = abs_values
        norm = plt.Normalize(vmin=dz.min(), vmax=dz.max() * factor)
        cmap = cm.Reds
        colors = cmap(norm(dz))
        ax.bar3d(x, y, z, dx=0.8, dy=0.8, dz=dz, color=colors, alpha=0.8)
        ax.set_zlim(0, dz.max() * factor)
        ax.text(max_idx[0], max_idx[1], max_value * 1.1,
                f"Max: {original_max_value:.6g}\n({max_idx[0]}, {max_idx[1]})",
                color='blue', fontsize=10, ha='center')
        ax.scatter(max_idx[0], max_idx[1], max_value, color='blue', s=50, label="Max Value")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Value")
        ax.set_title(sub_title)
        ax.legend(loc='upper right')
    fig.suptitle(title)
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

# Apply max pooling to reduce data size
def max_pool(data, pool_size=(2, 2)):
    data = data.astype(np.float32)  # Convert to float32
    pooled_data = maximum_filter(data, size=pool_size, mode='constant')
    return pooled_data[::pool_size[0], ::pool_size[1]]

# Profiling logic
def profile_block(epoch, batch_index, block_index, output_dir):
    # Data paths
    base_path = "/root/autodl-fs/mae/output_activations_gradients"
    epoch_path = f"{base_path}/epoch{epoch}"
    fixed_itr = "itr16"  # Fixed value for iteration
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    block_name = f"module.blocks.{block_index}"
    
    x1 = np.load(f"{epoch_path}/mae_input_activation_epoch{epoch}_{fixed_itr}.npy", allow_pickle=True).item()[f"{block_name}.norm1"][0][batch_index]
    x2 = np.load(f"{epoch_path}/mae_input_activation_epoch{epoch}_{fixed_itr}.npy", allow_pickle=True).item()[f"{block_name}.attn.qkv"][0][batch_index]
    
    x3 = np.load(f"{epoch_path}/mae_output_activation_epoch{epoch}_{fixed_itr}.npy", allow_pickle=True).item()[f"{block_name}.attn.qkv"][0][batch_index][:,:768] # Q
    x4 = np.load(f"{epoch_path}/mae_output_activation_epoch{epoch}_{fixed_itr}.npy", allow_pickle=True).item()[f"{block_name}.attn.qkv"][0][batch_index][:,768:1536] # K
    x5 = np.load(f"{epoch_path}/mae_output_activation_epoch{epoch}_{fixed_itr}.npy", allow_pickle=True).item()[f"{block_name}.attn.qkv"][0][batch_index][:,1536:2304] # V
    x6 = np.load(f"{epoch_path}/mae_output_activation_epoch{epoch}_{fixed_itr}.npy", allow_pickle=True).item()[f"{block_name}.attn.qk_matmul"][0][batch_index][0] # Head 0
    x7 = np.load(f"{epoch_path}/mae_output_activation_epoch{epoch}_{fixed_itr}.npy", allow_pickle=True).item()[f"{block_name}.attn.softmax"][0][batch_index][0] # Head 0
    x8 = np.load(f"{epoch_path}/mae_output_activation_epoch{epoch}_{fixed_itr}.npy", allow_pickle=True).item()[f"{block_name}.attn.attn_v_matmul"][0][batch_index]
    x9 = np.load(f"{epoch_path}/mae_output_activation_epoch{epoch}_{fixed_itr}.npy", allow_pickle=True).item()[f"{block_name}.attn.proj"][0][batch_index]
    
    y1 = np.load(f"{epoch_path}/mae_input_activation_epoch{epoch}_{fixed_itr}.npy", allow_pickle=True).item()[f"{block_name}.norm2"][0][batch_index]
    y2 = np.load(f"{epoch_path}/mae_input_activation_epoch{epoch}_{fixed_itr}.npy", allow_pickle=True).item()[f"{block_name}.mlp.fc1"][0][batch_index]
    
    y3 = np.load(f"{epoch_path}/mae_output_activation_epoch{epoch}_{fixed_itr}.npy", allow_pickle=True).item()[f"{block_name}.mlp.fc1"][0][batch_index]
    y4 = np.load(f"{epoch_path}/mae_output_activation_epoch{epoch}_{fixed_itr}.npy", allow_pickle=True).item()[f"{block_name}.mlp.act"][0][batch_index]
    y5 = np.load(f"{epoch_path}/mae_output_activation_epoch{epoch}_{fixed_itr}.npy", allow_pickle=True).item()[f"{block_name}.mlp.fc2"][0][batch_index]
    
    x2_grad = np.load(f"{epoch_path}/mae_input_grad_epoch{epoch}_{fixed_itr}.npy", allow_pickle=True).item()[f"{block_name}.attn.qkv"][0][batch_index]
    
    x3_grad = np.load(f"{epoch_path}/mae_output_grad_epoch{epoch}_{fixed_itr}.npy", allow_pickle=True).item()[f"{block_name}.attn.qkv"][0][batch_index][:,:768] # Q
    x4_grad = np.load(f"{epoch_path}/mae_output_grad_epoch{epoch}_{fixed_itr}.npy", allow_pickle=True).item()[f"{block_name}.attn.qkv"][0][batch_index][:,768:1536] # K
    x5_grad = np.load(f"{epoch_path}/mae_output_grad_epoch{epoch}_{fixed_itr}.npy", allow_pickle=True).item()[f"{block_name}.attn.qkv"][0][batch_index][:,1536:2304] # V
    x6_grad = np.load(f"{epoch_path}/mae_output_grad_epoch{epoch}_{fixed_itr}.npy", allow_pickle=True).item()[f"{block_name}.attn.qk_matmul"][0][batch_index][0] # Head 0
    x7_grad = np.load(f"{epoch_path}/mae_output_grad_epoch{epoch}_{fixed_itr}.npy", allow_pickle=True).item()[f"{block_name}.attn.softmax"][0][batch_index][0] # Head 0
    x8_grad = np.load(f"{epoch_path}/mae_output_grad_epoch{epoch}_{fixed_itr}.npy", allow_pickle=True).item()[f"{block_name}.attn.attn_v_matmul"][0][batch_index]
    x9_grad = np.load(f"{epoch_path}/mae_output_grad_epoch{epoch}_{fixed_itr}.npy", allow_pickle=True).item()[f"{block_name}.attn.proj"][0][batch_index]
    
    y2_grad = np.load(f"{epoch_path}/mae_input_grad_epoch{epoch}_{fixed_itr}.npy", allow_pickle=True).item()[f"{block_name}.mlp.fc1"][0][batch_index]
    
    y3_grad = np.load(f"{epoch_path}/mae_output_grad_epoch{epoch}_{fixed_itr}.npy", allow_pickle=True).item()[f"{block_name}.mlp.fc1"][0][batch_index]
    y4_grad = np.load(f"{epoch_path}/mae_output_grad_epoch{epoch}_{fixed_itr}.npy", allow_pickle=True).item()[f"{block_name}.mlp.act"][0][batch_index]
    y5_grad = np.load(f"{epoch_path}/mae_output_grad_epoch{epoch}_{fixed_itr}.npy", allow_pickle=True).item()[f"{block_name}.mlp.fc2"][0][batch_index]
    
    # Max pooling to reduce data size and accelerate plot function
    q_weight = max_pool(np.load(f"{epoch_path}/mae_weight_epoch{epoch}_{fixed_itr}.npy", allow_pickle=True).item()[f"{block_name}.attn.qkv"][0].T[:,:768]) # Q
    k_weight = max_pool(np.load(f"{epoch_path}/mae_weight_epoch{epoch}_{fixed_itr}.npy", allow_pickle=True).item()[f"{block_name}.attn.qkv"][0].T[:,768:1536]) # K
    v_weight = max_pool(np.load(f"{epoch_path}/mae_weight_epoch{epoch}_{fixed_itr}.npy", allow_pickle=True).item()[f"{block_name}.attn.qkv"][0].T[:,1536:2304]) # V
    attnproj_weight = max_pool(np.load(f"{epoch_path}/mae_weight_epoch{epoch}_{fixed_itr}.npy", allow_pickle=True).item()[f"{block_name}.attn.proj"][0])
    fc1_weight = max_pool(np.load(f"{epoch_path}/mae_weight_epoch{epoch}_{fixed_itr}.npy", allow_pickle=True).item()[f"{block_name}.mlp.fc1"][0])
    fc2_weight = max_pool(np.load(f"{epoch_path}/mae_weight_epoch{epoch}_{fixed_itr}.npy", allow_pickle=True).item()[f"{block_name}.mlp.fc2"][0])
    
    q_weight_grad = max_pool(x2.T @ x3_grad)
    k_weight_grad = max_pool(x2.T @ x4_grad)
    v_weight_grad = max_pool(x2.T @ x5_grad)
    attnproj_weight_grad = max_pool(x8.T @ x9_grad)
    fc1_weight_grad = max_pool(y2.T @ y3_grad)
    fc2_weight_grad =max_pool( y4.T @ y5_grad)

    # Plot data for X and Y
    x_data = [x1, x2, x3, x4, x5, x6, x7, x8, x9]
    x_titles = ["Norm1 X1", "QKV X2", "Q X3", "K X4", "V X5", "QK Matmul X6", "Softmax X7", "Attn_V Matmul X8", "Proj X9"]
    y_data = [y1, y2, y3, y4, y5]
    y_titles = ["MLP Y1", "Norm2 Y2", "FC1 Y3", "GELU Y4", "FC2 Y5"]

    plot_grid_combined(x_data, x_titles, 3, 3, f"{output_dir}/block{block_index}_x_distribution.png")
    plot_grid_combined(y_data, y_titles, 2, 3, f"{output_dir}/block{block_index}_y_distribution.png")
    
    # Combine Gradients and Weights for plotting
    grad_data = [x2_grad, x3_grad, x4_grad, x5_grad, x6_grad, x7_grad, x8_grad, x9_grad, y2_grad, y3_grad, y4_grad, y5_grad]
    grad_titles = ["QKV Input Grad", "Q Grad", "K Grad", "V Grad", "QK Matmul Grad", "Softmax Grad", "Attn_V Matmul Grad", "Proj Grad", "FC1 Input Grad", "FC1 Grad", "GELU Grad", "FC2 Grad"]
    weight_data = [q_weight, k_weight, v_weight, attnproj_weight, fc1_weight, fc2_weight]
    weight_titles = ["Q Weight", "K Weight", "V Weight", "Attn Proj Weight", "FC1 Weight", "FC2 Weight"]
    
    plot_grid_combined(grad_data, grad_titles, 4, 3, f"{output_dir}/block{block_index}_grad_distribution.png")
    plot_grid_combined(weight_data, weight_titles, 2, 3, f"{output_dir}/block{block_index}_weight_distribution.png")
    
   
    weight_grad_data = [q_weight_grad, k_weight_grad, v_weight_grad, attnproj_weight_grad, fc1_weight_grad, fc2_weight_grad]
    weight_grad_titles = ["Q Weight Grad", "K Weight Grad", "V Weight Grad", "Attn Proj Weight Grad", "FC1 Weight Grad", "FC2 Weight Grad"]

    plot_grid_combined(weight_grad_data, weight_grad_titles, 2, 3, f"{output_dir}/block{block_index}_weight_grad_distribution.png")
    
    # Clear memory
    del x1, x2, x3, x4, x5, x6, x7, x8, x9, y1, y2, y3, y4, y5
    del x2_grad, x3_grad, x4_grad, x5_grad, x6_grad, x7_grad, x8_grad, x9_grad, y2_grad, y3_grad, y4_grad, y5_grad
    del q_weight, k_weight, v_weight, attnproj_weight, fc1_weight, fc2_weight
    del q_weight_grad, k_weight_grad, v_weight_grad, attnproj_weight_grad, fc1_weight_grad, fc2_weight_grad
    gc.collect()
    
# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile QKV, FC1, and FC2 blocks.")
    parser.add_argument("--epoch", type=int, required=True, help="Specify the epoch number.")
    parser.add_argument("--batchid", type=int, required=True, help="Specify the batch index.")
    args = parser.parse_args()

    output_dir = f"./epoch{args.epoch}"
    for block_index in [1]:
        print(f"Profiling Block {block_index}...")
        gc.collect()
        profile_block(epoch=args.epoch, batch_index=args.batchid, block_index=block_index, output_dir=output_dir)

