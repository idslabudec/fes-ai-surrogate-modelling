import os
from os import getlogin
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

# Plot description name
date = ""
plot_description = "surrogate_models_outputs_combined"

# Setup
plt.close('all')
electrode_configs = ["concentric", "square"]
all_results = {}

fpath = "results"

csv_load_paths = [
    "simulated_data/v5_experiment_details.xlsx",
     "simulated_data/v5_experiment_details.xlsx",
]
input_cols_csvs = [
    ["cathode_diam_cm", "anode_inner_rad_cm", "anode_outer_rad_cm", "cathode_area_cm2", "anode_area_cm2",
     "interelectrode_distance_cm", "anode_inner_outer_diff_cm", "current_ma", "axon_depth_below_surface_cm"],
    ["side_cm", "interelectrode_distance_cm", "current_ma", "axon_depth_below_surface_cm", "pulse_duration_us"]
]
output_col_csv = ["cross_activation_threshold"]


class surrogate_model(nn.Module):
    def __init__(self, input_size):
        super(surrogate_model, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(input_size, 640),
            nn.Tanh(),
            nn.Linear(640, 120),
            nn.ReLU(),
            nn.Linear(120, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.hidden(x)

# Training loop for both configurations
for idx, config in enumerate(electrode_configs):
    input_cols_csv = input_cols_csvs[idx]
    csv_load_path = csv_load_paths[idx]
    sheet_name = "thigh_" + config

    df = pd.read_excel(csv_load_path, index_col=0, sheet_name=sheet_name).dropna()
    X = df[input_cols_csv].dropna().to_numpy()
    y = df[output_col_csv].dropna().to_numpy()

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    model = surrogate_model(len(input_cols_csv))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    epochs = 1000
    train_loss, val_loss, val_accuracy = [], [], []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_l = criterion(val_output, y_val).item()
            val_loss.append(val_l)

            pred = (val_output >= 0.5).float()
            acc = (pred == y_val).float().mean().item()
            val_accuracy.append(acc)

    all_results[config] = {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy
    }
# Plot with two aligned subplots and shared labels/legend
fig, axes = plt.subplots(2, 1, figsize=(10 / 2.54, 8.07 / 2.54), sharex=True, dpi=300)

# Use concentric colors for both
shared_colors = {
    'train_loss': '#77c8d0',
    'val_loss': '#9f86c0',
    'val_acc': '#5e548e'
}

lines = []
labels = []

min_loss = min(np.min(all_results[config]['val_loss']) for config in electrode_configs)
max_loss = max(np.max(all_results[config]['train_loss']) for config in electrode_configs) # or val_loss, whichever is highest

common_ymin = min_loss * 0.8 # Adjust as needed
common_ymax = max_loss * 1.2 # Adjust as needed


panel_labels = ['a)', 'b)']
for ax_idx, config in enumerate(electrode_configs):
    ax1 = axes[ax_idx]
    ax1.text(-0.1, 1.05, panel_labels[ax_idx], transform=ax1.transAxes,
             fontsize=8, va='top', ha='right')
    ax2 = ax1.twinx()
    results = all_results[config]
    min_len = min(len(results['val_loss']), len(all_results['square']['val_loss']))

    x = np.arange(min_len)
    l1 = ax1.semilogy(x, results['train_loss'][:min_len], label='Training Loss', color=shared_colors['train_loss'], linestyle='--', linewidth=1)
    l2 = ax1.semilogy(x, results['val_loss'][:min_len], label='Validation Loss', color=shared_colors['val_loss'], linewidth=1)
    l3 = ax2.plot(x, results['val_accuracy'][:min_len], label='Validation Accuracy', color=shared_colors['val_acc'], linewidth=1)

    ax1.set_ylim(common_ymin, common_ymax) 

    ax1.tick_params(axis='both', labelsize=6, length=4, width=1)
    ax2.tick_params(axis='y', labelsize=6, length=4, width=1)
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    if ax_idx == 0:
        lines.extend([l1[0], l2[0], l3[0]])
        labels.extend(["Training Loss", "Validation Loss", "Validation Accuracy"])


# Shared axis labels
fig.text(0.06, 0.43, 'Loss (log scale)', va='center', rotation='vertical', fontsize=8)
fig.text(0.96, 0.43, 'Validation Accuracy', va='center', rotation='vertical', fontsize=8)
fig.text(0.5, -0.05, 'Epoch', ha='center', fontsize=8)

# Shared legend
fig.legend(lines, labels, loc='lower center', fontsize=8, ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.15))
fig.subplots_adjust(hspace=0.3, bottom=0.05, left=0.12, right=0.88)

plt.savefig((os.path.join(fpath, f"{date}_{plot_description}.png")), bbox_inches='tight', dpi=1200)
plt.show()

