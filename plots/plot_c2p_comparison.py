import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def plot_c2p_ap_comparison(data, filename):
    # Maximum lightness (Pure White)
    bg0 = '#ffffff' 
    fg0 = '#282828' # Dark Text
    gray = '#928374'
    green = '#98971a' # Gruvbox Green for AP
    blue = '#458588'  # Gruvbox Blue for alternative
    
    # Font sizes
    TITLE_SIZE = 24
    LABEL_SIZE = 20
    TICK_SIZE = 16
    LEGEND_SIZE = 18
    
    plt.rcParams['text.color'] = fg0
    plt.rcParams['axes.labelcolor'] = fg0
    plt.rcParams['xtick.color'] = fg0
    plt.rcParams['ytick.color'] = fg0
    
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.patch.set_facecolor(bg0)
    ax.set_facecolor(bg0)
    
    datasets = list(data.keys())
    models = ['C2P-CLIP-GenImage', 'C2P-CLIP-AIGIBench']
    
    x = np.arange(len(datasets))
    width = 0.35
    
    ap_genimage = [data[ds]['C2P-CLIP-GenImage'] for ds in datasets]
    ap_aigibench = [data[ds]['C2P-CLIP-AIGIBench'] for ds in datasets]
    
    rects1 = ax.bar(x - width/2, ap_genimage, width, label='C2P-CLIP-GenImage', color=blue, alpha=0.9)
    rects2 = ax.bar(x + width/2, ap_aigibench, width, label='C2P-CLIP-AIGIBench', color=green, alpha=0.9)
    
    ax.set_ylabel('Scor AP', fontsize=LABEL_SIZE, color=fg0, fontweight='bold')
    ax.set_title('Comparație Performanță AP: Variante C2P-CLIP', fontsize=TITLE_SIZE, pad=30, color=fg0, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=TICK_SIZE)
    ax.tick_params(axis='y', labelsize=TICK_SIZE)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', linestyle='--', alpha=0.4, color=gray)
    
    legend = ax.legend(facecolor=bg0, edgecolor=gray, fontsize=LEGEND_SIZE, loc='lower right')
    plt.setp(legend.get_texts(), color=fg0)
    
    # Spines color
    for spine in ax.spines.values():
        spine.set_color(gray)
        spine.set_linewidth(1.5)

    fig.tight_layout()
    plt.savefig(filename, dpi=300, facecolor=bg0)
    plt.close()
    print(f"Saved plot to {filename}")

# Data extraction from the provided tables (Only AP)
c2p_ap_data = {
    'AIGC-Detection': {
        'C2P-CLIP-GenImage': 0.8568,
        'C2P-CLIP-AIGIBench': 0.9771
    },
    'MS-COCOAI': {
        'C2P-CLIP-GenImage': 0.8587,
        'C2P-CLIP-AIGIBench': 0.5876
    },
    '140k-Faces': {
        'C2P-CLIP-GenImage': 0.9632,
        'C2P-CLIP-AIGIBench': 0.6914
    }
}

if __name__ == "__main__":
    # Resolve project root dynamically
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(PROJECT_ROOT, 'images', 'benchmarks')
    os.makedirs(output_dir, exist_ok=True)

    plot_c2p_ap_comparison(
        c2p_ap_data, os.path.join(output_dir, 'c2p_clip_ap_comparison.png')
    )
