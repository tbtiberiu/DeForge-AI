import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def plot_dataset_metrics(dataset_name, df, filename):
    # Maximum lightness (Pure White)
    bg0 = '#ffffff' 
    fg0 = '#282828' # Dark Text
    gray = '#928374'
    blue = '#458588'
    green = '#98971a'
    red = '#cc241d'

    # Larger font sizes for better visibility
    TITLE_SIZE = 24
    LABEL_SIZE = 20
    TICK_SIZE = 16
    LEGEND_SIZE = 18


    plt.rcParams['text.color'] = fg0
    plt.rcParams['axes.labelcolor'] = fg0
    plt.rcParams['xtick.color'] = fg0
    plt.rcParams['ytick.color'] = fg0

    fig, ax = plt.subplots(figsize=(16, 10))  # Slightly larger figure
    fig.patch.set_facecolor(bg0)
    ax.set_facecolor(bg0)

    models = df['Model']
    accuracy = df['Accuracy']
    ap = df['AP']

    x = np.arange(len(models))
    width = 0.35

    # Using Gruvbox Blue and Green for the bars
    rects1 = ax.bar(
        x - width / 2, accuracy, width, label='Acuratețe', color=blue, alpha=0.9
    )
    rects2 = ax.bar(x + width / 2, ap, width, label='AP', color=green, alpha=0.9)

    ax.set_ylabel('Scoruri', fontsize=LABEL_SIZE, color=fg0, fontweight='bold')
    ax.set_title(
        f'Comparație Performanță: {dataset_name}',
        fontsize=TITLE_SIZE,
        pad=30,
        color=fg0,
        fontweight='bold',
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', color=fg0, fontsize=TICK_SIZE)
    ax.tick_params(axis='y', labelsize=TICK_SIZE)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', linestyle='--', alpha=0.4, color=gray)

    legend = ax.legend(
        facecolor=bg0, edgecolor=gray, fontsize=LEGEND_SIZE, loc='upper left'
    )
    plt.setp(legend.get_texts(), color=fg0)

    # Guiding lines for DeForge-AI
    if 'DeForge-AI' in models.values:
        idx = models[models == 'DeForge-AI'].index[0]
        deforge_acc = accuracy[idx]
        deforge_ap = ap[idx]

        # Horizontal lines matching the bar colors
        ax.axhline(y=deforge_acc, color=blue, linestyle=':', linewidth=2, alpha=0.9)
        ax.axhline(y=deforge_ap, color=green, linestyle=':', linewidth=2, alpha=0.9)

        # Highlight DeForge-AI label
        for i, model in enumerate(models):
            if model == 'DeForge-AI':
                ax.get_xticklabels()[i].set_fontweight('bold')
                ax.get_xticklabels()[i].set_color(red)
                ax.get_xticklabels()[i].set_fontsize(TICK_SIZE + 2)

    # Spines color
    for spine in ax.spines.values():
        spine.set_color(gray)
        spine.set_linewidth(1.5)

    fig.tight_layout()
    plt.savefig(filename, dpi=300, facecolor=bg0)
    plt.close()
    print(f'Saved plot to {filename}')


# Data for AIGC-Detection-Benchmark
aigc_data = {
    'Model': [
        'AIDE',
        'C2P-CLIP',
        'C2P-DINOv2',
        'CLIPDetection',
        'CNNDetection',
        'DeForge-AI',
        'DFFreq',
        'Effort',
        'FreqNet',
        'GramNet',
        'LaDeDa',
        'LGrad',
        'NPR',
        'ResNet50',
        'RIGID',
        'SAFE',
    ],
    'Accuracy': [
        0.7323,
        0.9018,
        0.8255,
        0.8406,
        0.6595,
        0.9466,
        0.8364,
        0.8924,
        0.8203,
        0.7094,
        0.8152,
        0.7262,
        0.7596,
        0.7500,
        0.7300,
        0.8089,
    ],
    'AP': [
        0.8647,
        0.9771,
        0.9518,
        0.9261,
        0.8199,
        0.9895,
        0.9404,
        0.9546,
        0.9109,
        0.8133,
        0.8821,
        0.8178,
        0.8591,
        0.8465,
        0.7872,
        0.9176,
    ],
}

# Data for MS-COCOAI
cocoai_data = {
    'Model': [
        'AIDE',
        'C2P-CLIP',
        'C2P-DINOv2',
        'CLIPDetection',
        'CNNDetection',
        'DeForge-AI',
        'DFFreq',
        'Effort',
        'FreqNet',
        'GramNet',
        'LaDeDa',
        'LGrad',
        'NPR',
        'ResNet50',
        'RIGID',
        'SAFE',
    ],
    'Accuracy': [
        0.5000,
        0.5018,
        0.5021,
        0.5436,
        0.5020,
        0.5893,
        0.4995,
        0.5001,
        0.4998,
        0.5013,
        0.5000,
        0.4998,
        0.4995,
        0.4997,
        0.5208,
        0.5016,
    ],
    'AP': [
        0.5008,
        0.5876,
        0.5242,
        0.6426,
        0.5448,
        0.6593,
        0.5012,
        0.5563,
        0.5077,
        0.5074,
        0.4996,
        0.4999,
        0.4993,
        0.5044,
        0.4962,
        0.4949,
    ],
}

# Data for 140k-Real-and-Fake-Faces
faces_data = {
    'Model': [
        'AIDE',
        'C2P-CLIP',
        'C2P-DINOv2',
        'CLIPDetection',
        'CNNDetection',
        'DeForge-AI',
        'DFFreq',
        'Effort',
        'FreqNet',
        'GramNet',
        'LaDeDa',
        'LGrad',
        'NPR',
        'ResNet50',
        'RIGID',
        'SAFE',
    ],
    'Accuracy': [
        0.5015,
        0.5025,
        0.5160,
        0.7375,
        0.4985,
        0.9080,
        0.5000,
        0.6715,
        0.4995,
        0.5000,
        0.5000,
        0.4760,
        0.5000,
        0.5000,
        0.8480,
        0.5000,
    ],
    'AP': [
        0.5391,
        0.6914,
        0.8568,
        0.8342,
        0.5594,
        0.9696,
        0.5015,
        0.9514,
        0.5110,
        0.4951,
        0.5000,
        0.4851,
        0.5000,
        0.4947,
        0.9205,
        0.5240,
    ],
}

if __name__ == '__main__':
    # Resolve project root dynamically
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(PROJECT_ROOT, 'images', 'benchmarks')
    os.makedirs(output_dir, exist_ok=True)

    plot_dataset_metrics(
        'AIGC-Detection-Benchmark',
        pd.DataFrame(aigc_data),
        os.path.join(output_dir, 'benchmark_AIGC.png'),
    )
    plot_dataset_metrics(
        'MS-COCOAI',
        pd.DataFrame(cocoai_data),
        os.path.join(output_dir, 'benchmark_MSCOCO.png'),
    )
    plot_dataset_metrics(
        '140k-Real-and-Fake-Faces',
        pd.DataFrame(faces_data),
        os.path.join(output_dir, 'benchmark_140k.png'),
    )

