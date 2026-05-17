import os
import random

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')

DATASET_CONFIGS = {
    'AIGC-Detection-Benchmark': {
        'path': 'TheKernel01/AIGC-Detection-Benchmark',
        'mapping': {
            0: 'Real',
            1: 'ADM',
            2: 'BigGAN',
            3: 'CycleGAN',
            4: 'DALLE2',
            5: 'GauGAN',
            6: 'GLIDE',
            7: 'Midjourney',
            8: 'ProGAN',
            9: 'SD14',
            10: 'SD15',
            11: 'SDXL',
            12: 'StarGAN',
            13: 'StyleGAN',
            14: 'StyleGAN2',
            15: 'VQDM',
            16: 'WhichFaceIsReal',
            17: 'Wukong',
        },
    },
    'MS-COCOAI': {
        'path': 'TheKernel01/MS-COCOAI',
        'mapping': {
            0: 'Real',
            1: 'SD21',
            2: 'SDXL',
            3: 'SD3',
            4: 'DALLE3',
            5: 'Midjourney 6',
        },
    },
    '140k-Real-and-Fake-Faces': {
        'path': 'TheKernel01/140k-Real-and-Fake-Faces',
        'mapping': {0: 'Real', 1: 'StyleGAN'},
    },
}


def generate_samples_grid(dataset_name, output_path):
    print(f'Loading dataset: {dataset_name}...')
    config = DATASET_CONFIGS[dataset_name]

    dataset = load_dataset(config['path'], split='test', token=HF_TOKEN)

    mapping = config['mapping']
    unique_generators = sorted(mapping.keys())

    print('  Pre-indexing generator labels...')
    all_generators = np.array(dataset['generator'])

    SAMPLES_PER_GEN = 3
    num_categories = len(unique_generators)

    if num_categories <= 9:
        cols = num_categories
        rows = SAMPLES_PER_GEN
    else:
        cols = 9
        rows = (num_categories // cols) * SAMPLES_PER_GEN

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.8, rows * 3))
    # Ensure axes is a 2D array for easier indexing
    if rows == 1 or cols == 1:
        axes = np.atleast_2d(axes)

    for i, gen_id in enumerate(unique_generators):
        gen_name = mapping[gen_id]
        indices = np.nonzero(all_generators == gen_id)[0]

        if len(indices) > 0:
            print(f'  Extracting {SAMPLES_PER_GEN} samples for: {gen_name}')
            num_to_pick = min(len(indices), SAMPLES_PER_GEN)
            selected_indices = random.sample(list(indices), num_to_pick)

            # Determine column and row offset for this category
            col_idx = i % cols
            row_offset = (i // cols) * SAMPLES_PER_GEN

            for r_idx, sample_idx in enumerate(selected_indices):
                ax = axes[row_offset + r_idx, col_idx]
                sample = dataset[int(sample_idx)]
                img = sample['image']

                ax.imshow(img)
                # Only put title on the first sample of each column to avoid clutter
                if r_idx == 0:
                    ax.set_title(gen_name, fontsize=12, fontweight='bold', pad=10)
                ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved vertical-column grid to: {output_path}')


def main():
    # Resolve project root dynamically
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(PROJECT_ROOT, 'images', 'samples')
    os.makedirs(output_dir, exist_ok=True)

    for ds_key in DATASET_CONFIGS.keys():
        output_file = os.path.join(output_dir, f'{ds_key.lower()}_samples.jpg')
        generate_samples_grid(ds_key, output_file)


if __name__ == '__main__':
    main()
