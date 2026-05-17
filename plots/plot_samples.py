import multiprocessing
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datasets import load_dataset
from dotenv import load_dotenv


def generate_comparison_grid(hf_token, output_path):
    """
    Generates a 3x3 grid with images from the AIGIBench dataset:
    Column 1: 3 x Real (generator=0)
    Column 2: 3 x ProGAN (generator=1)
    Column 3: 3 x SD14 (generator=2)
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print('Loading dataset TheKernel01/AIGIBench...')
    dataset = load_dataset(
        'TheKernel01/AIGIBench', token=hf_token, split='train', streaming=True
    )

    # Collect samples
    samples = {0: [], 1: [], 2: []}
    needed = 4

    print('Searching for samples for the grid...')
    for item in dataset:
        gen = item.get('generator')
        if gen in samples and len(samples[gen]) < needed:
            samples[gen].append(item['image'])

        # Stop once we have all needed samples
        if all(len(v) == needed for v in samples.values()):
            break

    # Column titles
    col_titles = ['Real', 'ProGAN', 'Stable Diffusion 1.4']
    gen_ids = [0, 1, 2]  # Mapping columns to generator IDs

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    for col_idx, gen_id in enumerate(gen_ids):
        images = samples[gen_id]
        for row_idx in range(3):
            ax = axes[row_idx, col_idx]
            if row_idx < len(images):
                img = images[row_idx]
                ax.imshow(img)
                if row_idx == 0:
                    ax.set_title(col_titles[col_idx], fontsize=14, fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'Missing Image', ha='center', va='center')

            ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Grid saved to: {output_path}')
    plt.close()


if __name__ == '__main__':
    # Set start method for multiprocessing to avoid finalization GIL issues
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')

    if not hf_token:
        print(
            'Warning: HF_TOKEN is not set in the .env file. Attempting without token...'
        )

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'images', 'samples', 'aigibench_samples.jpg')
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    generate_comparison_grid(hf_token, OUTPUT_FILE)
    print('Done!')
