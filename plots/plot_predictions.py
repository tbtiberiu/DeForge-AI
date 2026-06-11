import argparse
import os
import random
import sys

import matplotlib
import numpy as np
import torch
from datasets import load_dataset
from dotenv import load_dotenv

# Use Agg backend for headless environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to sys.path to allow imports from parent directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset import get_val_transforms
from model import DeForge_AI_Model

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


def main():
    parser = argparse.ArgumentParser(
        description='Plot model predictions with colored borders (green=correct, red=incorrect).'
    )
    parser.add_argument(
        '--dataset',
        '-d',
        type=str,
        required=True,
        choices=list(DATASET_CONFIGS.keys()),
        help='The dataset to use for plotting.',
    )
    parser.add_argument(
        '--cols',
        '-c',
        type=int,
        default=6,
        help='Number of columns in the output grid.',
    )
    parser.add_argument(
        '--rows', '-r', type=int, default=4, help='Number of rows in the output grid.'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/model_epoch_best.pth',
        help='Path to model checkpoint.',
    )
    parser.add_argument(
        '--image-size', type=int, default=256, help='Image size expected by the model.'
    )
    parser.add_argument(
        '--seed', type=int, default=42, help='Random seed for sample selection.'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for the generated plot image.',
    )

    args = parser.parse_args()

    # Load HF Token from .env
    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')

    # 1. Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    checkpoint_path = os.path.join(PROJECT_ROOT, args.checkpoint)
    if not os.path.exists(checkpoint_path):
        print(f'Error: Checkpoint {checkpoint_path} not found.')
        return

    # 2. Load model
    print(f'Loading checkpoint: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    checkpoint_args = checkpoint.get('args', {})

    model_kwargs = {
        'lora_r': checkpoint_args.get('lora_r', 16),
        'lora_alpha': checkpoint_args.get('lora_alpha', 32),
        'lora_dropout': checkpoint_args.get('lora_dropout', 0.5),
        'unfreeze_last_blocks': checkpoint_args.get('unfreeze_last_blocks', 0),
        'image_size': checkpoint_args.get('image_size', args.image_size),
        'forensic_dim': checkpoint_args.get('forensic_dim', 256),
    }

    lora_target_modules = checkpoint_args.get('lora_target_modules')
    if isinstance(lora_target_modules, str):
        model_kwargs['lora_target_modules'] = [
            m.strip() for m in lora_target_modules.split(',') if m.strip()
        ]
    elif lora_target_modules:
        model_kwargs['lora_target_modules'] = lora_target_modules

    model = DeForge_AI_Model(**model_kwargs).to(device)
    model.load_state_dict(
        checkpoint['model_state_dict']
        if 'model_state_dict' in checkpoint
        else checkpoint,
        strict=False,
    )
    model.eval()

    # 3. Load dataset config and dataset
    config = DATASET_CONFIGS[args.dataset]
    print(f'Loading dataset: {args.dataset} from {config["path"]}...')
    dataset = load_dataset(config['path'], split='test', token=hf_token)

    # 4. Filter indices to get one sample from each generator (as equally distributed as possible)
    print('Analyzing generator distribution for balanced sampling...')
    
    # Check column names to identify generator column
    if 'generator' in dataset.column_names:
        all_generators = np.array(dataset['generator'])
    elif 'label' in dataset.column_names:
        all_generators = np.array(dataset['label'])
    else:
        print('Warning: Neither generator nor label columns found. Defaulting generator IDs to 0.')
        all_generators = np.zeros(len(dataset), dtype=int)

    unique_generators = sorted(list(config['mapping'].keys()))
    
    # Group indices by generator ID
    indices_by_gen = {gen_id: [] for gen_id in unique_generators}
    for idx, gen_id in enumerate(all_generators):
        if gen_id in indices_by_gen:
            indices_by_gen[gen_id].append(idx)
        else:
            if 0 in indices_by_gen:
                indices_by_gen[0].append(idx)

    # Filter out generators that have no samples in this dataset split
    available_gens = [g for g in unique_generators if len(indices_by_gen[g]) > 0]
    print(f'Available generators: {[config["mapping"].get(g, f"Gen {g}") for g in available_gens]}')

    num_samples = args.cols * args.rows
    
    # Distribute the slots as evenly as possible among available generators
    samples_per_gen = {gen_id: 0 for gen_id in unique_generators}
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    slots_to_fill = num_samples
    gens_cycle = list(available_gens)
    
    # Simple round-robin distribution to allocate sample count per generator
    while slots_to_fill > 0 and gens_cycle:
        for gen_id in list(gens_cycle):
            if slots_to_fill == 0:
                break
            # Can we take another sample from this generator?
            current_allocated = samples_per_gen[gen_id]
            if current_allocated < len(indices_by_gen[gen_id]):
                samples_per_gen[gen_id] += 1
                slots_to_fill -= 1
            else:
                # No more samples left in this generator, remove from cycle
                gens_cycle.remove(gen_id)

    # Retrieve the indices
    selected_indices = []
    for gen_id, count in samples_per_gen.items():
        if count > 0:
            selected_indices.extend(random.sample(indices_by_gen[gen_id], count))

    # Shuffle selected indices to mix the subplots randomly
    random.shuffle(selected_indices)
    print(f'Selected a total of {len(selected_indices)} samples across generators.')

    # 5. Inference and visualization
    val_transform = get_val_transforms(size=args.image_size)
    mapping = config['mapping']

    print(f'Generating grid of {args.rows}x{args.cols} predictions...')
    fig, axes = plt.subplots(
        args.rows, args.cols, figsize=(args.cols * 3.5, args.rows * 4)
    )

    # Flatten the axes for simple 1D indexing
    if args.rows == 1 and args.cols == 1:
        axes_flat = [axes]
    else:
        axes_flat = axes.flatten()

    for idx, (ax, sample_idx) in enumerate(zip(axes_flat, selected_indices)):
        sample = dataset[int(sample_idx)]

        # Load and convert image
        image_pil = sample['image'].convert('RGB')

        # Extract ground truth target and generator info
        if 'generator' in sample:
            gen_id = sample['generator']
            target = 0.0 if gen_id == 0 else 1.0
            gen_name = mapping.get(gen_id, f'Gen {gen_id}')
        else:
            label_val = sample.get('label', 0)
            target = 0.0 if label_val == 0 else 1.0
            gen_name = 'Fake' if target == 1.0 else 'Real'

        # Apply model transforms and prepare for inference
        img_tensor = val_transform(image_pil).unsqueeze(0).to(device)

        with torch.inference_mode():
            logits = model(img_tensor)
            prob = torch.sigmoid(logits).squeeze().item()

        pred = 1.0 if prob > 0.5 else 0.0
        correct = pred == target

        # Define display info
        true_lbl = 'Real' if target == 0.0 else f'Fake ({gen_name})'
        pred_lbl = 'Fake' if pred == 1.0 else 'Real'
        conf = prob if pred == 1.0 else (1.0 - prob)

        # Plot resized image
        display_img = image_pil.resize((args.image_size, args.image_size))
        ax.imshow(display_img)

        # Add colored border
        border_color = (
            '#2ecc71' if correct else '#e74c3c'
        )  # Green for correct, Red for incorrect
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(border_color)
            spine.set_linewidth(6)

        ax.set_xticks([])
        ax.set_yticks([])

        # Add text title
        ax.set_title(
            f'True: {true_lbl}\nPred: {pred_lbl} ({conf:.1%})',
            color='#000000',
            fontsize=14,
            fontweight='bold',
            pad=10,
        )

    # Hide extra axes if we have more axes than samples (edge case)
    for ax in axes_flat[len(selected_indices) :]:
        ax.axis('off')

    # Figure labels and saving
    plt.suptitle(
        f'Model Predictions on {args.dataset}\n(Green = Correct, Red = Incorrect)',
        fontsize=20,
        fontweight='bold',
        y=0.98,
        color='#000000',
    )
    plt.tight_layout()
    plt.subplots_adjust(
        top=0.92 - (0.02 * (4 / args.rows) if args.rows > 1 else 0.1),
        hspace=0.3,
        wspace=0.1,
    )

    # Resolve output path
    if args.output is None:
        output_dir = os.path.join(PROJECT_ROOT, 'images', 'predictions')
        os.makedirs(output_dir, exist_ok=True)
        ds_filename = args.dataset.lower().replace('-', '_')
        args.output = os.path.join(output_dir, f'{ds_filename}_predictions_grid.jpg')
    else:
        # If absolute path, use it directly, else join with project root
        if not os.path.isabs(args.output):
            args.output = os.path.join(PROJECT_ROOT, args.output)
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    plt.savefig(args.output, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Grid plot saved to: {args.output}')


if __name__ == '__main__':
    main()
