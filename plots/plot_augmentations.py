import multiprocessing
import os

import matplotlib
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from torchvision.transforms import v2

import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset import DownUpResize, GaussianNoise

# Use Agg backend for headless environments
matplotlib.use('Agg')


def apply_transform(img_tensor, transform_type):
    """Applies a specific transform to an image tensor."""
    if transform_type == 'original':
        return img_tensor
    elif transform_type == 'downup':
        dur = DownUpResize(scale_range=(0.6, 0.6))
        return dur(img_tensor)
    elif transform_type == 'jitter':
        jitter = v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)
        return jitter(img_tensor)
    elif transform_type == 'blur':
        blur = v2.GaussianBlur(kernel_size=5, sigma=(1.2, 1.2))
        return blur(img_tensor)
    elif transform_type == 'jpeg':
        img_uint8 = (img_tensor * 255).to(torch.uint8)
        jpeg_transform = v2.JPEG(quality=40)
        img_jpeg = jpeg_transform(img_uint8)
        return img_jpeg.to(torch.float32) / 255.0
    elif transform_type == 'noise':
        gn = GaussianNoise(sigma_range=(0.015, 0.015))
        return gn(img_tensor)
    elif transform_type == 'train_pipe':
        # Create a visual version of train transforms (without Normalize)
        size = 256
        resize_size = int(round(size * 1.15))

        # We need to ensure uint8 for JPEG
        img_uint8 = (img_tensor * 255).to(torch.uint8)

        train_pipe = v2.Compose(
            [
                v2.Resize(resize_size, antialias=False),
                v2.RandomCrop(size),
                v2.RandomHorizontalFlip(p=1.0),  # Force flip for visibility
                DownUpResize(scale_range=(0.7, 0.7)),
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                v2.GaussianBlur(kernel_size=5, sigma=(0.8, 0.8)),
                v2.JPEG(quality=60),
                v2.ToDtype(torch.float32, scale=True),
                GaussianNoise(sigma_range=(0.01, 0.01)),
            ]
        )
        return train_pipe(img_uint8)
    elif transform_type == 'val_pipe':
        # Create a visual version of val transforms (without Normalize)
        size = 256
        resize_size = int(round(size * 1.15))
        val_pipe = v2.Compose(
            [
                v2.Resize(resize_size, antialias=False),
                v2.CenterCrop(size),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )
        return val_pipe(img_tensor)
    return img_tensor


def generate_augmentations_plot(hf_token, output_path):
    print('Loading dataset TheKernel01/AIGIBench (streaming)...')
    dataset = load_dataset(
        'TheKernel01/AIGIBench', token=hf_token, split='train', streaming=True
    )

    samples = {0: None, 1: None}
    print('Searching for one Real and one Fake sample...')
    for item in dataset:
        label = item['label']
        if label in samples and samples[label] is None:
            samples[label] = item['image'].convert('RGB').resize((256, 256))
            print(f' Found {"Real" if label == 0 else "Fake"} sample.')
        if all(v is not None for v in samples.values()):
            break

    transforms = [
        'original',
        'downup',
        'jitter',
        'blur',
        'jpeg',
        'noise',
        'train_pipe',
        'val_pipe',
    ]
    titles = [
        'Original',
        'Redimensionare',
        'Variații de culoare',
        'Blur Gaussian',
        'JPEG (Q=40)',
        'Zgomot Gaussian',
        'Flux antrenare',
        'Flux validare',
    ]

    fig, axes = plt.subplots(2, 8, figsize=(22, 7))

    # Common to_tensor transform
    to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    for row_idx, label in enumerate([0, 1]):  # 0: Real, 1: Fake
        img = samples[label]
        img_tensor = to_tensor(img)
        label_text = 'Real' if label == 0 else 'Fals'

        for col_idx, t_type in enumerate(transforms):
            ax = axes[row_idx, col_idx]

            # Apply transform
            transformed = apply_transform(img_tensor.clone(), t_type)

            # Convert back to numpy for plotting
            img_np = transformed.permute(1, 2, 0).numpy()
            # Clip to [0, 1] for safety in plotting
            img_np = img_np.clip(0, 1)
            ax.imshow(img_np)

            if row_idx == 0:
                ax.set_title(titles[col_idx], fontsize=10, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(label_text, fontsize=12, fontweight='bold', labelpad=10)

            ax.axis('off')

    plt.suptitle(
        'DeForge-AI: Strategii de augmentare a datelor',
        fontsize=16,
        y=1.05,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Augmentation plot saved to: {output_path}')
    plt.close()


if __name__ == '__main__':
    # Set start method for multiprocessing to avoid issues
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')

    if not hf_token:
        print(
            'Warning: HF_TOKEN is not set. Some datasets might require authentication.'
        )

    output_dir = os.path.join(PROJECT_ROOT, 'images', 'augmentations')
    output_file = os.path.join(output_dir, 'augmentations_samples.jpg')
    os.makedirs(output_dir, exist_ok=True)

    generate_augmentations_plot(hf_token, output_file)
    print('Done!')
