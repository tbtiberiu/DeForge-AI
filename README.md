# DeForge-AI

DeForge-AI is a framework for detecting AI-generated images (AIGI). It utilizes a dual-branch architecture that combines deep semantic features from a Vision Transformer (ViT) with forensic noise analysis to achieve high accuracy and robustness across various generation methods.

## Resources

- **Datasets**:
  - [AIGIBench](https://huggingface.co/datasets/TheKernel01/AIGIBench): Primary training dataset.
  - [AIGC-Detection-Benchmark](https://huggingface.co/datasets/TheKernel01/AIGC-Detection-Benchmark): Evaluation benchmark for AI-generated content detection.
- **Pre-trained Models**:
  - [AIGIBench_models](https://huggingface.co/TheKernel01/AIGIBench_models): Access pre-trained checkpoints for DeForge-AI.

## Architecture

DeForge-AI employs a two-pronged approach to image forensics:

1.  **RGB Branch**: A DINOv3 (ViT-L/16) backbone fine-tuned using Parameter-Efficient Fine-Tuning (PEFT) with LoRA. This branch captures high-level semantic features and global context.
2.  **Forensic Branch**: A specialized "Noise Pattern Residual" branch that captures high-frequency artifacts. It is initialized with SRM (Spatial Rich Model) filters to extract noise residuals, followed by a series of convolutional layers to learn forensic signatures.
3.  **Fusion**: Features from both branches are fused using an attention-based pooling mechanism and a classification head to produce the final detection score.

## Installation

The project uses `uv` for package management. To set up the environment:

```bash
# Clone the repository
git clone https://github.com/tbtiberiu/deforge-ai.git
cd deforge-ai

# Install dependencies using uv
uv sync
```

Alternatively, you can use `pip`:

```bash
pip install .
```

Ensure you have a `.env` file in the root directory with your Hugging Face token to download the datasets:

```env
HF_TOKEN=your_huggingface_token_here
```

## Usage

### Training

To train the model on the `AIGIBench` dataset:

```bash
python train.py --batch-size 16 --epochs 5 --lr 1e-4 --image-size 256
```

**Key Arguments:**
- `--lr`: Learning rate (default: 1e-4).
- `--batch-size`: Number of samples per batch.
- `--epochs`: Number of training epochs.
- `--max-steps`: Maximum number of training steps.
- `--lora-r`: LoRA rank (default: 16).
- `--unfreeze-last-blocks`: Number of backbone blocks to unfreeze (default: 0).

### Evaluation

To evaluate a trained checkpoint on the `AIGC-Detection-Benchmark`:

```bash
python test.py --checkpoint checkpoints/model_epoch_best.pth --batch-size 16 --image-size 256
```

**Key Arguments:**
- `--checkpoint`: Path to the model checkpoint.
- `--limit`: Limit the total number of samples to test.

## Project Structure

- `model.py`: Architecture definition (ViT + LoRA + ForensicBranch).
- `dataset.py`: Data loading and augmentation pipelines.
- `train.py`: Training script for `AIGIBench`.
- `test.py`: Benchmark evaluation script.
- `main.py`: Entry point for basic sanity checks.
