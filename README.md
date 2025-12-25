# Openclip-CLIP on CIFAR-10: Zero-Shot and Adaptation Methods

This repository implements several popular CLIP-based classification techniques on the CIFAR-10 dataset using the OpenCLIP library. It supports:

- **Zero-shot classification**
- **Linear probing** on frozen image embeddings
- **MLP probing** on frozen image embeddings
- **Learnable image + text projection heads** (lightweight trainalbe MLPs)

The code is modular, easy to extend, and designed for reproducible experiments.

## Features

- Any OpenCLIP model and pretrained weights (e.g., ViT-B/32 with LAION-2B or OpenAI weights)
- Automatic CIFAR-10 download (if isn't available on 'data' file) and CLIP-compatible preprocessing
- Customizable text prompt templates (we have two modes, 1. shared, 2. per_class)
- Validation-based model selection (best checkpoint on validation set)
- L2-normalized embeddings with cosine similarity classification

## Installation

### Requirements

- Python ≥ 3.8
- PyTorch (CUDA recommended)
- open-clip-torch

### Setup

```bash
# Install dependencies
pip install torch torchvision
pip install open-clip-torch tqdm scikit-learn
```

## Usage
Run experiments with main.py and command-line arguments.
### Examples
```bash
# Zero-shot classification
python main.py --method zero_shot

# Linear probe on frozen image features
python main.py --method linear_probe_img

# MLP probe on frozen image features
python main.py --method mlp_probe_img

# Image + text MLP projection heads
python main.py --method img_text_mlp_probe
```

### Key Arguments
```bash
Argument,          Description,                                                         Default
--method,          Method to run,                                                       zero_shot
--clip_model,      CLIP architecture                                                    ViT-B/32
--clip_pretrained  Pretrained weights                                                   laion2b_s34b_b79k
--train_batch_size Training batch size                                                  128
--test_batch_size  Test/validation batch                                                size,256
--templates,       Text prompt templates                                                """a photo of a {classname}.""" (multiple allowed)
--proj_epochs,     Epochs for projection head training                                  7
--mlp_epochs,      Epochs for MLP probe training                                        20
--template_mode,   specialized template for each class or a general template for all.   shared
--per_class_templates, Templates for each class                                         

```
## Project Structure

```bash
codes_python_based
├── main.py                        # Entry point & argument parsing
├── clip_model.py                  # Loads OpenCLIP model, tokenizer, preprocess
├── cifar10.py                     # CIFAR-10 loaders with validation split
├── methods/
│   ├── zero_shot.py               # Zero-shot classification
│   ├── embed_img_inference.py     # Linear & MLP probes on image embeddings
│   └── embed_txt_img_inference.py # Dual projection heads (image + text)
├── embeddings/
│   ├── image_embedding.py         # Extract frozen image features
│   └── text_embedding.py          # Compute class text embeddings
└── data/                          # Created automatically (CIFAR-10 data)
└── Report/                        # it contains a .md file and report about our obtained results and findings 
```

## Notes
1. The CLIP backbone is frozen in all methods (standard practice).
2. Embeddings are L2-normalized before similarity computation.
3. Validation set is a 45k/5k split from the original CIFAR-10 training set.
4. For better zero-shot results, experiment with multiple prompt templates(in shared template mode, when we use the same template for all of our 10 classes).
5. There is a `.ipynb` file outside the `codes_python_based` folder. It contains the same implementation as the `.py` files in `codes_python_based`, but in a shorter Jupyter Notebook format suitable for presentations rather than development.

## References
1. OpenCLIP: https://github.com/mlfoundations/open_clip
2. CLIP: Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (ICML 2021)

