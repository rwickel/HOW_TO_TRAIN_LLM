# Qwen3-1.7B Fine-Tuning Project

This project fine-tunes the Qwen3-1.7B language model on the OpenAssistant-Guanaco dataset using Hugging Face's Transformers and TRL libraries.

## Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Usage](#usage)
5. [Test](#test)

## Overview

- **Model**: [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) from Alibaba's Qwen series
- **Dataset**: [OpenAssistant-Guanaco](https://huggingface.co/datasets/timdettmers/openassistant-guanaco) (subset of 256 samples)
- **Training Approach**: Supervised Fine-Tuning (SFT) using `SFTTrainer`
- **Hardware**: Optimized for GPU with mixed-precision training (BF16/FP16)

## Features

- Custom chat template for Qwen3's role-based format
- Memory-efficient training with gradient accumulation
- Support for both BF16 and FP16 precision
- Training monitoring with logging and checkpointing
- Evaluation during training

## Requirements

- Python 3.8+
- PyTorch with CUDA support
- Hugging Face libraries:
  - `transformers`
  - `datasets`
  - `trl`
- GPU with at least 20GB memory (adjust `max_memory` in code for smaller GPUs)

## Installation

```bash
pip install torch transformers datasets trl
```

## Usage 

- Clone this repository

- Save the following code as train.py

- Run the training script:

```bash
python train.py
```

## Test 

- Run the fine tuned model:

```bash
python test.py
```