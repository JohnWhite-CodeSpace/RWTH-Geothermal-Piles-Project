# Advanced Setup Guide

This guide covers additional setup options beyond the quick start in [README.md](../README.md).

---

## Table of Contents

- [GPU Setup](#gpu-setup)
- [IDE Configuration](#ide-configuration)
- [Logging Configuration](#logging-configuration)
- [Weights & Biases Integration](#weights--biases-integration)

---

## GPU Setup

### CUDA Installation

For optimal training performance, install CUDA-enabled PyTorch.

**Step 1: Check NVIDIA GPU**

```bash
nvidia-smi
```

Note the CUDA version (e.g., 11.8, 12.1).

**Step 2: Install CUDA PyTorch**

Replace `cu118` with your CUDA version:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Step 3: Verify Installation**

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Should output:
```
True
NVIDIA GeForce RTX 3090
```

### Multi-GPU Setup

For distributed training on multiple GPUs:

```python
import torch
import torch.nn as nn

# Wrap model with DataParallel
model = PINN(input_dim=3, output_dim=2)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)
    model.to('cuda')
```


---

## IDE Configuration

### Visual Studio Code

Create `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.flake8Args": ["--max-line-length=100"],
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"],
    "[python]": {
        "editor.defaultFormatter": "ms-python.python",
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.organizeImports": true
        }
    },
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        "**/.pytest_cache": true
    }
}
```

### PyCharm/IntelliJ

1. **Set Python Interpreter:**
   - Settings → Project → Python Interpreter
   - Select `venv/bin/python`

2. **Enable Code Inspection:**
   - Settings → Editor → Inspections
   - Enable "PEP 8 naming convention"

3. **Configure Formatter:**
   - Settings → Tools → Python Integrated Tools
   - Set Code formatter to "Black"

4. **Run Tests:**
   - Right-click on `tests/` folder
   - Select "Run pytest in tests"

---

## Logging Configuration

### Setup Logging

Create `src/utils/logger.py`:

```python
import logging
import os
from pathlib import Path

def setup_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """Setup logger for the project."""
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level))
    
    # File handler
    fh = logging.FileHandler(log_dir / f"{name}.log")
    fh.setLevel(getattr(logging, log_level))
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, log_level))
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger
```

### Usage in Code

```python
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

logger.info("Starting training")
logger.warning("GPU not available")
logger.error("Training failed")
```

---

## Weights & Biases Integration

Track experiments with W&B.

### Installation

```bash
pip install wandb

# Login
wandb login
```

### Log Training Metrics

```python
import wandb
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger

# Setup
wandb_logger = WandbLogger(project="geothermal-pines", entity="your-team")

# Training
trainer = Trainer(
    logger=wandb_logger,
    max_epochs=1000,
    log_every_n_steps=10,
)

trainer.fit(model, train_dataloader, val_dataloader)

# Log hyperparameters
wandb.config.update({
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 1000
})
```

### View Results

- Visit [wandb.ai](https://wandb.ai)
- See training curves, compare runs, etc.

---

## Performance Monitoring

### TensorBoard

```bash
pip install tensorboard

# During training
tensorboard --logdir=logs/

# Open http://localhost:6006 in browser
```

### Profiling Training Speed

```python
import torch.profiler

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, 
                torch.profiler.ProfilerActivity.CUDA],
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs')
) as prof:
    # Training code
    pass

prof.step()
```

---

## Environment Variables

Create `.env` file (copy from `.env.example`):

```bash
cp .env.example .env
```

Edit for your setup:

```env
# GPU
USE_GPU=True
GPU_ID=0

# Data paths
DATA_RAW_PATH=data/raw/
DATA_PROCESSED_PATH=data/processed/

# Training
BATCH_SIZE=64
EPOCHS=1000
LEARNING_RATE=0.001

# Logging
LOG_LEVEL=INFO
```

Load in Python:

```python
import os
from dotenv import load_dotenv

load_dotenv()

batch_size = int(os.getenv("BATCH_SIZE", 32))
use_gpu = os.getenv("USE_GPU") == "True"
```

---

## Troubleshooting Setup

### CUDA Version Mismatch

```bash
# Check CUDA version
nvidia-smi

# Check PyTorch CUDA support
python -c "import torch; print(torch.version.cuda)"

# If mismatch, reinstall:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Out of Virtual Memory

```bash
# Increase swap (Linux)
sudo dd if=/dev/zero of=/swapfile bs=1G count=10
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Slow Data Loading

```bash
# Increase number of workers
NUM_WORKERS=8  # In .env

# Or set in code
loader = DataLoader(dataset, num_workers=8, pin_memory=True)
```

---

## Next Steps

- See [README.md](../README.md) for quick start
- See [theory.md](theory.md) for theory background
- See [../CONTRIBUTING.md](../CONTRIBUTING.md) for code guidelines

---

Last updated: 2026-07-20
