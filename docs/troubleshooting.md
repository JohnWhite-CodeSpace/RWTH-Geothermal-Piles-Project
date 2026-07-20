# Troubleshooting Guide

Common issues and solutions when working with the PINN project.

---

## Installation Issues

### "ModuleNotFoundError: No module named 'torch'"

**Solution:**
```bash
pip install -r requirements.txt
```

Make sure your virtual environment is activated:
```bash
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows
```

### Different Package Versions Across Team

**Problem:** Team members have different versions installed.

**Solution:** Everyone run:
```bash
pip install -r requirements.txt
```

The versions are pinned in `requirements.txt` to ensure consistency.

### Virtual Environment Won't Activate

**On Windows:**
- Make sure you're in the repo directory
- Use: `venv\Scripts\activate` (backslashes, not forward slashes)

**On macOS/Linux:**
- Use: `source venv/bin/activate`

---

## Training Issues

### GPU Not Being Used

**Check if GPU is available:**
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

**Solutions:**
1. Install CUDA-enabled PyTorch:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. Set GPU ID in `.env`:
   ```
   USE_GPU=True
   GPU_ID=0
   ```

### Out of Memory (OOM) Errors

**Solutions:**
1. Reduce batch size in `.env`:
   ```
   BATCH_SIZE=16  # Instead of 32
   ```

2. Reduce model size (hidden dimensions)

3. Reduce number of workers:
   ```
   NUM_WORKERS=0  # Instead of 4
   ```

### NaN Loss During Training

**Possible causes:**
- Learning rate too high → Reduce `LEARNING_RATE` in `.env`
- Unstable gradients → Use gradient clipping
- Data not normalized → Check data preprocessing

**Solution:**
```bash
# Check data
python scripts/preprocess_data.py --check

# Reduce learning rate in configs/training_params.yaml
LEARNING_RATE: 0.0001
```

### Training Too Slow

**Optimizations:**
1. Increase batch size (if GPU memory allows)
2. Reduce number of epochs
3. Use GPU: `USE_GPU=True` in `.env`
4. Increase number of workers: `NUM_WORKERS=4` in `.env`

---

## Testing Issues

### Tests Fail After Code Changes

**Solution:**
```bash
# Clear Python cache
make clean

# Reactivate venv
deactivate
source venv/bin/activate

# Run tests
pytest
```

### Specific Test Fails

```bash
# Run with verbose output
pytest -v tests/test_models.py::TestPINN::test_specific_test

# Run with print statements
pytest -s tests/test_models.py
```

---

## Jupyter Issues

### Jupyter Doesn't See Code Changes

**Solution:** In your notebook, run:
```python
%load_ext autoreload
%autoreload 2
```

### Port 8888 Already in Use

**Solution:**
```bash
jupyter lab --port 8889
```

### Kernel Crashes When Importing Project Code

**Solution:**
```bash
# Restart kernel (Kernel > Restart in menu)

# Or in terminal:
jupyter kernelspec list
jupyter kernelspec uninstall python3
python -m ipykernel install --user --name python3
```

---

## Git Issues

### Merge Conflicts

**Prevention:** 
- Always `git pull` before starting work
- Discuss with team before pushing major changes

**Resolution:**
```bash
# See conflicts
git status

# Resolve conflicts in your editor, then:
git add .
git commit -m "fix: resolve merge conflicts"
git push
```

### Accidentally Committed Large Files

```bash
# Remove from git history (before push)
git rm --cached large_file.pth
echo "*.pth" >> .gitignore
git add .gitignore
git commit --amend
git push
```

### Wrong Branch

```bash
# Check current branch
git branch

# Switch to correct branch
git checkout correct-branch
```

---

## Development Environment

### Black Formatter Complains

**Solution:**
```bash
make format
```

### Flake8 Linting Fails

**Solution:**
```bash
# See all issues
flake8 src/

# Some can be auto-fixed with Black
black src/
```

---

## Data Issues

### Data Files Not Found

Check that files are in correct paths:
```
data/raw/          # Raw input data
data/processed/    # After preprocessing
```

Set paths in `.env`:
```
DATA_RAW_PATH=data/raw/
DATA_PROCESSED_PATH=data/processed/
```

### CSV Column Mismatch

**Solution:**
```bash
# Check data format
python scripts/preprocess_data.py --validate

# See expected format
cat data/raw/README.md
```

---

## Performance

### Check GPU Usage

```bash
# Linux/macOS
nvidia-smi

# Watch in real-time
watch nvidia-smi
```

### Profile Training Speed

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your training code here

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 functions
```

---

## Need More Help?

1. Check [README.md](../README.md)
2. Search GitHub Issues
3. Open new Issue with:
   - Clear description
   - Steps to reproduce
   - Error message
   - Environment info (Python, PyTorch versions, OS)
4. Contact team lead

---

Last updated: 2026-07-20
