# RWTH-Geothermal-Piles-Project

## Physics-Informed Neural Networks for Geothermal Piles

Research project implementing Physics-Informed Neural Networks (PINNs) to model the effect of temperature on excess pore-water pressures and shaft bearing capacity of geothermal piles.

**Status:** Active Development | **Team:** 2 | **Language:** English 🌍

---

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Usage](#-usage)
- [Useful Commands](#-useful-commands)
- [Team Guidelines](#-team-guidelines)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)

---

## 🚀 Quick Start

**Get up and running in 5 minutes:**

```bash
# 1. Clone repository
git clone <REPOSITORY_URL>
cd geothermal-piles-pinn

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify installation
pytest

# 5. Start training
python scripts/train.py

# 6. View results
jupyter lab notebooks/03_results.ipynb
```

**Using Makefile (recommended):**

```bash
make setup     # One-time setup
make test      # Run all tests
make train     # Start training
make lint      # Check code quality
make format    # Auto-format code
```

---

## 🔧 Requirements

- **Python 3.13** (required for full compatibility)
- **pip** (Python package manager)
- **Git** (for version control)
- **GPU (optional):** NVIDIA GPU with CUDA 11.8+ for faster training
- **Memory:** 8GB+ RAM recommended

### Check Python version:

```bash
python --version
# or
python3 --version
```

---

## 📦 Installation

### Step 1: Clone Repository

```bash
git clone <REPOSITORY_URL>
cd geothermal-piles-pinn
```

### Step 2: Create Virtual Environment

**On macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Verify installation:**

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
```

### Step 4: (Optional) Install Development Tools

```bash
# For development and testing
pip install -r requirements.txt

# All checks will pass
make check
```

---

## 📁 Project Structure

```
RWTH-Geothermal-Piles-Project/
│
├── README.md
├── requirements.txt
├── .gitignore
├── Makefile
├── CONTRIBUTING.md
│
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── pinn.py
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py
│   └── utils/
│       ├── __init__.py
│       ├── data_loader.py
│       └── plotting.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── tests/
│   ├── __init__.py
│   └── test_models.py
│
├── scripts/
│   └── train.py
│
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory Data Analysis
│   ├── 02_training.ipynb       # Model Training
│   ├── 03_results.ipynb        # Results Analysis
│   └── 04_comparison_fdm.ipynb # Comparison with FDM
│
├── docs/
│   ├── troubleshooting.md
│   ├── theory.md
│   └── setup.md
│
├── models/
└── results/
```

---

## 🏃 Usage

### Training Models

```bash
# Basic training with default parameters
python scripts/train.py

# Training with custom parameters
python scripts/train.py \
    --epochs 1000 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --model forward

# Training with YAML configuration
python scripts/train.py --config configs/training_params.yaml

# Resume training from checkpoint
python scripts/train.py --checkpoint models/checkpoints/epoch_50.ckpt
```

### Evaluation

```bash
# Evaluate trained model
python scripts/evaluate.py --model models/saved/pinn_forward_v1.pth

# Compare with FDM results
python scripts/evaluate.py --model models/saved/pinn_forward_v1.pth --compare_fdm

# Generate plots
python scripts/visualize.py --model models/saved/pinn_forward_v1.pth
```

### Jupyter Notebooks

```bash
# Start Jupyter Lab
jupyter lab

# Start Jupyter Notebook
jupyter notebook

# Run specific notebook
jupyter notebook notebooks/02_training.ipynb
```

### Testing

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_models.py

# Run specific test
pytest tests/test_models.py::TestPINN::test_forward_pass

# Generate coverage report
pytest --cov=src --cov-report=html
```

---

## 📝 Useful Commands

### Using Makefile (Recommended)

```bash
make help          # Show all available commands
make setup         # One-time project setup
make install       # Install/update dependencies
make test          # Run all tests
make test-cov      # Run tests with coverage
make train         # Start training
make evaluate      # Evaluate model
make format        # Auto-format code with Black
make lint          # Check code quality
make check         # Run linting + tests
make notebook      # Start Jupyter Lab
make clean         # Remove cache and build files
make freeze        # Update requirements.txt
make status        # Show project status
```

### Git Workflow

```bash
# Check status
git status

# Create new feature branch
git checkout -b feature/description

# Add changes
git add .
git add src/models/pinn.py          # Specific file

# Commit with message
git commit -m "feat(models): add batch normalization to PINN"

# Push to remote
git push origin feature/description

# Pull latest changes
git pull origin main

# Merge branch locally
git merge feature/description
```

### Dependency Management

```bash
# Update requirements.txt after installing new package
pip install new-package
pip freeze > requirements.txt
git add requirements.txt
git commit -m "chore(deps): add new-package v1.2.3"

# Install all dependencies
pip install -r requirements.txt

# Check installed packages
pip list

# Upgrade all packages (careful!)
pip install --upgrade -r requirements.txt
```

### Code Quality

```bash
# Format code with Black
black src/

# Lint with Flake8
flake8 src/ --max-line-length=100

# Lint with Pylint
pylint src/

# Check types with mypy (if using type hints)
mypy src/

# All checks at once
make check
```

### Virtual Environment

```bash
# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Deactivate
deactivate

# Check which Python is active
which python
```

---

## 💡 Team Guidelines

### ✅ Before First Run

1. ✅ Clone repository and create venv (see [Installation](#-installation))
2. ✅ Install dependencies: `pip install -r requirements.txt`
3. ✅ Run tests to verify setup: `pytest`
4. ✅ Read [CONTRIBUTING.md](CONTRIBUTING.md) for coding standards
5. ✅ Copy `.env.example` to `.env` and customize

### 🔄 During Development

**Always use virtual environment:**

```bash
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows
```

**For every new dependency:**

```bash
pip install new-package
pip freeze > requirements.txt
git add requirements.txt
git commit -m "chore(deps): add new-package"
```

**Before committing:**

```bash
make format   # Format code
make lint     # Check for errors
make test     # Run tests
```

**Commit message format:**

```
feat(scope): brief description       # New feature
fix(scope): brief description        # Bug fix
docs(scope): brief description       # Documentation
refactor(scope): brief description   # Code refactoring
perf(scope): brief description       # Performance improvement
test(scope): brief description       # Add/update tests
chore(scope): brief description      # Build, dependencies, etc.

Examples:
feat(models): add physics-informed loss function
fix(training): resolve NaN loss in epoch 100
docs(readme): add GPU setup instructions
```

**Branch naming:**

```
feature/new-functionality        # New features
bugfix/description-of-bug        # Bug fixes
docs/description                 # Documentation updates
refactor/description             # Code refactoring
```

**Important:** 

- ✅ Always work with activated venv
- ✅ Never push data files or trained models (use `.gitignore`)
- ✅ All team members must use same package versions (`requirements.txt`)
- ✅ Run tests before pushing: `pytest`
- ✅ Format code before pushing: `make format`

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'torch'` | `pip install -r requirements.txt` |
| Different package versions in team | All: `pip install -r requirements.txt` |
| venv won't activate | Check path, Windows users use `venv\Scripts\activate` |
| PyTorch not using GPU | Check CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"` |
| Jupyter doesn't see code changes | Run `%load_ext autoreload` and `%autoreload 2` in notebook |
| Tests fail after changing code | Restart Python: `exit()`, reactivate venv, run tests |
| Git merge conflicts | Discuss with team before pushing to main |
| Port 8888 already in use (Jupyter) | `jupyter lab --port 8889` |
| Out of GPU memory | Reduce batch size or number of workers |

### Getting Help

1. Check [docs/troubleshooting.md](docs/troubleshooting.md)
2. Search existing GitHub Issues
3. Open new Issue with:
   - Clear problem description
   - Steps to reproduce
   - Environment info (Python, PyTorch versions, OS)
   - Error messages
4. Contact team lead or create Discussion

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on:

- Code style and conventions
- Testing requirements
- Documentation standards
- Pull request process
- Commit message conventions

---

## 📚 Documentation

- **Theory:** [docs/theory.md](docs/theory.md) - PINN equations and methods
- **Setup:** [docs/setup.md](docs/setup.md) - Advanced configuration
- **API:** [docs/api.md](docs/api.md) - Code documentation
- **Troubleshooting:** [docs/troubleshooting.md](docs/troubleshooting.md) - Common issues

---

## 📊 Performance Tips

- **GPU Training:** Install CUDA-enabled PyTorch for 10-50x speedup
- **Distributed Training:** Use `torch.nn.DataParallel` for multi-GPU
- **Batch Size:** Start with 32, increase if GPU memory allows
- **Data Loading:** Use `num_workers` in DataLoader for faster I/O
- **Monitoring:** Use TensorBoard or Weights & Biases for training visualization

---

## 📄 License

MIT License

---

## 📞 Contact

- **Issues & Questions:** GitHub Issues
- **Discussions:** GitHub Discussions

---

## 🎉 Acknowledgments

Part of the ENHANCE BIP project.

---

**Happy coding! 🚀**

Last updated: 2026-07-20 | Repository: https://github.com/JohnWhite-CodeSpace/RWTH-Geothermal-Piles-Project
