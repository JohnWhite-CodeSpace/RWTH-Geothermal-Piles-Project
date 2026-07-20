# Contributing Guidelines

Thank you for your interest in this project! Below are guidelines to ensure efficient collaboration across our international team.

---

## 📋 Before You Start

1. ✅ Read [README.md](README.md)
2. ✅ Set up your environment according to installation instructions
3. ✅ Run `pytest` - all tests should pass
4. ✅ Review existing Issues on GitHub

---

## 🔀 Git Workflow

### 1. Sync with Main Repository

```bash
# Update locally
git fetch origin
git merge origin/main

# Or better: use rebase
git rebase origin/main
```

### 2. Create New Branch

```bash
# Start from main
git checkout main
git pull origin main

# Create new branch
git checkout -b feature/my-feature
# or
git checkout -b bugfix/bug-name
# or
git checkout -b docs/documentation-update
```

### 3. Commit Messages

**Commit message convention:**

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code formatting (spaces, semicolons, etc.)
- `refactor`: Code refactoring without functional changes
- `perf`: Performance improvements
- `test`: Add or update tests
- `chore`: Build, dependencies, configuration changes

**Examples:**

```bash
git commit -m "feat(models): add batch normalization to PINN"
git commit -m "fix(training): resolve NaN loss at epoch 100"
git commit -m "docs(readme): add troubleshooting section"
git commit -m "test(utils): add data loader tests"
git commit -m "chore(deps): update PyTorch to 2.1.1"
```

### 4. Push and Pull Request

```bash
# Ensure code is formatted and tested
make format
make lint
make test

# Push to remote
git push origin feature/my-feature

# Open Pull Request on GitHub
# - Describe what you changed
# - Link to relevant Issue (if applicable)
# - Include screenshots/results if visual changes
# - Reference any breaking changes
```

---

## 💻 Code Conventions

### Python Style: PEP 8 + Black

```bash
# Format code
black src/

# Check quality
flake8 src/
pylint src/

# All checks at once
make check
```

### Code Structure

```python
# Imports at top (alphabetically organized)
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

# Constants
DEFAULT_BATCH_SIZE = 32
RANDOM_SEED = 42

# Classes and functions
class MyModel(nn.Module):
    """Brief description of the class."""
    
    def __init__(self, input_dim: int):
        """
        Initialize the model.
        
        Args:
            input_dim: Input dimension
        """
        super().__init__()
        self.layer = nn.Linear(input_dim, 64)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.layer(x)


def train_model(model: nn.Module, data: List) -> float:
    """
    Train the model.
    
    Args:
        model: Model to train
        data: List of training data
    
    Returns:
        Final loss value
    """
    loss = 0.0
    # ... training code
    return loss
```

### Type Hints (Always Use!)

```python
# ❌ Bad
def load_data(path):
    return data

# ✅ Good
def load_data(path: str) -> np.ndarray:
    """Load data from file."""
    return data
```

### Docstring Format

```python
def calculate_pressure(temperature: float, depth: float) -> float:
    """
    Calculate excess pore-water pressure in soil.
    
    Implements equation from Fuentes et al. (2016).
    
    Args:
        temperature: Temperature in °C
        depth: Depth in meters
    
    Returns:
        Pore pressure in Pa
    
    Raises:
        ValueError: If temperature < 0 or depth < 0
    
    Example:
        >>> pressure = calculate_pressure(30, 10)
        >>> print(f"Pressure: {pressure:.2f} Pa")
        Pressure: 150000.00 Pa
    """
    if temperature < 0 or depth < 0:
        raise ValueError("Temperature and depth must be >= 0")
    
    # Implementation
    pressure = temperature * depth * 500
    return pressure
```

---

## 🧪 Testing

### Writing Tests

```python
# tests/test_models.py
import pytest
import torch
from src.models.pinn import PINN

class TestPINN:
    """Test PINN model."""
    
    @pytest.fixture
    def model(self):
        """Fixture: PINN model instance."""
        return PINN(input_dim=3, output_dim=2)
    
    def test_forward_pass(self, model):
        """Test forward pass shape."""
        x = torch.randn(10, 3)
        output = model(x)
        assert output.shape == (10, 2)
    
    def test_model_training(self, model):
        """Test if model parameters update during training."""
        optimizer = torch.optim.Adam(model.parameters())
        x = torch.randn(5, 3)
        y = torch.randn(5, 2)
        
        initial_params = [p.clone() for p in model.parameters()]
        
        # Forward pass
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Verify parameters changed
        for initial, current in zip(initial_params, model.parameters()):
            assert not torch.allclose(initial, current)
```

### Running Tests

```bash
# All tests
pytest

# Specific file
pytest tests/test_models.py

# Specific test function
pytest tests/test_models.py::TestPINN::test_forward_pass

# With verbose output
pytest -v

# With coverage
pytest --cov=src
```

**Test coverage goal:** Aim for at least 80% coverage of new code.

---

## 📦 Updating Dependencies

When adding new libraries:

```bash
# 1. Install
pip install new-package

# 2. Update requirements.txt
pip freeze > requirements.txt

# 3. Commit
git add requirements.txt
git commit -m "chore(deps): add new-package v1.2.3"

# 4. Notify team
# Everyone: pip install -r requirements.txt
```

**Important:** Always pin versions in `requirements.txt` for reproducibility!

---

## 📝 Documentation

### Module Docstrings

```python
"""
Physics-Informed Neural Network module.

Implements PINN architecture with automatic differentiation
for solving coupled thermo-hydraulic equations.

Example:
    >>> model = PINN(input_dim=3, hidden_dim=128)
    >>> x = torch.randn(32, 3)
    >>> output = model(x)
"""
```

### Inline Comments

Use sparingly - code should be self-explanatory. Comments explain WHY, not WHAT.

```python
# ❌ Bad
x = x + 1  # Add 1 to x

# ✅ Good
# Offset temperature by 1°C to align with reference data
temperature = temperature + 1
```

---

## 🐛 Reporting Bugs

When opening an Issue:

1. **Title:** Clear and concise
   - ✅ "NaN loss during training on GPU with float32"
   - ❌ "Something doesn't work"

2. **Description:** Include:
   - What happened?
   - What should happen?
   - Steps to reproduce (step-by-step)
   - Your environment (Python, PyTorch version, OS)

3. **Reproducible Example:**

```python
import torch
from src.models.pinn import PINN

# Minimal code to reproduce
model = PINN(input_dim=3)
x = torch.randn(10, 3)
output = model(x)
# Output: RuntimeError: ...
```

---

## 📊 Code Review Checklist

Before opening a PR, ensure:

- [ ] Code formatted: `make format`
- [ ] Linting passes: `make lint`
- [ ] Tests pass: `make test`
- [ ] Tests added for new features
- [ ] `requirements.txt` updated if needed
- [ ] Documentation updated
- [ ] Commit messages are descriptive
- [ ] No merge conflicts
- [ ] No hardcoded paths or credentials

---

## 📚 Resources

- [PEP 8 Style Guide](https://pep8.org/)
- [Python Type Hints](https://realpython.com/python-type-hints/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Semantic Versioning](https://semver.org/)
- [Testing Best Practices](https://docs.pytest.org/en/stable/index.html)

---

## 🙋 Questions?

1. Check documentation in `docs/`
2. Search Issue history
3. Open Discussion on GitHub
4. Contact team lead

---

## 📌 Summary: Contribution Workflow

```bash
# 1. Create feature branch
git checkout -b feature/my-feature

# 2. Make changes and commit
git add .
git commit -m "feat(scope): description"

# 3. Format and test
make format
make test

# 4. Push and create PR
git push origin feature/my-feature

# 5. Address review comments

# 6. Merge when approved
```

---

**Thank you for contributing! 🎉**

Last updated: 2026-07-20
