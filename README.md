# pyCoastal

A Python module for Coastal Engineering calculations.

Copyright (c) 2025 Stefano Biondi
Licensed under the MIT License. See LICENSE file for details.

# Installation
# Clone the repository and navigate to the main folder
```bash
git clone https://github.com/stebiondi/pyCoastal.git
cd pyCoastal
```
# (Optional) Create a virtual environment for clean dependencies
```bash
python -m venv venv
```
# Activate it:
On Windows PowerShell:
```bash
venv\Scripts\Activate.ps1
```
On Linux/macOS:

```bash
source venv/bin/activate
```

# Install build tools
```bash
pip install --upgrade pip setuptools wheel build
```

# Build your package (creates dist/ folder)
```bash
python -m build
```

# (Optional) Check your distribution packages
```bash
pip install twine
twine check dist/*
```

# Install pyCoastal in editable (dev) mode
```bash
pip install -e 
```

# Test it in a Python session:
```bash
python - <<EOF
import pyCoastal as cs
print(cs.dispersion(T=8, h=5))
EOF
```

# or soon available:

# Install via pip:

```bash
pip install pyCoastal
```
