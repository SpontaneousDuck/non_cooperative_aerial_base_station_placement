

# Non-cooperative Aerial Base Station Placement via Stochastic Optimization

This repository contains both MATLAB and Python implementations of algorithms for optimizing the placement of aerial base stations (UAVs) to maximize utility for mobile users in a non-cooperative setting.

**Original Paper:** ["Non-cooperative Aerial Base Station Placement via Stochastic Optimization"](https://arxiv.org/abs/1905.03988) by Daniel Romero and Geert Leus.

**Video:** https://www.youtube.com/watch?v=ZNQiVQ3TtGI

## Repository Structure

This repository is organized into two main implementations:

### 📁 `matlab/` - Original MATLAB Implementation
The reference implementation containing the original research code and experimental framework.

- Complete MATLAB codebase with all original experiments
- Comprehensive experimental scenarios (1D and 2D placement)
- Advanced visualization and analysis tools
- Suitable for research and algorithm development

### 📁 `python/` - Modern Python Implementation  
A complete Python conversion with simplified APIs for external program integration.

- Clean, modern Python interface
- No MATLAB dependencies (uses NumPy, SciPy, matplotlib)
- Command-line interface for easy integration
- Simplified API for external programs
- Comprehensive testing and documentation

## Quick Start

### Python Implementation (Recommended for Integration)

```bash
# Navigate to Python implementation
cd python

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run example
python example.py
```

### MATLAB Implementation (Original Research Code)

```matlab
% Navigate to MATLAB implementation
cd matlab

% Initialize and run
gsimStartup
gsim(0)  % Run default experiment
```

## Which Implementation to Use?

**Choose Python if you:**
- Want to integrate UAV placement optimization into external programs
- Prefer modern, clean APIs and CLI interfaces
- Need to avoid MATLAB dependencies
- Want comprehensive testing and documentation

**Choose MATLAB if you:**
- Are conducting research and need the complete experimental framework
- Want to reproduce exact results from the original paper
- Need access to all original experiments and analysis tools
- Are familiar with the MATLAB ecosystem

## Algorithm Overview

Both implementations provide the same core algorithms for non-cooperative optimization where each aerial base station independently optimizes its position based on utility feedback from mobile users.

### Key Features

- **Multiple Optimization Methods**: Stochastic gradient descent and k-means-based approaches
- **Flexible Utility Functions**: Sigmoid and leaky ReLU utility functions  
- **Realistic Channel Models**: Free space propagation with configurable parameters
- **Visualization**: Built-in plotting and result visualization

### Key Components

- **Mobile Users**: Entities with utility functions based on received signal power
- **Aerial Base Stations**: Flying base stations that can adjust their positions
- **Channel Models**: Communication channel models (free space propagation)
- **Placement Simulator**: Orchestrates the optimization process

## Examples

### Python API

```python
from uav_placement import UAVPlacementOptimizer

# Simple optimization
results = UAVPlacementOptimizer.quick_optimize(
    num_base_stations=3,
    num_mobile_users=50
)

print(f"Final utility: {results['results']['final_utility']}")
print("Optimal positions:", results['results']['final_bs_positions'])
```

### MATLAB Experiments

```matlab
% Run 2D stochastic optimization  
gsim(0, 2008, 100)

% Compare optimization methods
gsim(0, 2010, 50)
```

### Command Line Interface (Python)

```bash
# Basic optimization
uav-placement --num-bs 3 --num-users 50 --iterations 100

# Advanced configuration with output files
uav-placement --method stochastic --utility sigmoid \
    --step-size 0.01 --batch-size 20 \
    --output results.json --plot optimization.png
```

## Installation and Setup

### Python Implementation

```bash
# Clone the repository
git clone https://github.com/SpontaneousDuck/non_cooperative_aerial_base_station_placement.git
cd non_cooperative_aerial_base_station_placement/python

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### MATLAB Implementation

```matlab
% Clone and navigate to MATLAB directory
cd non_cooperative_aerial_base_station_placement/matlab

% Run initialization
gsimStartup
```

## Documentation

- **[Python Implementation Guide](python/README.md)** - Detailed Python API documentation
- **[MATLAB Implementation Guide](matlab/README.md)** - Original MATLAB research code documentation

## Citation

If you use this code in your research, please cite:

```bibtex
@article{romero2019non,
  title={Non-cooperative aerial base station placement via stochastic optimization},
  author={Romero, Daniel and Leus, Geert},
  journal={IEEE Transactions on Wireless Communications},
  year={2019},
  publisher={IEEE}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
