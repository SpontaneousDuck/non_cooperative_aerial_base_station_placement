# Python Implementation

This directory contains the Python implementation of the non-cooperative aerial base station placement optimization algorithms, converted from the original MATLAB codebase with simplified APIs for external program integration.

## Features

- **Simplified API**: Clean, modern Python interface for external programs
- **Multiple Optimization Methods**: Stochastic gradient descent and k-means-based approaches
- **Flexible Utility Functions**: Sigmoid and leaky ReLU utility functions
- **Realistic Channel Models**: Free space propagation with configurable parameters
- **Command-Line Interface**: Run optimizations from the command line
- **Visualization**: Built-in plotting and result visualization
- **No MATLAB Dependencies**: Pure Python implementation using NumPy, SciPy, and matplotlib

## Installation

```bash
# Navigate to the python directory
cd python

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Quick Start

### Python API

```python
from uav_placement import UAVPlacementOptimizer

# Simple optimization
optimizer = UAVPlacementOptimizer()
results = optimizer.optimize(
    num_base_stations=3,
    num_mobile_users=50,
    region_size=7.0,
    num_iterations=100
)

print(f"Final utility: {results['results']['final_utility']}")
print("Optimal positions:", results['results']['final_bs_positions'])

# Save results and plot
optimizer.save_results('results.json')
optimizer.plot_results(save_path='optimization.png')
```

### Command Line Interface

```bash
# Basic optimization
uav-placement --num-bs 3 --num-users 50 --iterations 100

# Advanced parameters
uav-placement --num-bs 5 --num-users 100 --method stochastic \
    --utility sigmoid --step-size 0.01 --batch-size 20 \
    --output results.json --plot optimization.png

# K-means optimization
uav-placement --num-bs 3 --num-users 50 --method kmeans \
    --iterations 50 --output kmeans_results.json
```

## Package Structure

```
python/
├── uav_placement/           # Main package
│   ├── core/               # Core algorithm implementations
│   │   ├── mobile_user.py     # MobileUser class
│   │   ├── aerial_base_station.py  # Base station classes
│   │   ├── channel.py         # Channel models
│   │   └── placement_simulator.py  # Main simulator
│   ├── optimization/       # High-level optimizer interface
│   │   └── optimizer.py       # UAVPlacementOptimizer
│   ├── utils/              # Utility functions
│   ├── tests/              # Test suite
│   └── cli.py              # Command-line interface
├── example.py              # Usage examples
├── setup.py               # Package setup
├── requirements.txt       # Dependencies
└── pyproject.toml         # Project configuration
```

## Algorithm Overview

The package implements non-cooperative optimization where each aerial base station independently optimizes its position based on utility feedback from mobile users. Two main approaches are supported:

1. **Stochastic Gradient Descent**: Base stations use gradients of user utility functions to update positions
2. **K-means Clustering**: Base stations move toward centroids of users they serve

### Key Components

- **Mobile Users**: Entities with utility functions based on received signal power
- **Aerial Base Stations**: Flying base stations that can adjust their positions
- **Channel Model**: Free space propagation with realistic parameters
- **Placement Simulator**: Orchestrates the optimization process

## Examples

See `example.py` for a complete usage example:

```bash
python example.py
```

### External Program Integration

```python
# For Python programs
from uav_placement import optimize_uav_placement
results = optimize_uav_placement(num_base_stations=3, num_mobile_users=50)

# For non-Python programs via subprocess
import subprocess
result = subprocess.run([
    'uav-placement', '--num-bs', '3', '--num-users', '50', 
    '--output', 'results.json', '--quiet'
])
```

## API Reference

### Main Classes

- `UAVPlacementOptimizer`: High-level interface for optimization
- `MobileUser`: Represents mobile users with utility functions
- `AerialBaseStation`: Base class for aerial base stations
- `StochasticAerialBaseStation`: Gradient-based base station
- `KmeansAerialBaseStation`: K-means-based base station
- `Channel`: Communication channel models
- `PlacementSimulator`: Main simulation engine

### Key Parameters

- `num_base_stations`: Number of aerial base stations
- `num_mobile_users`: Number of mobile users in the region
- `region_size`: Size of the square deployment region (km)
- `optimization_method`: 'stochastic' or 'kmeans'
- `utility_function`: 'sigmoid' or 'leakyRelu'
- `num_iterations`: Number of optimization steps
- `step_size`: Learning rate for gradient descent

## Testing

Run the test suite:

```bash
pytest uav_placement/tests/
```

## Performance Comparison with MATLAB

The Python implementation maintains the same algorithmic behavior as the original MATLAB code while providing:

- **Better Performance**: Optimized NumPy operations
- **Easier Integration**: Simple API for external programs
- **Enhanced Usability**: Built-in visualization and result export
- **Modern Practices**: Type hints, comprehensive testing, clean architecture

## Dependencies

- **numpy**: Numerical computations
- **scipy**: Scientific computing
- **matplotlib**: Plotting and visualization
- **scikit-learn**: Machine learning utilities (optional)

## Citation

If you use this Python implementation in your research, please cite:

```bibtex
@article{romero2019non,
  title={Non-cooperative aerial base station placement via stochastic optimization},
  author={Romero, Daniel and Leus, Geert},
  journal={IEEE Transactions on Wireless Communications},
  year={2019},
  publisher={IEEE}
}
```

## Original Paper

["Non-cooperative Aerial Base Station Placement via Stochastic Optimization"](https://arxiv.org/abs/1905.03988)

## Video Demonstration

[YouTube Video](https://www.youtube.com/watch?v=ZNQiVQ3TtGI)