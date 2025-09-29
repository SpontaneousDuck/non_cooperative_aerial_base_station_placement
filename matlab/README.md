# MATLAB Implementation

This directory contains the original MATLAB implementation of the non-cooperative aerial base station placement optimization algorithms.

## Overview

The MATLAB implementation provides the reference algorithms and experimental framework described in the paper "Non-cooperative Aerial Base Station Placement via Stochastic Optimization" by Daniel Romero and Geert Leus.

## Requirements

- MATLAB R2016b or later
- Signal Processing Toolbox
- Statistics and Machine Learning Toolbox (optional, for some experiments)

## Structure

- **`gsim.m`** - Main entry point for running experiments
- **`gsimStartup.m`** - Initialization script
- **`Execution/`** - Core execution framework
- **`Experiments/`** - Experiment definitions and configurations
- **`ProcessingBlocks/`** - Core algorithm implementations
  - `MobileUser.m` - Mobile user utility functions
  - `AerialBaseStation.m` - Base station positioning algorithms
  - `Channel.m` - Communication channel models
  - `PlacementSimulator.m` - Main simulation engine
- **`Simulators/`** - High-level simulation orchestrators
- **`utilities/`** - Helper functions and utilities
- **`Personal/`** - Personal configurations and customizations

## Usage

### Basic Execution

```matlab
% Add paths and initialize
gsimStartup

% Run default experiment
gsim(0)

% Run specific experiment with parameters
gsim(0, 1001, 200)  % Execute experiment 1001 with 200 iterations

% Plot results without re-execution
gsim(1, 1001)  % Plot results of experiment 1001
```

### Key Experiments

The `Experiments/UAVPlacementExperiments.m` file contains various scenarios:

- **Experiment 1001-1005**: 1D placement scenarios
- **Experiment 2001-2012**: 2D placement with different configurations
- **Experiment 9901-9902**: Utility function analysis

### Configuration

Edit the parameters in `gsim.m` to customize:

- `defaultExperimentClassName`: Choose experiment file
- `defaultExperimentIndex`: Default experiment to run
- `defaultNiter`: Default number of iterations
- Display and plotting settings

## Key Classes

### MobileUser
Represents mobile users with configurable utility functions:
- Sigmoid utility
- Leaky ReLU utility
- Power aggregation methods

### AerialBaseStation
Base station positioning algorithms:
- `StochasticAerialBaseStation`: Gradient-based positioning
- `KmeansAerialBaseStation`: K-means clustering approach

### Channel
Communication channel models:
- Free space propagation
- Realistic antenna patterns and path loss

## Examples

```matlab
% Run 2D stochastic optimization
gsim(0, 2008, 100)

% Compare stochastic vs k-means
gsim(0, 2010, 50)

% Analyze utility functions
gsim(0, 9901)
```

## Citation

If you use this MATLAB implementation in your research, please cite:

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