"""
Non-cooperative Aerial Base Station Placement via Stochastic Optimization

This package implements stochastic optimization algorithms for positioning 
aerial base stations (UAVs) to maximize utility for mobile users.

Based on the paper: "Non-cooperative Aerial Base Station Placement via 
Stochastic Optimization" by Daniel Romero and Geert Leus.
Paper: https://arxiv.org/abs/1905.03988
"""

__version__ = "0.1.0"

from .core.mobile_user import MobileUser
from .core.aerial_base_station import AerialBaseStation, StochasticAerialBaseStation
from .core.channel import Channel
from .core.placement_simulator import PlacementSimulator
from .optimization.optimizer import UAVPlacementOptimizer

__all__ = [
    "MobileUser",
    "AerialBaseStation",
    "StochasticAerialBaseStation", 
    "Channel",
    "PlacementSimulator",
    "UAVPlacementOptimizer",
]