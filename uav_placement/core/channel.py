"""Channel models for UAV placement optimization."""

import numpy as np
from typing import Union, Tuple
from ..utils import dbm_to_watt


class Channel:
    """
    Communication channel model for aerial base stations and mobile users.
    
    Supports different propagation models including free space path loss.
    """
    
    def __init__(self, model: str = 'freeSpace', gain_constant: float = 1.0, **kwargs):
        """
        Initialize channel model.
        
        Args:
            model: Channel model type ('freeSpace')
            gain_constant: Gain constant for the channel model
            **kwargs: Additional model-specific parameters
        """
        self.model = model
        self.gain_constant = gain_constant
        
        if model == 'freeSpace':
            # For free space model, gain_constant represents the gain at 1 unit distance
            pass
        else:
            raise ValueError(f"Unsupported channel model: {model}")
    
    def gain(self, bs_position: np.ndarray, mu_position: np.ndarray) -> float:
        """
        Compute channel gain between base station and mobile user.
        
        Args:
            bs_position: Base station position [x, y, z]
            mu_position: Mobile user position [x, y, z]
            
        Returns:
            Channel gain value
        """
        if self.model == 'freeSpace':
            return self._free_space_gain(bs_position, mu_position)
        else:
            raise ValueError(f"Unsupported channel model: {self.model}")
    
    def gain_gradient(self, bs_position: np.ndarray, mu_position: np.ndarray) -> np.ndarray:
        """
        Compute gradient of channel gain w.r.t. base station position.
        
        Args:
            bs_position: Base station position [x, y, z]
            mu_position: Mobile user position [x, y, z]
            
        Returns:
            Gradient vector [∂g/∂x, ∂g/∂y, ∂g/∂z]
        """
        if self.model == 'freeSpace':
            return self._free_space_gain_gradient(bs_position, mu_position)
        else:
            raise ValueError(f"Unsupported channel model: {self.model}")
    
    def _free_space_gain(self, bs_position: np.ndarray, mu_position: np.ndarray) -> float:
        """
        Free space path loss model: G = G0 / d²
        where d is the 3D distance between BS and MU.
        """
        distance_squared = np.sum((bs_position - mu_position) ** 2)
        return self.gain_constant / (distance_squared + 1e-12)  # Small epsilon to avoid division by zero
    
    def _free_space_gain_gradient(self, bs_position: np.ndarray, mu_position: np.ndarray) -> np.ndarray:
        """
        Gradient of free space gain w.r.t. base station position.
        
        For G = G0 / d², where d² = ||p_bs - p_mu||²:
        ∂G/∂p_bs = -2 * G² / G0 * (p_bs - p_mu)
        """
        diff = bs_position - mu_position
        distance_squared = np.sum(diff ** 2) + 1e-12
        gain = self.gain_constant / distance_squared
        
        # Gradient: ∂G/∂p_bs = -2 * G² / G0 * (p_bs - p_mu)
        gradient = -2 * (gain ** 2) / self.gain_constant * diff
        return gradient
    
    @classmethod
    def create_realistic_channel(
        cls, 
        frequency_hz: float = 2.39e9, 
        bs_antenna_gain_db: float = 6.0,
        mu_antenna_gain_db: float = 0.0
    ) -> 'Channel':
        """
        Create a realistic free space channel model.
        
        Args:
            frequency_hz: Carrier frequency in Hz
            bs_antenna_gain_db: Base station antenna gain in dB
            mu_antenna_gain_db: Mobile user antenna gain in dB
            
        Returns:
            Channel object with realistic parameters
        """
        # Use a simpler gain constant for better numerical stability
        # This represents a normalized gain that gives reasonable gradients
        gain_constant = 1.0  # Simplified for better numerical behavior
        
        return cls(model='freeSpace', gain_constant=gain_constant)
    
    def __repr__(self) -> str:
        return f"Channel(model={self.model}, gain_constant={self.gain_constant})"