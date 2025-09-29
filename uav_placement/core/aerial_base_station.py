"""Aerial Base Station implementations for UAV placement optimization."""

import numpy as np
from typing import List, Optional, TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from .channel import Channel
    from .mobile_user import MobileUser


class AerialBaseStation(ABC):
    """
    Abstract base class for aerial base stations (UAVs).
    
    Represents a flying base station that can transmit power and potentially
    move to optimize its position based on utility feedback from mobile users.
    """
    
    def __init__(
        self, 
        position: np.ndarray = None,
        power: float = 1.0,
        fixed_height: bool = True
    ):
        """
        Initialize aerial base station.
        
        Args:
            position: 3D position [x, y, z] in meters/km. Default: [0, 0, 1]
            power: Transmission power in Watts
            fixed_height: Whether the height (z-coordinate) is fixed during optimization
        """
        self.position = np.array([0.0, 0.0, 1.0]) if position is None else np.array(position)
        self.power = power
        self.fixed_height = fixed_height
    
    @abstractmethod
    def update_position(
        self, 
        channel: 'Channel', 
        utility_gradients: np.ndarray, 
        mu_positions: np.ndarray
    ) -> None:
        """
        Update base station position based on utility gradients.
        
        Args:
            channel: Channel model for computing gains
            utility_gradients: Gradients from mobile users (one per MU)
            mu_positions: Positions of mobile users (3 x num_MU)
        """
        pass
    
    def clone(self) -> 'AerialBaseStation':
        """Create a copy of this base station."""
        # This will be overridden in concrete classes
        raise NotImplementedError("clone() must be implemented in concrete classes")
    
    @classmethod
    def clone_at_random_positions(
        cls, 
        template: 'AerialBaseStation',
        powers: np.ndarray,
        region_size: float,
        same_height: bool = True,
        two_d: bool = True
    ) -> List['AerialBaseStation']:
        """
        Create multiple base stations at random positions.
        
        Args:
            template: Template base station to clone
            powers: Array of transmission powers for each BS
            region_size: Size of the square region for random placement
            same_height: Whether to keep the same height as template
            two_d: If False, all BSs will have y-coordinate = 0
            
        Returns:
            List of base stations at random positions
        """
        base_stations = []
        for power in powers:
            bs = template.clone()
            bs.power = power
            if same_height:
                if two_d:
                    bs.position[:2] = np.random.rand(2) * region_size
                else:
                    bs.position[0] = np.random.rand() * region_size
                    bs.position[1] = 0
            else:
                raise NotImplementedError("Variable height not implemented")
            base_stations.append(bs)
        return base_stations
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(pos={self.position}, power={self.power})"


class StochasticAerialBaseStation(AerialBaseStation):
    """
    Aerial base station that uses stochastic gradient descent for position optimization.
    
    Updates its position using gradients computed from utility feedback of mobile users.
    """
    
    def __init__(
        self, 
        position: np.ndarray = None,
        power: float = 1.0,
        fixed_height: bool = True,
        step_size: float = 1.0,
        debug: bool = False
    ):
        """
        Initialize stochastic aerial base station.
        
        Args:
            position: 3D position [x, y, z] in meters/km
            power: Transmission power in Watts
            fixed_height: Whether the height (z-coordinate) is fixed during optimization
            step_size: Step size for gradient descent updates
            debug: Whether to enable debug output
        """
        super().__init__(position, power, fixed_height)
        self.step_size = step_size
        self.debug = debug
    
    def update_position(
        self, 
        channel: 'Channel', 
        utility_gradients: np.ndarray, 
        mu_positions: np.ndarray
    ) -> None:
        """
        Update position using stochastic gradient descent.
        
        Args:
            channel: Channel model for computing gains
            utility_gradients: Utility gradients from mobile users (num_MU,)
            mu_positions: Mobile user positions (3, num_MU)
        """
        num_mu = mu_positions.shape[1]
        if len(utility_gradients) != num_mu:
            raise ValueError("Number of utility gradients must match number of MU positions")
        
        # Compute position gradient
        stochastic_position_gradient = np.zeros(3)
        
        for i in range(num_mu):
            # Gradient of received power w.r.t. BS position
            power_gradient = self.power * channel.gain_gradient(
                self.position, mu_positions[:, i]
            )
            
            # Chain rule: utility gradient * power gradient
            stochastic_position_gradient += power_gradient * utility_gradients[i]
        
        # Average over mini-batch
        stochastic_position_gradient /= num_mu
        
        if self.debug:
            step = self.step_size * stochastic_position_gradient
            print(f"Position gradient: {stochastic_position_gradient}")
            print(f"Step: {step}")
            if np.linalg.norm(step) > 100:
                print("Warning: Large step detected!")
        
        # Apply height constraint
        if self.fixed_height:
            stochastic_position_gradient[2] = 0
        
        # Update position
        self.position += self.step_size * stochastic_position_gradient
    
    def clone(self) -> 'StochasticAerialBaseStation':
        """Create a copy of this base station."""
        return StochasticAerialBaseStation(
            position=self.position.copy(),
            power=self.power,
            fixed_height=self.fixed_height,
            step_size=self.step_size,
            debug=self.debug
        )


class KmeansAerialBaseStation(AerialBaseStation):
    """
    Aerial base station that uses k-means-like positioning.
    
    Moves toward the centroid of the mobile users it serves.
    """
    
    def __init__(
        self, 
        position: np.ndarray = None,
        power: float = 1.0,
        fixed_height: bool = True
    ):
        """
        Initialize k-means aerial base station.
        
        Args:
            position: 3D position [x, y, z] in meters/km
            power: Transmission power in Watts
            fixed_height: Whether the height (z-coordinate) is fixed during optimization
        """
        super().__init__(position, power, fixed_height)
    
    def update_position(
        self, 
        channel: 'Channel', 
        utility_gradients: np.ndarray, 
        mu_positions: np.ndarray
    ) -> None:
        """
        Update position toward centroid of served mobile users.
        
        For k-means, we move toward the centroid of all mobile users,
        weighted by their utility gradients (serving as assignment weights).
        
        Args:
            channel: Channel model (not used in k-means)
            utility_gradients: Utility gradients from mobile users (num_MU,)
            mu_positions: Mobile user positions (3, num_MU)
        """
        num_mu = mu_positions.shape[1]
        if len(utility_gradients) != num_mu:
            raise ValueError("Number of utility gradients must match number of MU positions")
        
        if num_mu == 0:
            return
        
        # Compute weighted centroid
        weights = np.abs(utility_gradients)  # Use absolute values as weights
        total_weight = np.sum(weights)
        
        if total_weight > 0:
            weighted_centroid = np.sum(mu_positions * weights[np.newaxis, :], axis=1) / total_weight
            
            # Apply height constraint
            if self.fixed_height:
                weighted_centroid[2] = self.position[2]
            
            # Move toward centroid (could add learning rate here)
            self.position = weighted_centroid
    
    def clone(self) -> 'KmeansAerialBaseStation':
        """Create a copy of this base station."""
        return KmeansAerialBaseStation(
            position=self.position.copy(),
            power=self.power,
            fixed_height=self.fixed_height
        )