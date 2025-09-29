"""High-level optimizer interface for UAV placement."""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union
import json

from ..core.mobile_user import MobileUser
from ..core.aerial_base_station import StochasticAerialBaseStation, KmeansAerialBaseStation
from ..core.channel import Channel
from ..core.placement_simulator import PlacementSimulator
from ..utils import dbm_to_watt, watt_to_dbm


class UAVPlacementOptimizer:
    """
    High-level interface for UAV placement optimization.
    
    This class provides a simple API for external programs to use the 
    aerial base station placement optimization algorithms.
    """
    
    def __init__(self):
        """Initialize the optimizer."""
        self.results = None
        self.simulator = None
    
    def _create_even_grid(self, num_users: int, region_size: float, height: float) -> np.ndarray:
        """
        Create an even grid of positions for mobile users.
        
        Args:
            num_users: Number of users (must be a perfect square)
            region_size: Size of the square region (km)
            height: Height of users (z-coordinate)
            
        Returns:
            Array of shape (3, num_users) with evenly spaced positions
        """
        grid_size = int(np.sqrt(num_users))
        
        # Create evenly spaced coordinates
        x = np.linspace(0, region_size, grid_size)
        y = np.linspace(0, region_size, grid_size)
        
        # Create meshgrid
        xx, yy = np.meshgrid(x, y)
        
        # Flatten and create position array
        positions = np.zeros((3, num_users))
        positions[0, :] = xx.flatten()
        positions[1, :] = yy.flatten()
        positions[2, :] = height
        
        return positions
    
    def optimize(
        self,
        num_base_stations: int = 3,
        num_mobile_users: int = 50,
        region_size: float = 7.0,
        bs_powers_dbm: List[float] = None,
        bs_height: float = 0.03,  # km
        optimization_method: str = 'stochastic',
        utility_function: str = 'sigmoid',
        power_threshold_dbm: float = -91.0,
        power_upper_dbm: float = -89.0,
        num_iterations: int = 100,
        mini_batch_size: Optional[int] = None,
        step_size: float = 0.01,
        frequency_ghz: float = 2.39,
        random_seed: Optional[int] = None,
        user_distribution: str = 'random'
    ) -> Dict:
        """
        Run UAV placement optimization.
        
        Args:
            num_base_stations: Number of aerial base stations
            num_mobile_users: Number of mobile users
            region_size: Size of the square region (km)
            bs_powers_dbm: List of BS transmission powers in dBm (auto-generated if None)
            bs_height: Height of base stations in km
            optimization_method: 'stochastic' or 'kmeans'
            utility_function: 'sigmoid' or 'leakyRelu'
            power_threshold_dbm: Lower power threshold for utility function
            power_upper_dbm: Upper power threshold for sigmoid utility
            num_iterations: Number of optimization iterations
            mini_batch_size: Mini-batch size for stochastic updates (None = full batch)
            step_size: Step size for gradient descent
            frequency_ghz: Carrier frequency in GHz
            random_seed: Random seed for reproducibility
            user_distribution: 'random' or 'even' - How to distribute mobile users
            
        Returns:
            Dictionary with optimization results and metrics
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Set default powers if not provided
        if bs_powers_dbm is None:
            bs_powers_dbm = [7 + i for i in range(num_base_stations)]
        
        if len(bs_powers_dbm) != num_base_stations:
            raise ValueError("Length of bs_powers_dbm must match num_base_stations")
        
        # Validate user distribution
        if user_distribution not in ['random', 'even']:
            raise ValueError(f"user_distribution must be 'random' or 'even', got '{user_distribution}'")
        
        # For even distribution, check that num_mobile_users is a perfect square
        if user_distribution == 'even':
            grid_size = int(np.sqrt(num_mobile_users))
            if grid_size * grid_size != num_mobile_users:
                raise ValueError(
                    f"For even distribution, num_mobile_users must be a perfect square. "
                    f"Got {num_mobile_users}, nearest perfect squares are {grid_size**2} and {(grid_size+1)**2}"
                )
        
        # Convert power thresholds to Watts
        power_start = dbm_to_watt(power_threshold_dbm)
        power_stop = dbm_to_watt(power_upper_dbm)
        
        # Create mobile users
        template_mu = MobileUser(
            utility_type=utility_function,
            aggregate_type='smoothMaxPower',
            power_start=power_start,
            power_stop=power_stop,
            use_dbm=True
        )
        
        if user_distribution == 'random':
            mobile_users = MobileUser.clone_at_random_positions(
                template_mu, num_mobile_users, region_size, same_height=True, two_d=True
            )
        else:  # even distribution
            even_positions = self._create_even_grid(num_mobile_users, region_size, template_mu.position[2])
            mobile_users = MobileUser.clone_at_positions(template_mu, even_positions)
        
        # Create aerial base stations
        if optimization_method == 'stochastic':
            template_bs = StochasticAerialBaseStation(
                position=np.array([0, 0, bs_height]),
                fixed_height=True,
                step_size=step_size
            )
        elif optimization_method == 'kmeans':
            template_bs = KmeansAerialBaseStation(
                position=np.array([0, 0, bs_height]),
                fixed_height=True
            )
            # For k-means, change MU utility type
            for mu in mobile_users:
                mu.utility_type = 'kmeans'
        else:
            raise ValueError(f"Unknown optimization method: {optimization_method}")
        
        # Convert powers and create BSs
        bs_powers_watt = [dbm_to_watt(p) for p in bs_powers_dbm]
        aerial_base_stations = template_bs.__class__.clone_at_random_positions(
            template_bs, bs_powers_watt, region_size / 3, same_height=True, two_d=True
        )
        
        # Create realistic channel model
        channel = Channel.create_realistic_channel(
            frequency_hz=frequency_ghz * 1e9,
            bs_antenna_gain_db=6.0,
            mu_antenna_gain_db=0.0
        )
        
        # Create and run simulator
        self.simulator = PlacementSimulator(
            num_steps=num_iterations,
            mini_batch_size=mini_batch_size,
            mobile_users=mobile_users,
            aerial_base_stations=aerial_base_stations,
            channel=channel
        )
        
        print(f"Running {optimization_method} optimization with {num_base_stations} BSs and {num_mobile_users} MUs...")
        position_history, metrics = self.simulator.generate_sample_path()
        
        # Compile results
        final_positions = metrics['final_positions']
        final_bs_positions_dict = {}
        for i in range(num_base_stations):
            final_bs_positions_dict[f'bs_{i+1}'] = {
                'x_km': float(final_positions[0, i]),
                'y_km': float(final_positions[1, i]),
                'z_km': float(final_positions[2, i]),
                'power_dbm': float(bs_powers_dbm[i])
            }
        
        self.results = {
            'optimization_method': optimization_method,
            'parameters': {
                'num_base_stations': num_base_stations,
                'num_mobile_users': num_mobile_users,
                'region_size_km': region_size,
                'bs_height_km': bs_height,
                'utility_function': utility_function,
                'power_threshold_dbm': power_threshold_dbm,
                'power_upper_dbm': power_upper_dbm,
                'num_iterations': num_iterations,
                'mini_batch_size': mini_batch_size,
                'step_size': step_size,
                'frequency_ghz': frequency_ghz
            },
            'results': {
                'final_utility': float(metrics['final_utility']),
                'converged': bool(metrics['converged']),
                'final_bs_positions': final_bs_positions_dict,
                'utility_improvement': float(metrics['utility_history'][-1] - metrics['utility_history'][0]) if metrics['utility_history'] else 0.0
            },
            'convergence': {
                'utility_history': [float(u) for u in metrics['utility_history']],
                'gradient_norms': [float(g) for g in metrics['gradient_norms']]
            }
        }
        
        return self.results
    
    def get_optimal_positions(self) -> Dict[str, Dict[str, float]]:
        """
        Get the optimal base station positions.
        
        Returns:
            Dictionary mapping base station IDs to their optimal positions
        """
        if self.results is None:
            raise ValueError("No optimization results available. Run optimize() first.")
        
        return self.results['results']['final_bs_positions']
    
    def save_results(self, filename: str) -> None:
        """
        Save optimization results to JSON file.
        
        Args:
            filename: Path to save the results
        """
        if self.results is None:
            raise ValueError("No optimization results available. Run optimize() first.")
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Results saved to {filename}")
    
    def plot_results(self, save_path: Optional[str] = None):
        """
        Plot the optimization results.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if self.simulator is None:
            raise ValueError("No simulation results available. Run optimize() first.")
        
        region_size = self.results['parameters']['region_size_km']
        optimization_method = self.results['optimization_method']
        return self.simulator.plot_results(region_size=region_size, save_path=save_path, optimization_method=optimization_method)
    
    @classmethod
    def quick_optimize(
        cls,
        num_base_stations: int = 3,
        num_mobile_users: int = 50,
        region_size: float = 7.0,
        num_iterations: int = 100
    ) -> Dict:
        """
        Quick optimization with default parameters.
        
        Args:
            num_base_stations: Number of aerial base stations
            num_mobile_users: Number of mobile users
            region_size: Size of the square region (km)
            num_iterations: Number of optimization iterations
            
        Returns:
            Dictionary with optimization results
        """
        optimizer = cls()
        return optimizer.optimize(
            num_base_stations=num_base_stations,
            num_mobile_users=num_mobile_users,
            region_size=region_size,
            num_iterations=num_iterations
        )


# Convenience function for external API
def optimize_uav_placement(**kwargs) -> Dict:
    """
    Convenience function for UAV placement optimization.
    
    Args:
        **kwargs: Parameters passed to UAVPlacementOptimizer.optimize()
        
    Returns:
        Dictionary with optimization results
    """
    optimizer = UAVPlacementOptimizer()
    return optimizer.optimize(**kwargs)