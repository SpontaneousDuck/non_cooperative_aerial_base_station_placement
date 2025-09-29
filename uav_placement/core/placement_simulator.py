"""Placement simulator for UAV optimization."""

import numpy as np
from typing import List, Tuple, Optional, Union
import matplotlib.pyplot as plt

from .aerial_base_station import AerialBaseStation
from .mobile_user import MobileUser
from .channel import Channel


class PlacementSimulator:
    """
    Main simulator for aerial base station placement optimization.
    
    Orchestrates the optimization process by:
    1. Selecting mini-batches of mobile users
    2. Computing utility gradients
    3. Updating base station positions
    4. Tracking convergence and generating visualizations
    """
    
    def __init__(
        self,
        num_steps: int = 20,
        mini_batch_size: Optional[int] = None,
        mobile_users: List[MobileUser] = None,
        aerial_base_stations: List[AerialBaseStation] = None,
        channel: Channel = None,
        debug: bool = False
    ):
        """
        Initialize placement simulator.
        
        Args:
            num_steps: Number of optimization steps
            mini_batch_size: Size of mini-batch for stochastic updates (None = full batch)
            mobile_users: List of mobile users
            aerial_base_stations: List of aerial base stations
            channel: Channel model
            debug: Whether to enable debug output
        """
        self.num_steps = num_steps
        self.mini_batch_size = mini_batch_size
        self.mobile_users = mobile_users or []
        self.aerial_base_stations = aerial_base_stations or []
        self.channel = channel or Channel()
        self.debug = debug
        
        # Tracking variables
        self.position_history = []
        self.utility_history = []
        self.gradient_history = []
    
    def generate_sample_path(self) -> Tuple[np.ndarray, dict]:
        """
        Run the optimization simulation.
        
        Returns:
            position_history: Array of shape (3, num_bs, num_steps) with BS positions
            metrics: Dictionary with convergence metrics and final results
        """
        num_bs = len(self.aerial_base_stations)
        num_mu = len(self.mobile_users)
        
        if num_bs == 0 or num_mu == 0:
            raise ValueError("Need at least one base station and one mobile user")
        
        # Initialize tracking
        position_history = np.zeros((3, num_bs, self.num_steps))
        utility_history = []
        gradient_norms = []
        
        print(f"Starting optimization with {num_bs} BSs and {num_mu} MUs for {self.num_steps} steps")
        
        # Generate mini-batch indices for each step
        batch_indices = self._generate_batch_indices(num_mu)
        
        for step in range(self.num_steps):
            # Record current positions
            for i, bs in enumerate(self.aerial_base_stations):
                position_history[:, i, step] = bs.position
            
            # Select mini-batch for this step
            selected_mu_indices = batch_indices[step]
            selected_mus = [self.mobile_users[i] for i in selected_mu_indices]
            mu_positions = np.array([mu.position for mu in selected_mus]).T  # 3 x num_selected
            
            # Compute utility gradients from selected MUs
            bs_powers = np.array([bs.power for bs in self.aerial_base_stations])
            bs_positions = np.array([bs.position for bs in self.aerial_base_stations]).T  # 3 x num_bs
            
            utility_gradients = self._compute_utility_gradients(
                bs_positions, bs_powers, mu_positions, selected_mus
            )
            
            # Compute average utility for monitoring
            avg_utility = self._compute_average_utility(bs_positions, bs_powers)
            utility_history.append(float(avg_utility))
            
            # Compute gradient norm for convergence monitoring
            gradient_norm = np.mean([np.linalg.norm(grad) for grad in utility_gradients])
            gradient_norms.append(float(gradient_norm))
            
            if self.debug or step % max(1, self.num_steps // 10) == 0:
                print(f"Step {step}: Avg utility = {float(avg_utility):.4f}, Gradient norm = {float(gradient_norm):.4f}")
            
            # Update base station positions
            for i, bs in enumerate(self.aerial_base_stations):
                bs.update_position(self.channel, utility_gradients[i], mu_positions)
        
        # Store final results
        self.position_history = position_history
        self.utility_history = utility_history
        
        metrics = {
            'final_utility': utility_history[-1] if utility_history else 0,
            'utility_history': utility_history,
            'gradient_norms': gradient_norms,
            'converged': gradient_norms[-1] < 1e-6 if gradient_norms else False,
            'final_positions': position_history[:, :, -1]
        }
        
        return position_history, metrics
    
    def _generate_batch_indices(self, num_mu: int) -> List[List[int]]:
        """Generate mini-batch indices for each optimization step."""
        batch_indices = []
        
        for _ in range(self.num_steps):
            if self.mini_batch_size is None or self.mini_batch_size >= num_mu:
                # Full batch
                indices = list(range(num_mu))
            else:
                # Random mini-batch
                indices = np.random.choice(num_mu, self.mini_batch_size, replace=False).tolist()
            batch_indices.append(indices)
        
        return batch_indices
    
    def _compute_utility_gradients(
        self, 
        bs_positions: np.ndarray, 
        bs_powers: np.ndarray,
        mu_positions: np.ndarray,
        selected_mus: List[MobileUser]
    ) -> List[np.ndarray]:
        """
        Compute utility gradients for each base station.
        
        Args:
            bs_positions: Base station positions (3, num_bs)
            bs_powers: Base station powers (num_bs,)
            mu_positions: Selected MU positions (3, num_selected)
            selected_mus: List of selected mobile users
            
        Returns:
            List of gradient arrays, one per base station
        """
        num_bs = bs_positions.shape[1]
        num_selected = len(selected_mus)
        
        # Compute all channel gains: (num_bs, num_selected)
        gains = np.zeros((num_bs, num_selected))
        for i in range(num_bs):
            for j in range(num_selected):
                gains[i, j] = self.channel.gain(bs_positions[:, i], mu_positions[:, j])
        
        # Compute received powers for each MU: (num_bs, num_selected)
        rx_powers = bs_powers[:, np.newaxis] * gains
        
        # Compute utility gradients from each MU w.r.t. received power from each BS
        utility_gradients = []
        for i in range(num_bs):
            bs_utility_gradients = np.zeros(num_selected)
            for j, mu in enumerate(selected_mus):
                # Get all received powers for this MU
                mu_rx_powers = rx_powers[:, j]
                # Compute gradient w.r.t. power from BS i
                bs_utility_gradients[j] = mu.utility_gradient(mu_rx_powers)[i]
            utility_gradients.append(bs_utility_gradients)
        
        return utility_gradients
    
    def _compute_average_utility(self, bs_positions: np.ndarray, bs_powers: np.ndarray) -> float:
        """Compute average utility across all mobile users."""
        total_utility = 0.0
        
        for mu in self.mobile_users:
            # Compute received powers from all BSs
            rx_powers = np.zeros(len(self.aerial_base_stations))
            for i, bs in enumerate(self.aerial_base_stations):
                gain = self.channel.gain(bs_positions[:, i], mu.position)
                rx_powers[i] = bs_powers[i] * gain
            
            total_utility += mu.utility(rx_powers)
        
        return total_utility / len(self.mobile_users)
    
    def plot_results(self, region_size: float = 10.0, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot optimization results.
        
        Args:
            region_size: Size of the region for plotting
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure object
        """
        if not self.position_history:
            raise ValueError("No simulation results to plot. Run generate_sample_path() first.")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Final positions
        ax = axes[0, 0]
        mu_positions = np.array([mu.position for mu in self.mobile_users])
        final_bs_positions = self.position_history[:, :, -1]
        
        ax.scatter(mu_positions[:, 0], mu_positions[:, 1], c='blue', marker='o', label='Mobile Users', alpha=0.6)
        ax.scatter(final_bs_positions[0, :], final_bs_positions[1, :], c='red', marker='^', s=100, label='Base Stations')
        ax.set_xlim(0, region_size)
        ax.set_ylim(0, region_size)
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_title('Final Positions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Utility convergence
        ax = axes[0, 1]
        if self.utility_history:
            ax.plot(self.utility_history)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Average Utility')
            ax.set_title('Utility Convergence')
            ax.grid(True, alpha=0.3)
        
        # Plot 3: BS trajectories
        ax = axes[1, 0]
        for i in range(self.position_history.shape[1]):
            trajectory = self.position_history[:2, i, :]  # x, y coordinates
            ax.plot(trajectory[0, :], trajectory[1, :], '-', alpha=0.7, label=f'BS {i+1}')
            ax.scatter(trajectory[0, 0], trajectory[1, 0], marker='o', s=50)  # Start
            ax.scatter(trajectory[0, -1], trajectory[1, -1], marker='s', s=50)  # End
        
        ax.set_xlim(0, region_size)
        ax.set_ylim(0, region_size)
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_title('BS Trajectories')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Position evolution over time
        ax = axes[1, 1]
        if self.position_history.shape[2] > 1:
            for i in range(self.position_history.shape[1]):
                x_positions = self.position_history[0, i, :]
                ax.plot(x_positions, label=f'BS {i+1} X')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('X Position (km)')
        ax.set_title('X-Position Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def get_final_metrics(self) -> dict:
        """Get final optimization metrics."""
        if not self.utility_history:
            return {}
        
        return {
            'final_utility': self.utility_history[-1],
            'initial_utility': self.utility_history[0],
            'utility_improvement': self.utility_history[-1] - self.utility_history[0],
            'num_iterations': len(self.utility_history),
            'final_bs_positions': self.position_history[:, :, -1].tolist() if len(self.position_history) > 0 else []
        }