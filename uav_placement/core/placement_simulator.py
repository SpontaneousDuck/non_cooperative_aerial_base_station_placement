"""Placement simulator for UAV optimization."""

import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

from .aerial_base_station import AerialBaseStation, KmeansAerialBaseStation
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
        gradient_norms = []  # store step norms in km for interpretability
        
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
            
            # Check if k-means mode (utility_gradients contains indices, not gradient values)
            is_kmeans = isinstance(self.aerial_base_stations[0], KmeansAerialBaseStation)
            
            # Save old positions to compute movement for k-means
            if is_kmeans:
                old_positions = [bs.position.copy() for bs in self.aerial_base_stations]
            
            # Update base station positions
            for i, bs in enumerate(self.aerial_base_stations):
                bs.update_position(self.channel, utility_gradients[i], mu_positions)
            
            # Compute position step norms (km) for convergence monitoring
            if is_kmeans:
                # For k-means, compute actual movement distance
                step_norms_km = []
                for i, bs in enumerate(self.aerial_base_stations):
                    movement = bs.position - old_positions[i]
                    step_norm = np.linalg.norm(movement[:2])  # Only horizontal (x,y)
                    step_norms_km.append(step_norm)
                gradient_norm = float(np.mean(step_norms_km)) if step_norms_km else 0.0
            else:
                # For gradient-based methods, compute step size * gradient norm
                step_norms_km = []
                for i, bs in enumerate(self.aerial_base_stations):
                    pos_grad = bs.compute_position_gradient(self.channel, utility_gradients[i], mu_positions)
                    step_vec = bs.step_size * pos_grad
                    step_norm = np.linalg.norm(step_vec[:2])  # Only horizontal (x,y)
                    step_norms_km.append(step_norm)
                gradient_norm = float(np.mean(step_norms_km)) if step_norms_km else 0.0
            
            gradient_norms.append(gradient_norm)
            
            if self.debug or step % max(1, self.num_steps // 10) == 0:
                print(f"Step {step}: Avg utility = {float(avg_utility):.4f}, Step norm = {float(gradient_norm):.4e} km")
        
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
    
    def _compute_kmeans_assignments(
        self,
        bs_positions: np.ndarray,
        bs_powers: np.ndarray,
        mu_positions: np.ndarray
    ) -> List[np.ndarray]:
        """
        Compute k-means assignments: which MUs are assigned to each BS.
        
        Args:
            bs_positions: Base station positions (3, num_bs)
            bs_powers: Base station powers (num_bs,)
            mu_positions: Selected MU positions (3, num_selected)
            
        Returns:
            List of MU index arrays, one per base station
        """
        num_bs = bs_positions.shape[1]
        num_selected = mu_positions.shape[1]
        
        # Compute all channel gains: (num_bs, num_selected)
        gains = np.zeros((num_bs, num_selected))
        for i in range(num_bs):
            for j in range(num_selected):
                gains[i, j] = self.channel.gain(bs_positions[:, i], mu_positions[:, j])
        
        # Compute received powers for each MU: (num_bs, num_selected)
        rx_powers = bs_powers[:, np.newaxis] * gains
        
        # Assign each MU to the BS with strongest signal
        best_bs_per_mu = np.argmax(rx_powers, axis=0)  # (num_selected,)
        
        # Create list of assigned MU indices for each BS
        assignments = []
        for i in range(num_bs):
            assigned_mu_indices = np.where(best_bs_per_mu == i)[0]
            assignments.append(assigned_mu_indices)
        
        return assignments

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
            List of gradient arrays, one per base station (or MU index arrays for k-means)
        """
        num_bs = bs_positions.shape[1]
        num_selected = len(selected_mus)
        
        # Check if we're in k-means mode
        is_kmeans = selected_mus[0].utility_type == 'kmeans' if selected_mus else False
        
        if is_kmeans:
            # For k-means, return MU assignments instead of gradients
            return self._compute_kmeans_assignments(bs_positions, bs_powers, mu_positions)
        
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
        """Compute average utility across all mobile users.
        
        For k-means, we compute utility using sigmoid (for comparison purposes),
        even though the optimization uses assignment-based updates.
        """
        total_utility = 0.0
        
        # Check if we're in k-means mode
        is_kmeans = isinstance(self.aerial_base_stations[0], KmeansAerialBaseStation)
        
        for mu in self.mobile_users:
            # Compute received powers from all BSs
            rx_powers = np.zeros(len(self.aerial_base_stations))
            for i, bs in enumerate(self.aerial_base_stations):
                gain = self.channel.gain(bs_positions[:, i], mu.position)
                rx_powers[i] = bs_powers[i] * gain
            
            # For k-means, temporarily use sigmoid utility for monitoring
            if is_kmeans:
                original_type = mu.utility_type
                mu.utility_type = 'sigmoid'
                total_utility += mu.utility(rx_powers)
                mu.utility_type = original_type
            else:
                total_utility += mu.utility(rx_powers)
        
        return total_utility / len(self.mobile_users)
    
    def _compute_power_grid(self, region_size: float, num_points: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute max power received across a grid of test points.
        
        Args:
            region_size: Size of the region (km)
            num_points: Number of grid points per dimension
            
        Returns:
            X, Y, Z meshgrids where Z is the max received power at each point
        """
        # Create grid of test points
        x = np.linspace(0, region_size, num_points)
        y = np.linspace(0, region_size, num_points)
        X, Y = np.meshgrid(x, y)
        
        # Get final BS positions and powers
        final_bs_positions = self.position_history[:, :, -1]  # 3 x num_bs
        bs_powers = np.array([bs.power for bs in self.aerial_base_stations])
        
        # Compute max received power at each grid point
        Z = np.zeros_like(X)
        for i in range(num_points):
            for j in range(num_points):
                test_point = np.array([X[i, j], Y[i, j], 0.0])
                
                # Compute received power from each BS
                rx_powers = np.zeros(len(self.aerial_base_stations))
                for k, bs in enumerate(self.aerial_base_stations):
                    gain = self.channel.gain(final_bs_positions[:, k], test_point)
                    rx_powers[k] = bs_powers[k] * gain
                
                # Store max received power (in dBm)
                max_power_watt = np.max(rx_powers)
                Z[i, j] = 10 * np.log10(max_power_watt * 1000)  # Convert to dBm
        
        return X, Y, Z
    
    def plot_stochastic(self, region_size: float = 10.0, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot stochastic optimization results with trajectories and power heatmap.
        
        Args:
            region_size: Size of the region for plotting
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure object
        """
        if len(self.position_history) == 0:
            raise ValueError("No simulation results to plot. Run generate_sample_path() first.")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        mu_positions = np.array([mu.position for mu in self.mobile_users])
        final_bs_positions = self.position_history[:, :, -1]
        
        # Plot 1: Final positions with power heatmap
        ax = axes[0, 0]
        
        # Compute and plot power heatmap
        print("Computing power distribution heatmap...")
        X, Y, Z = self._compute_power_grid(region_size, num_points=50)
        
        # Get power threshold for clipping
        if len(self.mobile_users) > 0:
            power_threshold = self.mobile_users[0].power_start
            power_threshold_dbm = 10 * np.log10(power_threshold * 1000)
            power_upper = self.mobile_users[0].power_stop
            power_upper_dbm = 10 * np.log10(power_upper * 1000)
        else:
            power_threshold_dbm = -91.0
            power_upper_dbm = -89.0
        
        # Clip the power values for better visualization
        Z_clipped = np.clip(Z, power_threshold_dbm, power_upper_dbm)
        
        # Create heatmap
        im = ax.imshow(Z_clipped, extent=[0, region_size, 0, region_size], 
                      origin='lower', cmap='viridis', aspect='auto',
                      vmin=power_threshold_dbm, vmax=power_upper_dbm)
        
        # Plot BS trajectories
        for i in range(self.position_history.shape[1]):
            trajectory = self.position_history[:2, i, :]
            ax.plot(trajectory[0, :], trajectory[1, :], '-', color='green', alpha=0.7, linewidth=2)
        
        # Plot mobile users as white dots
        ax.scatter(mu_positions[:, 0], mu_positions[:, 1], c='white', marker='o', 
                  s=20, alpha=0.8, edgecolors='black', linewidths=1)
        
        # Plot final BS positions
        ax.scatter(final_bs_positions[0, :], final_bs_positions[1, :], 
                  c='red', marker='X', s=100, edgecolors='black', linewidths=1)
        
        ax.set_xlim(0, region_size)
        ax.set_ylim(0, region_size)
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_title('Clipped Max Power')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Max Received Power (dBm)')
        
        ax.grid(False)  # Turn off grid for heatmap
        
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
            ax.scatter(trajectory[0, -1], trajectory[1, -1], marker='X', s=100, c='red', linewidths=1)  # End
        
        ax.set_xlim(0, region_size)
        ax.set_ylim(0, region_size)
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_title('BS Trajectories')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Received power distribution across all MUs
        ax = axes[1, 1]
        
        # Compute received power for each MU from the best serving BS
        mu_positions = np.array([mu.position for mu in self.mobile_users])
        bs_powers = np.array([bs.power for bs in self.aerial_base_stations])
        num_bs = len(self.aerial_base_stations)
        
        received_powers_dbm = []
        for i, mu in enumerate(self.mobile_users):
            # Find maximum received power from all BSs
            max_rx_power = 0
            for j in range(num_bs):
                gain = self.channel.gain(final_bs_positions[:, j], mu.position)
                rx_power = bs_powers[j] * gain
                max_rx_power = max(max_rx_power, rx_power)
            rx_power_dbm = 10 * np.log10(max_rx_power * 1000)  # Convert to dBm
            received_powers_dbm.append(rx_power_dbm)
        
        # Create histogram
        n, bins, patches = ax.hist(received_powers_dbm, bins=30, color='steelblue', 
                                   alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add vertical lines for key statistics
        mean_power = np.mean(received_powers_dbm)
        median_power = np.median(received_powers_dbm)
        ax.axvline(mean_power, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_power:.2f} dBm')
        ax.axvline(median_power, color='orange', linestyle='--', linewidth=2,
                  label=f'Median: {median_power:.2f} dBm')
        
        # Add power threshold if available
        if len(self.mobile_users) > 0:
            power_threshold = self.mobile_users[0].power_start
            power_threshold_dbm = 10 * np.log10(power_threshold * 1000)
            ax.axvline(power_threshold_dbm, color='green', linestyle=':', 
                      linewidth=2, label=f'Min Threshold: {power_threshold_dbm:.2f} dBm')
        
        ax.set_xlabel('Received Power (dBm)')
        ax.set_ylabel('Number of Mobile Users')
        ax.set_title('Distribution of Received Power')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_kmeans(self, region_size: float = 10.0, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot k-means optimization results with cluster assignments and power heatmap.
        
        Args:
            region_size: Size of the region for plotting
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure object
        """
        if len(self.position_history) == 0:
            raise ValueError("No simulation results to plot. Run generate_sample_path() first.")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        mu_positions = np.array([mu.position for mu in self.mobile_users])
        final_bs_positions = self.position_history[:, :, -1]
        bs_powers = np.array([bs.power for bs in self.aerial_base_stations])
        
        # Compute MU assignments to BSs
        num_mu = len(self.mobile_users)
        num_bs = len(self.aerial_base_stations)
        
        # Compute which BS each MU is assigned to (strongest signal)
        mu_assignments = np.zeros(num_mu, dtype=int)
        for i, mu in enumerate(self.mobile_users):
            rx_powers = np.zeros(num_bs)
            for j in range(num_bs):
                gain = self.channel.gain(final_bs_positions[:, j], mu.position)
                rx_powers[j] = bs_powers[j] * gain
            mu_assignments[i] = np.argmax(rx_powers)
        
        # Plot 1: Final positions with power heatmap
        ax = axes[0, 0]
        
        # Compute and plot power heatmap
        print("Computing power distribution heatmap...")
        X, Y, Z = self._compute_power_grid(region_size, num_points=50)
        
        # Get power threshold for clipping
        if len(self.mobile_users) > 0:
            power_threshold = self.mobile_users[0].power_start
            power_threshold_dbm = 10 * np.log10(power_threshold * 1000)
            power_upper = self.mobile_users[0].power_stop
            power_upper_dbm = 10 * np.log10(power_upper * 1000)
        else:
            power_threshold_dbm = -91.0
            power_upper_dbm = -89.0
        
        # Clip the power values for better visualization
        Z_clipped = np.clip(Z, power_threshold_dbm, power_upper_dbm)
        
        # Create heatmap
        im = ax.imshow(Z_clipped, extent=[0, region_size, 0, region_size], 
                      origin='lower', cmap='viridis', aspect='auto',
                      vmin=power_threshold_dbm, vmax=power_upper_dbm)
        
        # Plot mobile users colored by assignment
        colors = plt.cm.tab10(np.linspace(0, 1, num_bs))
        for bs_idx in range(num_bs):
            assigned_mus = mu_positions[mu_assignments == bs_idx]
            if len(assigned_mus) > 0:
                ax.scatter(assigned_mus[:, 0], assigned_mus[:, 1], 
                          c=[colors[bs_idx]], marker='o', s=30, alpha=0.7,
                          edgecolors='black', linewidths=1,
                          label=f'BS {bs_idx+1} cluster')
        
        # Plot final BS positions
        for bs_idx in range(num_bs):
            ax.scatter(final_bs_positions[0, bs_idx], final_bs_positions[1, bs_idx], 
                      c=[colors[bs_idx]], marker='X', s=150, edgecolors='black', linewidths=1)
        
        ax.set_xlim(0, region_size)
        ax.set_ylim(0, region_size)
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_title('Clipped Max Power with K-Means Clusters')
        ax.legend(loc='upper right', fontsize=8)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Max Received Power (dBm)')
        ax.grid(False)
        
        # Plot 2: Utility convergence
        ax = axes[0, 1]
        if self.utility_history:
            ax.plot(self.utility_history, linewidth=2)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Average Utility')
            ax.set_title('Utility Convergence (K-Means)')
            ax.grid(True, alpha=0.3)
        
        # Plot 3: Cluster sizes
        ax = axes[1, 0]
        cluster_sizes = [np.sum(mu_assignments == i) for i in range(num_bs)]
        bars = ax.bar(range(num_bs), cluster_sizes, color=colors)
        ax.set_xlabel('Base Station')
        ax.set_ylabel('Number of Assigned MUs')
        ax.set_title('Cluster Sizes')
        ax.set_xticks(range(num_bs))
        ax.set_xticklabels([f'BS {i+1}' for i in range(num_bs)])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add count labels on bars
        for i, (bar, size) in enumerate(zip(bars, cluster_sizes)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(size)}', ha='center', va='bottom')
        
        # Plot 4: Received power distribution per cluster
        ax = axes[1, 1]
        
        # Compute received power for each MU from its assigned BS
        received_powers_dbm = []
        cluster_labels = []
        for bs_idx in range(num_bs):
            assigned_mu_indices = np.where(mu_assignments == bs_idx)[0]
            for mu_idx in assigned_mu_indices:
                mu = self.mobile_users[mu_idx]
                gain = self.channel.gain(final_bs_positions[:, bs_idx], mu.position)
                rx_power = bs_powers[bs_idx] * gain
                rx_power_dbm = 10 * np.log10(rx_power * 1000)  # Convert to dBm
                received_powers_dbm.append(rx_power_dbm)
                cluster_labels.append(bs_idx)
        
        # Create box plot data grouped by cluster
        box_data = [[] for _ in range(num_bs)]
        for power, label in zip(received_powers_dbm, cluster_labels):
            box_data[label].append(power)
        
        # Create box plot
        bp = ax.boxplot(box_data, labels=[f'BS {i+1}' for i in range(num_bs)],
                       patch_artist=True, showmeans=True,
                       meanprops=dict(marker='D', markerfacecolor='green', markeredgecolor='green', markersize=6),
                       medianprops=dict(color='darkred', linewidth=2.5))
        
        # Color boxes to match clusters
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax.set_xlabel('Base Station')
        ax.set_ylabel('Received Power (dBm)')
        ax.set_title('Received Power Distribution per Cluster')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Create legend elements
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='darkred', linewidth=2.5, label='Median'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='green', 
                   markeredgecolor='green', markersize=6, label='Mean'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='white', 
                   markeredgecolor='black', markersize=5, label='Outliers')
        ]
        
        # Add horizontal line for power threshold if available
        if len(self.mobile_users) > 0:
            power_threshold = self.mobile_users[0].power_start
            power_threshold_dbm = 10 * np.log10(power_threshold * 1000)
            ax.axhline(y=power_threshold_dbm, color='r', linestyle='--', 
                      linewidth=1, alpha=0.5, label='Min Threshold')
            legend_elements.append(Line2D([0], [0], color='r', linestyle='--', 
                                         linewidth=1, alpha=0.5, label='Min Threshold'))
        
        ax.legend(handles=legend_elements, fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_placement(self, region_size: float = 10.0, save_path: Optional[str] = None, optimization_method: Optional[str] = None) -> plt.Figure:
        """
        Plot optimization results, automatically detecting the optimization method.
        
        This unified plotting function works for both stochastic and k-means optimization.
        
        Args:
            region_size: Size of the region for plotting
            save_path: Path to save the plot (optional)
            optimization_method: Optimization method used ('stochastic' or 'kmeans'). If None, auto-detect from BS type.
            
        Returns:
            Matplotlib figure object
        """
        if len(self.position_history) == 0:
            raise ValueError("No simulation results to plot. Run generate_sample_path() first.")
        
        # Determine optimization type
        if optimization_method is None:
            # Auto-detect from BS type (may be unreliable if simulator was reused)
            if isinstance(self.aerial_base_stations[0], KmeansAerialBaseStation):
                optimization_method = 'kmeans'
            else:
                optimization_method = 'stochastic'
        
        # Call appropriate plotting method
        if optimization_method == 'kmeans':
            return self.plot_kmeans(region_size=region_size, save_path=save_path)
        else:
            return self.plot_stochastic(region_size=region_size, save_path=save_path)
    
    def plot_results(self, region_size: float = 10.0, save_path: Optional[str] = None, optimization_method: Optional[str] = None) -> plt.Figure:
        """
        Plot optimization results (legacy method, use plot_placement instead).
        
        This method is kept for backward compatibility and calls plot_placement.
        
        Args:
            region_size: Size of the region for plotting
            save_path: Path to save the plot (optional)
            optimization_method: Optimization method used ('stochastic' or 'kmeans'). If None, auto-detect from BS type.
            
        Returns:
            Matplotlib figure object
        """
        return self.plot_placement(region_size=region_size, save_path=save_path, optimization_method=optimization_method)
    
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