"""Tests for the UAV placement optimization package."""

import numpy as np
import pytest

from uav_placement.core.mobile_user import MobileUser
from uav_placement.core.aerial_base_station import StochasticAerialBaseStation, KmeansAerialBaseStation  
from uav_placement.core.channel import Channel
from uav_placement.core.placement_simulator import PlacementSimulator
from uav_placement.optimization.optimizer import UAVPlacementOptimizer
from uav_placement.utils import dbm_to_watt, watt_to_dbm, sigmoid


class TestUtils:
    """Test utility functions."""
    
    def test_power_conversions(self):
        """Test dBm to Watt conversions."""
        # Test known conversions
        assert np.isclose(dbm_to_watt(30), 1.0)  # 30 dBm = 1 W
        assert np.isclose(dbm_to_watt(0), 0.001)  # 0 dBm = 1 mW
        assert np.isclose(watt_to_dbm(1.0), 30.0)  # 1 W = 30 dBm
        assert np.isclose(watt_to_dbm(0.001), 0.0)  # 1 mW = 0 dBm
    
    def test_sigmoid(self):
        """Test sigmoid function."""
        assert np.isclose(sigmoid(0), 0.5)
        assert np.isclose(sigmoid(-1000), 0.0, atol=1e-10)
        assert np.isclose(sigmoid(1000), 1.0, atol=1e-10)


class TestMobileUser:
    """Test MobileUser class."""
    
    def test_initialization(self):
        """Test mobile user initialization."""
        mu = MobileUser()
        assert np.allclose(mu.position, [0, 0, 0])
        assert mu.utility_type == 'sigmoid'
        assert mu.aggregate_type == 'maxPower'
    
    def test_sigmoid_utility(self):
        """Test sigmoid utility function."""
        mu = MobileUser(
            utility_type='sigmoid',
            power_start=1.0,
            power_stop=2.0
        )
        
        # Test utility values
        assert mu.utility([0.5]) < 0.5  # Below threshold
        assert np.isclose(mu.utility([1.5]), 0.5, atol=0.1)  # Middle
        assert mu.utility([2.5]) > 0.5  # Above threshold
    
    def test_leaky_relu_utility(self):
        """Test leaky ReLU utility function."""
        mu = MobileUser(
            utility_type='leakyRelu',
            power_start=1.0,
            positive_scaling=0.1
        )
        
        # Test utility values
        assert np.isclose(mu.utility([0.5]), -0.5)  # Below threshold: linear
        assert np.isclose(mu.utility([1.5]), 0.05)  # Above threshold: scaled
    
    def test_power_aggregation(self):
        """Test power aggregation methods."""
        mu_total = MobileUser(aggregate_type='totalPower')
        mu_max = MobileUser(aggregate_type='maxPower')
        
        powers = np.array([1.0, 2.0, 3.0])
        
        assert np.isclose(mu_total.aggregate_power(powers), 6.0)
        assert np.isclose(mu_max.aggregate_power(powers), 3.0)
    
    def test_clone_at_random_positions(self):
        """Test cloning mobile users at random positions."""
        template = MobileUser(position=np.array([0, 0, 1]))
        users = MobileUser.clone_at_random_positions(template, 5, 10.0)
        
        assert len(users) == 5
        for user in users:
            assert 0 <= user.position[0] <= 10
            assert 0 <= user.position[1] <= 10
            assert user.position[2] == 1  # Same height


class TestChannel:
    """Test Channel class."""
    
    def test_free_space_gain(self):
        """Test free space channel gain."""
        channel = Channel(model='freeSpace', gain_constant=1.0)
        
        bs_pos = np.array([0, 0, 1])
        mu_pos = np.array([1, 0, 0])
        
        gain = channel.gain(bs_pos, mu_pos)
        expected_gain = 1.0 / (1**2 + 1**2)  # Distance squared = 2
        assert np.isclose(gain, expected_gain)
    
    def test_gain_gradient(self):
        """Test channel gain gradient."""
        channel = Channel(model='freeSpace', gain_constant=1.0)
        
        bs_pos = np.array([0, 0, 1])
        mu_pos = np.array([1, 0, 0])
        
        gradient = channel.gain_gradient(bs_pos, mu_pos)
        assert len(gradient) == 3
        # Gradient should point from MU toward BS
        assert gradient[0] < 0  # BS should move toward MU in x direction


class TestAerialBaseStation:
    """Test AerialBaseStation classes."""
    
    def test_stochastic_bs_initialization(self):
        """Test stochastic base station initialization."""
        bs = StochasticAerialBaseStation(
            position=np.array([1, 2, 3]),
            power=5.0,
            step_size=0.1
        )
        
        assert np.allclose(bs.position, [1, 2, 3])
        assert bs.power == 5.0
        assert bs.step_size == 0.1
    
    def test_kmeans_bs_initialization(self):
        """Test k-means base station initialization."""
        bs = KmeansAerialBaseStation(
            position=np.array([1, 2, 3]),
            power=5.0
        )
        
        assert np.allclose(bs.position, [1, 2, 3])
        assert bs.power == 5.0
    
    def test_clone_at_random_positions(self):
        """Test cloning base stations at random positions."""
        template = StochasticAerialBaseStation(position=np.array([0, 0, 1]))
        powers = [1.0, 2.0, 3.0]
        base_stations = StochasticAerialBaseStation.clone_at_random_positions(
            template, powers, 10.0
        )
        
        assert len(base_stations) == 3
        for i, bs in enumerate(base_stations):
            assert bs.power == powers[i]
            assert 0 <= bs.position[0] <= 10
            assert 0 <= bs.position[1] <= 10
            assert bs.position[2] == 1  # Same height


class TestPlacementSimulator:
    """Test PlacementSimulator class."""
    
    def test_simulator_initialization(self):
        """Test simulator initialization."""
        sim = PlacementSimulator(num_steps=10, mini_batch_size=5)
        assert sim.num_steps == 10
        assert sim.mini_batch_size == 5
    
    def test_simple_simulation(self):
        """Test a simple simulation run."""
        # Create a minimal simulation setup
        mu = MobileUser(utility_type='sigmoid', power_start=1.0, power_stop=2.0)
        mobile_users = [mu.clone() for _ in range(5)]
        
        bs_template = StochasticAerialBaseStation(step_size=0.1)
        aerial_base_stations = StochasticAerialBaseStation.clone_at_random_positions(
            bs_template, [1.0, 2.0], 5.0
        )
        
        channel = Channel(model='freeSpace', gain_constant=1.0)
        
        sim = PlacementSimulator(
            num_steps=5,
            mini_batch_size=2,
            mobile_users=mobile_users,
            aerial_base_stations=aerial_base_stations,
            channel=channel
        )
        
        position_history, metrics = sim.generate_sample_path()
        
        # Check outputs
        assert position_history.shape == (3, 2, 5)  # 3D, 2 BSs, 5 steps
        assert 'final_utility' in metrics
        assert 'utility_history' in metrics
        assert len(metrics['utility_history']) == 5


class TestOptimizer:
    """Test UAVPlacementOptimizer class."""
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        optimizer = UAVPlacementOptimizer()
        assert optimizer.results is None
        assert optimizer.simulator is None
    
    def test_quick_optimize(self):
        """Test quick optimization with minimal parameters."""
        results = UAVPlacementOptimizer.quick_optimize(
            num_base_stations=2,
            num_mobile_users=10,
            region_size=5.0,
            num_iterations=5
        )
        
        # Check result structure
        assert 'optimization_method' in results
        assert 'parameters' in results
        assert 'results' in results
        assert 'convergence' in results
        
        # Check that we have the right number of base stations
        bs_positions = results['results']['final_bs_positions']
        assert len(bs_positions) == 2
        
        # Check that positions are within region
        for bs_id, pos in bs_positions.items():
            assert 0 <= pos['x_km'] <= 5.0
            assert 0 <= pos['y_km'] <= 5.0
    
    def test_optimization_methods(self):
        """Test different optimization methods."""
        # Test stochastic method
        optimizer1 = UAVPlacementOptimizer()
        results1 = optimizer1.optimize(
            num_base_stations=2,
            num_mobile_users=5,
            optimization_method='stochastic',
            num_iterations=3
        )
        assert results1['optimization_method'] == 'stochastic'
        
        # Test k-means method
        optimizer2 = UAVPlacementOptimizer()
        results2 = optimizer2.optimize(
            num_base_stations=2,
            num_mobile_users=5,
            optimization_method='kmeans',
            num_iterations=3
        )
        assert results2['optimization_method'] == 'kmeans'


if __name__ == '__main__':
    pytest.main([__file__])