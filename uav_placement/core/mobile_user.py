"""Mobile User implementation for UAV placement optimization."""

import numpy as np
from typing import List
from ..utils import sigmoid, sigmoid_derivative, dbm_to_watt, watt_to_dbm, log_sum_exp


class MobileUser:
    """
    Represents a mobile user in the UAV placement optimization.
    
    The mobile user has a utility function that depends on the received power
    from aerial base stations. Different utility and aggregation methods are supported.
    """
    
    def __init__(
        self, 
        position: np.ndarray = None,
        utility_type: str = 'sigmoid',
        aggregate_type: str = 'maxPower',
        power_start: float = 2.0,
        power_stop: float = 3.0,
        positive_scaling: float = 0.2,
        use_dbm: bool = False
    ):
        """
        Initialize a mobile user.
        
        Args:
            position: 3D position [x, y, z] in meters/km. Default: [0, 0, 0]
            utility_type: Type of utility function ('sigmoid', 'leakyRelu', 'kmeans')
            aggregate_type: How to aggregate power from multiple BSs 
                          ('totalPower', 'smoothMaxPower', 'maxPower')
            power_start: Power threshold for utility functions (in Watts)
            power_stop: Upper power threshold for sigmoid (in Watts)  
            positive_scaling: Scaling factor for leaky ReLU positive part
            use_dbm: Whether to operate internally in dBm
        """
        self.position = np.array([0.0, 0.0, 0.0]) if position is None else np.array(position)
        self.utility_type = utility_type
        self.aggregate_type = aggregate_type
        self.power_start = power_start
        self.power_stop = power_stop
        self.positive_scaling = positive_scaling
        self.use_dbm = use_dbm
        
        # Validate inputs
        if self.utility_type not in ['sigmoid', 'leakyRelu', 'kmeans']:
            raise ValueError(f"Invalid utility_type: {self.utility_type}")
        if self.aggregate_type not in ['totalPower', 'smoothMaxPower', 'maxPower']:
            raise ValueError(f"Invalid aggregate_type: {self.aggregate_type}")
    
    def aggregate_power(self, rx_powers: np.ndarray) -> float:
        """
        Aggregate received power from multiple base stations.
        
        Args:
            rx_powers: Array of received powers from different BSs
            
        Returns:
            Aggregated power value
        """
        if len(rx_powers) == 0:
            return 0.0
            
        if self.aggregate_type == 'totalPower':
            return np.sum(rx_powers)
        elif self.aggregate_type == 'maxPower':
            return np.max(rx_powers)
        elif self.aggregate_type == 'smoothMaxPower':
            if self.use_dbm:
                # Convert to dBm for smooth max computation
                rx_powers_dbm = watt_to_dbm(rx_powers)
                return dbm_to_watt(log_sum_exp(rx_powers_dbm))
            else:
                return log_sum_exp(np.log(rx_powers + 1e-12))
        else:
            raise ValueError(f"Unknown aggregate_type: {self.aggregate_type}")
    
    def utility(self, rx_powers: np.ndarray) -> float:
        """
        Compute utility for given received powers.
        
        Args:
            rx_powers: Array of received powers from different BSs
            
        Returns:
            Utility value
        """
        if isinstance(rx_powers, (int, float)):
            rx_powers = np.array([rx_powers])
        
        aggregated_power = self.aggregate_power(rx_powers)
        
        if self.utility_type == 'sigmoid':
            return self._sigmoid_utility(aggregated_power)
        elif self.utility_type == 'leakyRelu':
            return self._leaky_relu_utility(aggregated_power)
        elif self.utility_type == 'kmeans':
            # For kmeans, utility is typically just the distance/power
            return aggregated_power
        else:
            raise ValueError(f"Unknown utility_type: {self.utility_type}")
    
    def utility_gradient(self, rx_powers: np.ndarray) -> np.ndarray:
        """
        Compute gradient of utility w.r.t. received powers.
        
        Args:
            rx_powers: Array of received powers from different BSs
            
        Returns:
            Gradient array with same shape as rx_powers
        """
        if isinstance(rx_powers, (int, float)):
            rx_powers = np.array([rx_powers])
        
        aggregated_power = self.aggregate_power(rx_powers)
        
        # Compute gradient of utility w.r.t. aggregated power
        if self.utility_type == 'sigmoid':
            utility_grad = self._sigmoid_utility_gradient(aggregated_power)
        elif self.utility_type == 'leakyRelu':
            utility_grad = self._leaky_relu_utility_gradient(aggregated_power)
        elif self.utility_type == 'kmeans':
            utility_grad = 1.0  # Linear utility
        else:
            raise ValueError(f"Unknown utility_type: {self.utility_type}")
        
        # Compute gradient of aggregated power w.r.t. individual powers
        if self.aggregate_type == 'totalPower':
            power_grads = np.ones_like(rx_powers)
        elif self.aggregate_type == 'maxPower':
            power_grads = np.zeros_like(rx_powers)
            if len(rx_powers) > 0:
                max_idx = np.argmax(rx_powers)
                power_grads[max_idx] = 1.0
        elif self.aggregate_type == 'smoothMaxPower':
            if self.use_dbm:
                rx_powers_dbm = watt_to_dbm(rx_powers)
                max_dbm = np.max(rx_powers_dbm)
                exp_vals = np.exp(rx_powers_dbm - max_dbm)
                sum_exp = np.sum(exp_vals)
                power_grads = exp_vals / sum_exp
                # Convert gradient from dBm to Watt space: dP_dbm/dP = 10 / (P * ln(10))
                power_grads = power_grads * 10 / rx_powers / np.log(10)
            else:
                max_log = np.max(np.log(rx_powers + 1e-12))
                exp_vals = np.exp(np.log(rx_powers + 1e-12) - max_log)
                sum_exp = np.sum(exp_vals)
                power_grads = exp_vals / sum_exp / (rx_powers + 1e-12)
        else:
            raise ValueError(f"Unknown aggregate_type: {self.aggregate_type}")
        
        return utility_grad * power_grads
    
    def _sigmoid_utility(self, power: float) -> float:
        """Shifted and scaled sigmoid utility function."""
        if self.use_dbm:
            power_dbm = watt_to_dbm(power)
            start_dbm = watt_to_dbm(self.power_start)
            stop_dbm = watt_to_dbm(self.power_stop)
            x = 3 * (2 * power_dbm - start_dbm - stop_dbm) / (stop_dbm - start_dbm)
        else:
            x = 3 * (2 * power - self.power_start - self.power_stop) / (self.power_stop - self.power_start)
        return sigmoid(x)
    
    def _sigmoid_utility_gradient(self, power: float) -> float:
        """Gradient of shifted and scaled sigmoid utility function."""
        if self.use_dbm:
            power_dbm = watt_to_dbm(power)
            start_dbm = watt_to_dbm(self.power_start)
            stop_dbm = watt_to_dbm(self.power_stop)
            x = 3 * (2 * power_dbm - start_dbm - stop_dbm) / (stop_dbm - start_dbm)
            # Chain rule: d/dP = d/dP_dbm * dP_dbm/dP
            dbm_grad = sigmoid_derivative(x) * 6 / (stop_dbm - start_dbm)
            # Correct conversion: dP_dbm/dP = 10 / (P * ln(10))
            return dbm_grad * 10 / power / np.log(10)
        else:
            x = 3 * (2 * power - self.power_start - self.power_stop) / (self.power_stop - self.power_start)
            return sigmoid_derivative(x) * 6 / (self.power_stop - self.power_start)
    
    def _leaky_relu_utility(self, power: float) -> float:
        """Leaky ReLU utility function."""
        diff = power - self.power_start
        if diff <= 0:
            return diff  # Linear part below threshold
        else:
            return self.positive_scaling * diff  # Scaled linear part above threshold
    
    def _leaky_relu_utility_gradient(self, power: float) -> float:
        """Gradient of leaky ReLU utility function."""
        if power <= self.power_start:
            return 1.0
        else:
            return self.positive_scaling
    
    def clone(self) -> 'MobileUser':
        """Create a copy of this mobile user."""
        return MobileUser(
            position=self.position.copy(),
            utility_type=self.utility_type,
            aggregate_type=self.aggregate_type,
            power_start=self.power_start,
            power_stop=self.power_stop,
            positive_scaling=self.positive_scaling,
            use_dbm=self.use_dbm
        )
    
    @classmethod
    def clone_at_random_positions(
        cls, 
        template: 'MobileUser', 
        num_users: int, 
        region_size: float, 
        same_height: bool = True,
        two_d: bool = True
    ) -> List['MobileUser']:
        """
        Create multiple mobile users at random positions.
        
        Args:
            template: Template mobile user to clone
            num_users: Number of users to create
            region_size: Size of the square region for random placement
            same_height: Whether to keep the same height as template
            two_d: If False, all users will have y-coordinate = 0
            
        Returns:
            List of mobile users at random positions
        """
        users = []
        for _ in range(num_users):
            user = template.clone()
            if same_height:
                if two_d:
                    user.position[:2] = np.random.rand(2) * region_size
                else:
                    user.position[0] = np.random.rand() * region_size
                    user.position[1] = 0
            else:
                raise NotImplementedError("Variable height not implemented")
            users.append(user)
        return users
    
    @classmethod
    def clone_at_positions(cls, template: 'MobileUser', positions: np.ndarray) -> List['MobileUser']:
        """
        Create multiple mobile users at specified positions.
        
        Args:
            template: Template mobile user to clone
            positions: Array of shape (3, N) with N positions
            
        Returns:
            List of mobile users at specified positions
        """
        users = []
        for i in range(positions.shape[1]):
            user = template.clone()
            user.position = positions[:, i].copy()
            users.append(user)
        return users
    
    def __repr__(self) -> str:
        return (f"MobileUser(pos={self.position}, utility={self.utility_type}, "
                f"aggregate={self.aggregate_type})")