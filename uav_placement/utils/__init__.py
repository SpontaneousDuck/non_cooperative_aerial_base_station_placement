"""Utility functions for the UAV placement package."""

import numpy as np


def dbm_to_watt(power_dbm):
    """Convert power from dBm to Watts.
    
    Args:
        power_dbm: Power in dBm
        
    Returns:
        Power in Watts
    """
    return 10 ** ((power_dbm - 30) / 10)


def watt_to_dbm(power_watt):
    """Convert power from Watts to dBm.
    
    Args:
        power_watt: Power in Watts
        
    Returns:
        Power in dBm
    """
    return 10 * np.log10(power_watt) + 30


def sigmoid(x):
    """Sigmoid function with numerical stability.
    
    Args:
        x: Input value or array
        
    Returns:
        Sigmoid of x
    """
    return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)), 
                    np.exp(x) / (1 + np.exp(x)))


def sigmoid_derivative(x):
    """Derivative of sigmoid function.
    
    Args:
        x: Input value or array
        
    Returns:
        Derivative of sigmoid at x
    """
    s = sigmoid(x)
    return s * (1 - s)


def log_sum_exp(x, axis=None):
    """Numerically stable log-sum-exp computation.
    
    Args:
        x: Input array
        axis: Axis along which to compute (default: None for all elements)
        
    Returns:
        log(sum(exp(x))) computed in a numerically stable way
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    return x_max + np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))


def vector_to_str(vector, precision=3):
    """Convert vector to formatted string representation.
    
    Args:
        vector: Input vector
        precision: Number of decimal places
        
    Returns:
        String representation of vector
    """
    if np.isscalar(vector):
        return f"{vector:.{precision}f}"
    return "[" + ", ".join([f"{x:.{precision}f}" for x in vector]) + "]"