"""
False Alarm Probability (FAP) Module

Provides neural network-based FAP calculation for period detection.

The FAP quantifies the probability that a detected period is a false alarm
due to noise or sampling artifacts rather than a true periodic signal.
"""

from .nn_fap import NeuralNetworkFAP, calculate_fap

__all__ = ["NeuralNetworkFAP", "calculate_fap"]

