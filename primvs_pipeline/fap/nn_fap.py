"""
Neural Network False Alarm Probability Module

Provides neural network-based FAP calculation using a pre-trained GRU model.

The neural network takes phase-folded lightcurve features as input and
predicts the probability that the detected period is a false alarm.

Author: Niall Miller (refactored)
Date: 2025-10-21
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Import TensorFlow/Keras
try:
    from tensorflow.keras.models import model_from_json
    from sklearn.neighbors import KNeighborsRegressor
    TENSORFLOW_AVAILABLE = True
except ImportError:
    logger.warning("TensorFlow not available - FAP calculation will not work")
    TENSORFLOW_AVAILABLE = False

# Import phasing utility
from ..utils.phasing import phaser


class NeuralNetworkFAP:
    """
    Neural network-based False Alarm Probability calculator.
    
    Uses a pre-trained 12-layer GRU network to predict FAP from
    phase-folded lightcurve features.
    
    Attributes:
        model: Trained Keras model
        knn: KNN regressor for feature smoothing
        n_points: Number of points for neural network input
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        n_points: int = 200,
        knn_neighbors: int = 10
    ):
        """
        Initialize FAP calculator.
        
        Args:
            model_path: Path to model directory (contains _model.json and _model.h5)
            n_points: Number of points for NN input
            knn_neighbors: Number of neighbors for KNN regressor
        """
        self.n_points = n_points
        self.knn_neighbors = knn_neighbors
        self.model = None
        self.knn = None
        
        if model_path is not None:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> None:
        """
        Load pre-trained model from disk.
        
        Args:
            model_path: Path to model directory
            
        Raises:
            FileNotFoundError: If model files not found
            ImportError: If TensorFlow not available
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for FAP calculation")
        
        model_path = Path(model_path)
        
        json_file = model_path / '_model.json'
        weights_file = model_path / '_model.h5'
        
        if not json_file.exists():
            raise FileNotFoundError(f"Model JSON not found: {json_file}")
        if not weights_file.exists():
            raise FileNotFoundError(f"Model weights not found: {weights_file}")
        
        logger.info(f"Loading FAP model from: {model_path}")
        
        # Load model architecture
        with open(json_file, 'r') as f:
            model_json = f.read()
        
        self.model = model_from_json(model_json)
        
        # Load weights
        self.model.load_weights(str(weights_file))
        
        # Initialize KNN regressor
        self.knn = KNeighborsRegressor(
            n_neighbors=self.knn_neighbors,
            weights='distance'
        )
        
        logger.info("FAP model loaded successfully")
    
    def _adjust_arrays(
        self,
        array1: np.ndarray,
        array2: np.ndarray,
        target_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Adjust arrays to target length by random deletion or zero-padding.
        
        Args:
            array1: First array
            array2: Second array
            target_length: Target length
            
        Returns:
            Tuple of adjusted arrays
        """
        # Random deletion if too long
        while len(array1) > target_length:
            idx = np.random.randint(0, len(array1))
            array1 = np.delete(array1, idx)
            array2 = np.delete(array2, idx)
        
        # Zero-padding if too short
        if len(array1) < target_length:
            padding_length = target_length - len(array1)
            array1 = np.pad(array1, (0, padding_length), 'constant')
            array2 = np.pad(array2, (0, padding_length), 'constant')
        
        return array1, array2
    
    def _running_scatter(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_bins: int
    ) -> np.ndarray:
        """
        Calculate running scatter (interquartile range) in bins.
        
        Args:
            x: X values (phase)
            y: Y values (magnitude)
            n_bins: Number of bins
            
        Returns:
            Array of scatter values
        """
        scatter = []
        x_bins = np.linspace(np.min(x), np.max(x), n_bins)
        
        for i in range(len(x_bins)):
            if i < 1:
                mask = x < x_bins[i + 2] if i + 2 < len(x_bins) else x < x_bins[-1]
            elif i == len(x_bins) - 1:
                mask = x >= x_bins[i - 2]
            else:
                mask = (x >= x_bins[i - 2]) & (x <= x_bins[i])
            
            if np.sum(mask) > 1:
                q75, q25 = np.percentile(y[mask], [75, 25])
                scatter.append(abs(q75 - q25))
            else:
                scatter.append(0)
        
        return np.array(scatter)
    
    def _smooth(self, y: np.ndarray, box_pts: int) -> np.ndarray:
        """
        Smooth array using box filter.
        
        Args:
            y: Array to smooth
            box_pts: Box size
            
        Returns:
            Smoothed array
        """
        box = np.ones(box_pts) / box_pts
        return np.convolve(y, box, mode='same')
    
    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data to [0, 1] range.
        
        Args:
            data: Data to normalize
            
        Returns:
            Normalized data
        """
        data_min = np.min(data)
        data_max = np.max(data)
        
        if data_max == data_min:
            return np.zeros_like(data)
        
        return (data - data_min) / (data_max - data_min)
    
    def _generate_channels(
        self,
        mag: np.ndarray,
        phase: np.ndarray
    ) -> np.ndarray:
        """
        Generate multi-channel features for neural network input.
        
        Args:
            mag: Magnitude array
            phase: Phase array
            
        Returns:
            Feature array of shape (7, n_points)
        """
        # Adjust to target length
        mag, phase = self._adjust_arrays(mag, phase, self.n_points)
        
        # Sort by phase
        sort_idx = np.argsort(phase)
        mag = mag[sort_idx]
        phase = phase[sort_idx]
        
        # Remove NaN values
        mask = ~np.isnan(mag) & ~np.isnan(phase)
        mag = mag[mask]
        phase = phase[mask]
        
        # KNN smoothing
        knn_smooth = self.knn.fit(
            phase[:, np.newaxis],
            mag
        ).predict(
            np.linspace(0, 1, self.n_points)[:, np.newaxis]
        )
        
        # Running scatter
        scatter = self._running_scatter(phase, mag, self.n_points)
        
        # Phase differences
        delta_phase = np.diff(phase, prepend=0)
        
        # Smooth at different scales
        smooth_fine = self._smooth(knn_smooth, int(self.n_points / 20))
        smooth_coarse = self._smooth(knn_smooth, int(self.n_points / 5))
        
        # Stack channels
        channels = np.vstack([
            mag,
            knn_smooth,
            scatter,
            phase,
            smooth_fine,
            smooth_coarse,
            delta_phase
        ])
        
        return channels
    
    def calculate(
        self,
        period: float,
        mag: np.ndarray,
        time: np.ndarray
    ) -> float:
        """
        Calculate FAP for a given period and lightcurve.
        
        Args:
            period: Period to test (days)
            mag: Magnitude array
            time: Time array (MJD)
            
        Returns:
            False alarm probability [0, 1]
            
        Example:
            >>> fap_calc = NeuralNetworkFAP(model_path='models/fap_nn')
            >>> fap = fap_calc.calculate(2.5, mag, time)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Adjust length if needed
        if len(mag) < self.n_points:
            mag = np.pad(mag, (0, self.n_points - len(mag)), 'wrap')
            time = np.pad(time, (0, self.n_points - len(time)), 'wrap')
        elif len(mag) > self.n_points:
            # Random deletion
            n_delete = len(mag) - self.n_points
            delete_idx = np.random.choice(len(mag), n_delete, replace=False)
            mag = np.delete(mag, delete_idx)
            time = np.delete(time, delete_idx)
        
        # Calculate phase
        phase = phaser(time, period)
        
        # Normalize magnitude
        mag = self._normalize(mag)
        
        # Generate features
        features = self._generate_channels(mag, phase)
        
        # Predict FAP
        fap = self.model.predict(np.array([features]), verbose=0)[0][0]
        
        return float(fap)


def calculate_fap(
    period: float,
    mag: np.ndarray,
    time: np.ndarray,
    model_path: str,
    n_points: int = 200
) -> float:
    """
    Convenience function to calculate FAP.
    
    Args:
        period: Period to test (days)
        mag: Magnitude array
        time: Time array (MJD)
        model_path: Path to model directory
        n_points: Number of points for NN input
        
    Returns:
        False alarm probability [0, 1]
        
    Example:
        >>> fap = calculate_fap(2.5, mag, time, 'models/fap_nn')
    """
    calculator = NeuralNetworkFAP(model_path=model_path, n_points=n_points)
    return calculator.calculate(period, mag, time)

