"""
Deep ensemble for uncertainty quantification over GeothermalPINN.

Trains several independently-initialized PINNs and combines their
predictions into a mean and standard deviation, giving a confidence
interval instead of a single point estimate. Most PINN write-ups only
report a point prediction; a deep ensemble is a cheap way to also
report *how confident* the model is, and -- via the coverage check in
`evaluate()` -- whether that confidence is actually trustworthy.
"""

from typing import Dict, List, Tuple

import numpy as np
import torch

from src.models.pinn import GeothermalPINN, error_metrics


class PINNEnsemble:
    """Wraps a list of independently-trained GeothermalPINN models."""

    def __init__(self, models: List[GeothermalPINN]):
        """
        Args:
            models: Trained GeothermalPINN instances. For the spread in
                their predictions to reflect genuine model uncertainty
                (rather than a confound), they should share the same
                architecture, hyperparameters, and training data, and
                differ only in random seed.
        """
        if not models:
            raise ValueError("PINNEnsemble needs at least one model")
        self.models = models

    def predict(
        self, r: torch.Tensor, t: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict T*/u* mean and standard deviation across the ensemble.

        Args:
            r: Dimensionless radius.
            t: Dimensionless time.

        Returns:
            Tuple (T_mean, T_std, u_mean, u_std), each shape (N, 1).
        """
        T_preds, u_preds = [], []
        for model in self.models:
            T_pred, u_pred = model.predict(r, t)
            T_preds.append(T_pred)
            u_preds.append(u_pred)
        T_stack = np.stack(T_preds, axis=0)
        u_stack = np.stack(u_preds, axis=0)
        return T_stack.mean(0), T_stack.std(0), u_stack.mean(0), u_stack.std(0)

    def evaluate(
        self,
        r: torch.Tensor,
        t: torch.Tensor,
        T_true: np.ndarray,
        u_true: np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluate the ensemble mean against reference data, plus calibration.

        Coverage is the fraction of true values that actually fall
        inside the +/-1-std and +/-2-std bands around the ensemble
        mean. For a well-calibrated Gaussian uncertainty estimate these
        should land near 0.68 and 0.95; far below that means the
        ensemble is overconfident (its stated uncertainty doesn't
        reflect its actual error), which is worth reporting honestly
        rather than only showing the mean prediction.

        Args:
            r: Dimensionless radius of the reference points, shape (N, 1).
            t: Dimensionless time of the reference points, shape (N, 1).
            T_true: Reference dimensionless temperature, shape (N, 1).
            u_true: Reference dimensionless pore pressure, shape (N, 1).

        Returns:
            Dictionary with MSE/relative-L2/NRMSE for T and u (computed
            on the ensemble mean), plus 1-sigma/2-sigma coverage for
            each.
        """
        T_mean, T_std, u_mean, u_std = self.predict(r, t)

        T_mse, T_rel_l2, T_nrmse = error_metrics(T_mean, T_true)
        u_mse, u_rel_l2, u_nrmse = error_metrics(u_mean, u_true)

        return {
            "T_mse": T_mse,
            "T_rel_l2": T_rel_l2,
            "T_nrmse": T_nrmse,
            "u_mse": u_mse,
            "u_rel_l2": u_rel_l2,
            "u_nrmse": u_nrmse,
            "T_coverage_1sigma": self._coverage(T_mean, T_std, T_true, 1.0),
            "T_coverage_2sigma": self._coverage(T_mean, T_std, T_true, 2.0),
            "u_coverage_1sigma": self._coverage(u_mean, u_std, u_true, 1.0),
            "u_coverage_2sigma": self._coverage(u_mean, u_std, u_true, 2.0),
        }

    @staticmethod
    def _coverage(
        mean: np.ndarray, std: np.ndarray, true: np.ndarray, n_sigma: float
    ) -> float:
        """Fraction of true values within n_sigma standard deviations of mean."""
        within = np.abs(true - mean) <= (n_sigma * std + 1e-12)
        return float(within.mean())
