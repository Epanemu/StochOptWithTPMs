"""
Neural Network Probabilistic Model (NNPM) for constraint satisfaction prediction.

Uses a neural network to learn P(sat=1 | x) where x are decision variables.
The NN is trained with various loss functions (BOLT, Brier Score, Focal Loss)
using fresh scenario batches sampled at each training step.
Label smoothing is implemented by averaging satisfaction over multiple scenarios
per decision sample.

The trained model is encoded into a Pyomo MILP formulation using OMLT.
"""

import logging
import math
import os
import tempfile
from typing import Any, Dict, List, Optional, cast

import numpy as np
import numpy.typing as npt
import pyomo.environ as pyo
import torch
import torch.nn as nn

from stochopt.problem.base import BaseProblem

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


class BOLTLoss(nn.Module):
    """
    Bayes Optimal Learning Threshold (BOLT) loss for binary classification.
    From arXiv:2501.07754v1.
    """

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        loss = targets * (1 - probs) + (1 - targets) * probs
        return loss.mean()


class BrierScoreLoss(nn.Module):
    """
    Brier Score (Mean Squared Error for Probabilities).
    BS = mean((probs - targets)^2)
    """

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        loss = (probs - targets) ** 2
        return loss.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    FL = - (1-p)^gamma * log(p) if y=1
         - p^gamma * log(1-p) if y=0
    """

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        eps = 1e-8
        probs = probs.clamp(eps, 1.0 - eps)

        loss = -targets * ((1 - probs) ** self.gamma) * torch.log(probs) - (
            1 - targets
        ) * (probs**self.gamma) * torch.log(1 - probs)
        return loss.mean()


class BCELoss(nn.Module):
    """
    Standard Binary Cross Entropy loss (supporting soft targets).
    """

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return nn.functional.binary_cross_entropy_with_logits(logits, targets)


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------


class SatisfactionNN(nn.Module):
    """Feedforward neural network for predicting constraint satisfaction."""

    def __init__(self, input_size: int, hidden_sizes: List[int]):
        super().__init__()
        layers: list[nn.Module] = []
        prev_size = input_size
        for h_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, h_size))
            layers.append(nn.ReLU())
            prev_size = h_size
        # Final linear output – 1 logit
        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.network(x))


# ---------------------------------------------------------------------------
# NNPM wrapper
# ---------------------------------------------------------------------------


class NNPM:
    """Neural Network Probabilistic Model wrapper."""

    def __init__(self) -> None:
        self.model: Optional[SatisfactionNN] = None
        self.input_bounds: Optional[Dict[int, tuple]] = None
        self.n_x: int = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(
        self,
        problem: BaseProblem,
        train_samples: npt.NDArray[np.float64],
        *,
        epochs: int = 1000,
        batch_size: int = 256,
        lr: float = 1e-3,
        hidden_size_factors: Optional[List[float]] = None,
        min_hidden_size: int = 5,
        max_hidden_size: int = 100,
        val_size: int = 10000,
        log_every: int = 10,
        seed: int = 42,
        loss_type: str = "bolt",
        focal_gamma: float = 2.0,
        folder: str | None = None,
    ) -> "NNPM":
        """
        Train the NNPM on fresh batches of decision samples and scenarios.

        Args:
            problem: Optimization problem instance.
            train_samples: Samples of uncertain parameters for reference.
            epochs: Number of training epochs.
            batch_size: Number of decision samples AND scenario samples per epoch.
            lr: Learning rate.
            hidden_size_factors: Factors to determine layer widths from input size.
            min_hidden_size: Minimum hidden layer width.
            max_hidden_size: Maximum hidden layer width.
            val_size: Number of samples in the validation set.
            log_every: Interval for logging to MLflow.
            seed: Random seed.
            loss_type: Type of loss function to use.
            focal_gamma: Gamma parameter for focal loss.
            folder: Folder to save the trained model.

        Returns:
            self
        """
        import mlflow

        torch.manual_seed(seed)
        np.random.seed(seed)

        if folder is None:
            folder = tempfile.mkdtemp()

        best_checkpoint_path = os.path.join(folder, "best_checkpoint.pt")

        # Determine input dimensionality
        x_probe = problem.generate_decision_samples(1, seed=seed, xi=train_samples)
        self.n_x = x_probe.shape[1]

        # Determine hidden sizes
        if hidden_size_factors is None:
            hidden_size_factors = [1.0, 1.0, 1.0]
        hidden_sizes = [
            max(min_hidden_size, min(max_hidden_size, math.ceil(f * self.n_x)))
            for f in hidden_size_factors
        ]

        logger.info(f"NNPM architecture: input={self.n_x}, hidden={hidden_sizes}")
        logger.info(
            f"Loss: {loss_type}, Label Smoothing: {batch_size} scenarios/decision"
        )

        # Build model and optimizer
        self.model = SatisfactionNN(self.n_x, hidden_sizes).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        criterion: nn.Module
        if loss_type == "bolt":
            criterion = BOLTLoss()
        elif loss_type == "brier":
            criterion = BrierScoreLoss()
        elif loss_type == "focal":
            criterion = FocalLoss(gamma=focal_gamma)
        elif loss_type == "bce":
            criterion = BCELoss()
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        # Build validation set
        logger.info("Generating validation set...")
        val_xi = problem.generate_samples(val_size, seed=seed + 1000)
        val_x = problem.generate_decision_samples(
            val_size, seed=seed + 2000, xi=train_samples
        )
        # compute mean satisfaction
        val_sat = np.zeros(val_size)
        for i in range(val_size):
            x_i_rep = np.tile(val_x[i], (val_size, 1))
            val_sat[i] = problem.compute_satisfaction(val_xi, x_i_rep).mean()
        val_sat = val_sat.astype(float).flatten()

        val_x_t = torch.tensor(val_x, dtype=torch.float32, device=self.device)
        val_sat_t = torch.tensor(val_sat, dtype=torch.float32, device=self.device)

        # Bounds tracking
        bounds_min = val_x.min(axis=0).copy()
        bounds_max = val_x.max(axis=0).copy()

        best_val_loss = float("inf")

        # Training loop
        for epoch in range(epochs):
            self.model.train()

            x_batch = problem.generate_decision_samples(batch_size, xi=train_samples)
            xi_samples = problem.generate_samples(batch_size)

            targets = np.zeros(batch_size, dtype=np.float32)
            for i in range(batch_size):
                x_i_rep = np.tile(x_batch[i], (batch_size, 1))
                targets[i] = problem.compute_satisfaction(xi_samples, x_i_rep).mean()

            # Update bounds
            np.minimum(bounds_min, x_batch.min(axis=0), out=bounds_min)
            np.maximum(bounds_max, x_batch.max(axis=0), out=bounds_max)

            x_t = torch.tensor(x_batch, dtype=torch.float32, device=self.device)
            targets_t = torch.tensor(targets, dtype=torch.float32, device=self.device)

            optimizer.zero_grad()
            logits = self.model(x_t).squeeze(-1)
            loss = criterion(logits, targets_t)
            loss.backward()
            optimizer.step()

            if epoch % log_every == 0 or epoch == epochs - 1:
                self.model.eval()
                with torch.no_grad():
                    val_logits = self.model(val_x_t).squeeze(-1)
                    val_loss = criterion(val_logits, val_sat_t)
                    # val_preds = (torch.sigmoid(val_logits) > 0.5).float()
                    val_preds = torch.sigmoid(val_logits)
                    # val_acc = (val_preds == val_sat_t).float().mean()
                    val_mae = (val_preds - val_sat_t).abs().mean()

                mlflow.log_metric("nn_train_loss", loss.item(), step=epoch)
                mlflow.log_metric("nn_val_loss", val_loss.item(), step=epoch)
                # mlflow.log_metric("nn_val_accuracy", val_acc.item(), step=epoch)
                mlflow.log_metric("nn_val_mae", val_mae.item(), step=epoch)
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_loss": val_loss,
                        },
                        best_checkpoint_path,
                    )

                if epoch % (log_every * 10) == 0:
                    # logger.info(f"Epoch {epoch}/{epochs}: loss={loss.item():.4f}, val_acc={val_acc.item():.4f}")
                    logger.info(
                        f"Epoch {epoch}/{epochs}: loss={loss.item():.4f}, val_mae={val_mae.item():.4f}"
                    )

        mlflow.log_artifact(best_checkpoint_path, artifact_path="checkpoints")
        mlflow.log_metric("best_val_loss", best_val_loss)

        # Store input bounds
        margin = (bounds_max - bounds_min) * 0.01 + 1e-6
        self.input_bounds = {
            i: (float(bounds_min[i] - margin[i]), float(bounds_max[i] + margin[i]))
            for i in range(self.n_x)
        }
        return self

    def predict_prob(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Predict the probability of satisfaction for decision vectors."""
        if self.model is None:
            raise ValueError("Model not trained")
        self.model.eval()
        with torch.no_grad():
            x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
            logits = self.model(x_t).squeeze(-1)
            probs = torch.sigmoid(logits)
        return probs.cpu().numpy()

    def encode(
        self,
        model_block: pyo.Block,
        inputs: List[pyo.Var],
        solver: str = "gurobi",
        **kwargs: Any,
    ) -> pyo.Var:
        """Encode the trained neural network into a Pyomo model using OMLT."""
        if self.model is None:
            raise ValueError("Model not trained")
        if self.input_bounds is None:
            raise ValueError("Input bounds not computed")

        from omlt import OmltBlock
        from omlt.io import (
            load_onnx_neural_network_with_bounds,
            write_onnx_model_with_bounds,
        )
        from omlt.neuralnet import FullSpaceNNFormulation

        self.model.eval()
        self.model.cpu()
        n_inputs = len(inputs)
        dummy_input = torch.randn(1, n_inputs)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx_path = f.name
            torch.onnx.export(
                self.model,
                (dummy_input,),
                onnx_path,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            )
            write_onnx_model_with_bounds(
                onnx_path, None, list(self.input_bounds.values())
            )

        network_definition = load_onnx_neural_network_with_bounds(onnx_path)
        model_block.nn = OmltBlock()
        model_block.nn.build_formulation(FullSpaceNNFormulation(network_definition))

        print("model encoded to pyomo")

        for i, inp_var in enumerate(inputs):
            model_block.add_component(
                f"nn_input_link_{i}",
                pyo.Constraint(expr=model_block.nn.inputs[i] == inp_var),
            )

        return model_block.nn.outputs[0]
