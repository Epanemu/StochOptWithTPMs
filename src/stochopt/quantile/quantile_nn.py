"""
Quantile Neural Network (QuantileNN) for constraint margin prediction.

Uses a neural network to learn the q-th quantile of the satisfaction margin f(x, xi) given x.
The NN is trained using the Pinball (Quantile) loss.

The trained model can be encoded into a Pyomo MILP formulation using OMLT.
"""

import logging
import math
import os
import tempfile
from typing import Any, List, Optional, cast

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


class PinballLoss(nn.Module):
    """
    Pinball Loss (Quantile Loss) for learning the q-th quantile.
    L_q(y, y_hat) = max(q * (y - y_hat), (q - 1) * (y - y_hat))
    """

    def __init__(self, q: float = 0.95):
        super().__init__()
        self.q = q

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        errors = targets - preds
        loss = torch.max(self.q * errors, (self.q - 1) * errors)
        return loss.mean()


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------


class QuantileNet(nn.Module):
    """Feedforward neural network for predicting constraint margin quantile."""

    def __init__(self, input_size: int, hidden_sizes: List[int]):
        super().__init__()
        layers: list[nn.Module] = []
        prev_size = input_size
        for h_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, h_size))
            layers.append(nn.ReLU())
            prev_size = h_size
        # Final linear output – 1 value (the quantile of the margin)
        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.network(x))


# ---------------------------------------------------------------------------
# QuantileNN wrapper
# ---------------------------------------------------------------------------


class QuantileNN:
    """Quantile Neural Network wrapper."""

    def __init__(self) -> None:
        self.model: Optional[QuantileNet] = None
        self.input_min: Optional[np.ndarray] = None
        self.input_max: Optional[np.ndarray] = None
        self.target_scale: float = 1.0
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
        quantile: float = 0.95,
        folder: str | None = None,
    ) -> "QuantileNN":
        """
        Train the QuantileNN on fresh batches of decision samples and scenarios.

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
            quantile: Target quantile to learn (e.g., 0.95 for risk=0.05).
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
        x_probe = problem.generate_decision_samples(1, seed=seed)
        self.n_x = x_probe.shape[1]

        # Determine hidden sizes
        if hidden_size_factors is None:
            hidden_size_factors = [1.0, 1.0, 1.0]
        hidden_sizes = [
            max(min_hidden_size, min(max_hidden_size, math.ceil(f * self.n_x)))
            for f in hidden_size_factors
        ]

        logger.info(f"QuantileNN architecture: input={self.n_x}, hidden={hidden_sizes}")
        logger.info(f"Loss: PinballLoss, Quantile (q): {quantile}")

        # Build model and optimizer
        self.model = QuantileNet(self.n_x, hidden_sizes).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        criterion = PinballLoss(q=quantile)

        # Build validation set
        logger.info("Generating validation set...")
        val_xi = problem.generate_samples(val_size, seed=seed + 1000)
        val_x = problem.generate_decision_samples(val_size, seed=seed + 2000)

        # compute margin directly for each pair (val_x_i, val_xi_i)
        val_margin = problem.compute_margin(val_xi, val_x).astype(float).flatten()

        # Compute and store normalization parameters
        bounds_min = val_x.min(axis=0).copy()
        bounds_max = val_x.max(axis=0).copy()
        # Add a small buffer to avoid boundary issues
        buffer = (bounds_max - bounds_min) * 0.05 + 1e-5
        self.input_min = bounds_min - buffer
        self.input_max = bounds_max + buffer
        self.target_scale = float(np.max(np.abs(self.input_max)))
        if self.target_scale < 1e-5:
            self.target_scale = 1.0

        # Normalize validation set
        val_x_norm = (val_x - self.input_min) / (self.input_max - self.input_min + 1e-9)
        val_margin_norm = val_margin / self.target_scale

        val_x_t = torch.tensor(val_x_norm, dtype=torch.float32, device=self.device)
        val_margin_t = torch.tensor(
            val_margin_norm, dtype=torch.float32, device=self.device
        )

        best_val_loss = float("inf")

        # Training loop
        for epoch in range(epochs):
            self.model.train()

            x_batch = problem.generate_decision_samples(batch_size)
            xi_samples = problem.generate_samples(batch_size)

            targets = problem.compute_margin(xi_samples, x_batch).flatten()

            # Normalize training batch
            x_batch_norm = (x_batch - self.input_min) / (
                self.input_max - self.input_min + 1e-9
            )
            targets_norm = targets / self.target_scale

            x_t = torch.tensor(x_batch_norm, dtype=torch.float32, device=self.device)
            targets_t = torch.tensor(
                targets_norm, dtype=torch.float32, device=self.device
            )

            optimizer.zero_grad()
            preds = self.model(x_t).squeeze(-1)
            loss = criterion(preds, targets_t)
            loss.backward()
            optimizer.step()

            if epoch % log_every == 0 or epoch == epochs - 1:
                self.model.eval()
                with torch.no_grad():
                    val_preds = self.model(val_x_t).squeeze(-1)
                    val_loss = criterion(val_preds, val_margin_t)
                    # Compute MAE in actual units
                    val_mae = (
                        (
                            (val_preds * self.target_scale)
                            - torch.tensor(
                                val_margin, dtype=torch.float32, device=self.device
                            )
                        )
                        .abs()
                        .mean()
                    )

                if mlflow.active_run():
                    mlflow.log_metric("qnn_train_loss", loss.item(), step=epoch)
                    mlflow.log_metric("qnn_val_loss", val_loss.item(), step=epoch)
                    mlflow.log_metric("qnn_val_mae", val_mae.item(), step=epoch)

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
                    logger.info(
                        f"Epoch {epoch}/{epochs}: loss={loss.item():.4f}, "
                        f"val_loss={val_loss.item():.4f}, val_mae={val_mae.item():.4f}"
                    )

        if mlflow.active_run():
            mlflow.log_artifact(best_checkpoint_path, artifact_path="checkpoints")
            mlflow.log_metric("best_val_loss", best_val_loss)

        # Load best checkpoint
        if os.path.exists(best_checkpoint_path):
            checkpoint = torch.load(best_checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(
                f"Loaded best checkpoint from epoch {checkpoint['epoch']} with val_loss {checkpoint['val_loss']:.4f}"
            )

        return self

    def predict_quantile(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Predict the quantile of the margin for decision vectors."""
        if self.model is None:
            raise ValueError("Model not trained")
        if self.input_min is None or self.input_max is None:
            raise ValueError("Normalization parameters not set")
        self.model.eval()
        with torch.no_grad():
            x_norm = (x - self.input_min) / (self.input_max - self.input_min + 1e-9)
            x_t = torch.tensor(x_norm, dtype=torch.float32, device=self.device)
            preds = self.model(x_t).squeeze(-1)
            # Scale prediction back
            preds_scaled = preds * self.target_scale
        return cast(npt.NDArray[np.float64], preds_scaled.cpu().numpy())

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
        if self.input_min is None or self.input_max is None:
            raise ValueError("Normalization parameters not set")

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

        # The OMLT network will take normalized inputs in [0, 1]
        onnx_input_bounds = {i: (0.0, 1.0) for i in range(n_inputs)}

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
                onnx_path, None, list(onnx_input_bounds.values())
            )

        network_definition = load_onnx_neural_network_with_bounds(onnx_path)
        model_block.nn = OmltBlock()
        model_block.nn.build_formulation(FullSpaceNNFormulation(network_definition))

        print("QuantileNN model encoded to pyomo")

        # Link variables: nn.inputs[i] * (input_max - input_min) == inp_var - input_min
        for i, inp_var in enumerate(inputs):
            range_i = float(self.input_max[i] - self.input_min[i])
            model_block.add_component(
                f"nn_input_link_{i}",
                pyo.Constraint(
                    expr=model_block.nn.inputs[i] * range_i
                    == inp_var - float(self.input_min[i])
                ),
            )

        return model_block.nn.outputs[0]
