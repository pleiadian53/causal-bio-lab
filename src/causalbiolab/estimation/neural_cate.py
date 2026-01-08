"""Neural network-based CATE estimators with modern architectures.

This module implements advanced neural CATE estimators including:
- Siamese T-Learner: Shared encoder with separate heads
- JEPA-style Causal Learner: Joint embedding predictive architecture
- Contrastive CATE: Representation learning for treatment effects

These methods are particularly useful for high-dimensional biological data
(gene expression, images, etc.) where representation learning is beneficial.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .cate import CATEEstimator, CATEResult

if TYPE_CHECKING:
    from numpy.typing import NDArray


class SharedEncoder(nn.Module):
    """Shared encoder for Siamese architecture.
    
    Maps high-dimensional inputs to lower-dimensional representations.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [512, 256, 128],
        dropout: float = 0.1,
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class OutcomeHead(nn.Module):
    """Outcome prediction head for treatment arm."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z).squeeze(-1)


class SiameseTLearner(CATEEstimator):
    """Siamese T-Learner with shared encoder and separate heads.
    
    Architecture:
        x → SharedEncoder(x) → z
                              ↙  ↘
                        Head₀(z)  Head₁(z)
                            ↓       ↓
                          μ₀(x)   μ₁(x)
    
    CATE(x) = μ₁(x) - μ₀(x)
    
    Benefits:
    - Shared representation learning across treatment arms
    - Better with limited data per arm
    - Regularization through parameter sharing
    - Particularly effective for high-dimensional inputs
    
    Args:
        input_dim: Input feature dimension
        hidden_dims: List of hidden dimensions for encoder
        learning_rate: Learning rate for Adam optimizer
        batch_size: Batch size for training
        n_epochs: Number of training epochs
        device: 'cuda' or 'cpu'
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [512, 256, 128],
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        n_epochs: int = 100,
        device: str = "cpu",
        verbose: bool = False,
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = device
        self.verbose = verbose
        
        # Initialize networks
        self.encoder = SharedEncoder(input_dim, hidden_dims)
        self.head_0 = OutcomeHead(self.encoder.output_dim)
        self.head_1 = OutcomeHead(self.encoder.output_dim)
        
        # Move to device
        self.encoder.to(device)
        self.head_0.to(device)
        self.head_1.to(device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.head_0.parameters()) +
            list(self.head_1.parameters()),
            lr=learning_rate,
        )
    
    def fit(
        self,
        X: NDArray[np.floating],
        T: NDArray[np.integer],
        Y: NDArray[np.floating],
    ) -> "SiameseTLearner":
        X, T, Y = self._validate_inputs(X, T, Y)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        T_tensor = torch.FloatTensor(T).to(self.device)
        Y_tensor = torch.FloatTensor(Y).to(self.device)
        
        # Create dataset and loader
        dataset = TensorDataset(X_tensor, T_tensor, Y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training loop
        self.encoder.train()
        self.head_0.train()
        self.head_1.train()
        
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            for X_batch, T_batch, Y_batch in loader:
                self.optimizer.zero_grad()
                
                # Forward pass through shared encoder
                z = self.encoder(X_batch)
                
                # Predict outcomes for both arms
                y0_pred = self.head_0(z)
                y1_pred = self.head_1(z)
                
                # Compute loss only for observed outcomes
                # For control (T=0): use head_0
                # For treated (T=1): use head_1
                y_pred = T_batch * y1_pred + (1 - T_batch) * y0_pred
                loss = F.mse_loss(y_pred, Y_batch)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            if self.verbose and (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / n_batches
                print(f"Epoch {epoch + 1}/{self.n_epochs}, Loss: {avg_loss:.4f}")
        
        return self
    
    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        X = np.asarray(X)
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        self.encoder.eval()
        self.head_0.eval()
        self.head_1.eval()
        
        with torch.no_grad():
            z = self.encoder(X_tensor)
            y0 = self.head_0(z)
            y1 = self.head_1(z)
            cate = y1 - y0
        
        return cate.cpu().numpy()
    
    def get_representations(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Get learned representations from shared encoder."""
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        self.encoder.eval()
        with torch.no_grad():
            z = self.encoder(X_tensor)
        
        return z.cpu().numpy()


class JEPACausalLearner(CATEEstimator):
    """JEPA-style (Joint Embedding Predictive Architecture) for causal inference.
    
    Architecture:
        Context:   x → φ_context(x) → z_x
        Treatment: t → φ_treatment(t) → z_t
        Predictor: [z_x, z_t] → ψ(·) → ẑ_y
        Target:    y → φ_target(y) → z_y (EMA)
    
    Loss:
        L_pred = ||ẑ_y - z_y||²  (prediction in embedding space)
        L_inv = MMD(z_x[T=0], z_x[T=1])  (invariance to treatment)
    
    Key idea: Learn representations where:
    - z_x captures covariates relevant for outcomes
    - z_x is invariant to treatment assignment (causal)
    - Predictor learns E[Y | do(T), X] in embedding space
    
    Args:
        input_dim: Input feature dimension
        context_dim: Dimension of context embeddings
        treatment_dim: Dimension of treatment embeddings
        target_dim: Dimension of target embeddings
        learning_rate: Learning rate
        batch_size: Batch size
        n_epochs: Number of epochs
        lambda_inv: Weight for invariance loss
        ema_decay: Exponential moving average decay for target encoder
    """
    
    def __init__(
        self,
        input_dim: int,
        context_dim: int = 128,
        treatment_dim: int = 32,
        target_dim: int = 64,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        n_epochs: int = 100,
        lambda_inv: float = 0.1,
        ema_decay: float = 0.99,
        device: str = "cpu",
        verbose: bool = False,
    ):
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.treatment_dim = treatment_dim
        self.target_dim = target_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lambda_inv = lambda_inv
        self.ema_decay = ema_decay
        self.device = device
        self.verbose = verbose
        
        # Context encoder (for covariates X)
        self.context_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, context_dim),
        ).to(device)
        
        # Treatment encoder (for T)
        self.treatment_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, treatment_dim),
        ).to(device)
        
        # Predictor (z_x, z_t → z_y)
        self.predictor = nn.Sequential(
            nn.Linear(context_dim + treatment_dim, 128),
            nn.ReLU(),
            nn.Linear(128, target_dim),
        ).to(device)
        
        # Target encoder (for Y, with EMA)
        self.target_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, target_dim),
        ).to(device)
        
        # EMA target encoder (not trained directly)
        self.target_encoder_ema = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, target_dim),
        ).to(device)
        
        # Initialize EMA with same weights
        self.target_encoder_ema.load_state_dict(self.target_encoder.state_dict())
        
        # Optimizer (only for online networks)
        self.optimizer = torch.optim.Adam(
            list(self.context_encoder.parameters()) +
            list(self.treatment_encoder.parameters()) +
            list(self.predictor.parameters()) +
            list(self.target_encoder.parameters()),
            lr=learning_rate,
        )
    
    def _update_ema(self):
        """Update EMA target encoder."""
        for param, ema_param in zip(
            self.target_encoder.parameters(),
            self.target_encoder_ema.parameters()
        ):
            ema_param.data.mul_(self.ema_decay).add_(
                param.data, alpha=1 - self.ema_decay
            )
    
    def _mmd_loss(self, z0: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
        """Maximum Mean Discrepancy for distribution matching."""
        # Simple RBF kernel MMD
        def rbf_kernel(x, y, sigma=1.0):
            dist = torch.cdist(x, y, p=2)
            return torch.exp(-dist ** 2 / (2 * sigma ** 2))
        
        k_00 = rbf_kernel(z0, z0).mean()
        k_11 = rbf_kernel(z1, z1).mean()
        k_01 = rbf_kernel(z0, z1).mean()
        
        return k_00 + k_11 - 2 * k_01
    
    def fit(
        self,
        X: NDArray[np.floating],
        T: NDArray[np.integer],
        Y: NDArray[np.floating],
    ) -> "JEPACausalLearner":
        X, T, Y = self._validate_inputs(X, T, Y)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        T_tensor = torch.FloatTensor(T).unsqueeze(1).to(self.device)
        Y_tensor = torch.FloatTensor(Y).unsqueeze(1).to(self.device)
        
        dataset = TensorDataset(X_tensor, T_tensor, Y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training loop
        for epoch in range(self.n_epochs):
            epoch_loss_pred = 0.0
            epoch_loss_inv = 0.0
            n_batches = 0
            
            for X_batch, T_batch, Y_batch in loader:
                self.optimizer.zero_grad()
                
                # Encode context (covariates)
                z_x = self.context_encoder(X_batch)
                
                # Encode treatment
                z_t = self.treatment_encoder(T_batch)
                
                # Predict target embedding
                z_y_pred = self.predictor(torch.cat([z_x, z_t], dim=1))
                
                # Get target embedding (EMA, no gradient)
                with torch.no_grad():
                    z_y_target = self.target_encoder_ema(Y_batch)
                
                # Prediction loss in embedding space
                loss_pred = F.mse_loss(z_y_pred, z_y_target)
                
                # Invariance loss: z_x should be similar for T=0 and T=1
                mask_0 = (T_batch == 0).squeeze()
                mask_1 = (T_batch == 1).squeeze()
                
                if mask_0.sum() > 0 and mask_1.sum() > 0:
                    z_x_0 = z_x[mask_0]
                    z_x_1 = z_x[mask_1]
                    loss_inv = self._mmd_loss(z_x_0, z_x_1)
                else:
                    loss_inv = torch.tensor(0.0, device=self.device)
                
                # Total loss
                loss = loss_pred + self.lambda_inv * loss_inv
                
                loss.backward()
                self.optimizer.step()
                
                # Update EMA
                self._update_ema()
                
                epoch_loss_pred += loss_pred.item()
                epoch_loss_inv += loss_inv.item()
                n_batches += 1
            
            if self.verbose and (epoch + 1) % 10 == 0:
                avg_pred = epoch_loss_pred / n_batches
                avg_inv = epoch_loss_inv / n_batches
                print(
                    f"Epoch {epoch + 1}/{self.n_epochs}, "
                    f"Pred Loss: {avg_pred:.4f}, Inv Loss: {avg_inv:.4f}"
                )
        
        return self
    
    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        X = np.asarray(X)
        X_tensor = torch.FloatTensor(X).to(self.device)
        n = len(X)
        
        self.context_encoder.eval()
        self.treatment_encoder.eval()
        self.predictor.eval()
        self.target_encoder_ema.eval()
        
        with torch.no_grad():
            # Encode context
            z_x = self.context_encoder(X_tensor)
            
            # Predict for T=0 and T=1
            z_t_0 = self.treatment_encoder(torch.zeros(n, 1, device=self.device))
            z_t_1 = self.treatment_encoder(torch.ones(n, 1, device=self.device))
            
            z_y_0 = self.predictor(torch.cat([z_x, z_t_0], dim=1))
            z_y_1 = self.predictor(torch.cat([z_x, z_t_1], dim=1))
            
            # CATE in embedding space (could decode, but we use norm as proxy)
            cate = (z_y_1 - z_y_0).norm(dim=1)
        
        return cate.cpu().numpy()
    
    def get_causal_representations(
        self, X: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Get causal (treatment-invariant) representations."""
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        self.context_encoder.eval()
        with torch.no_grad():
            z_x = self.context_encoder(X_tensor)
        
        return z_x.cpu().numpy()


def benchmark_neural_cate(
    X: NDArray[np.floating],
    T: NDArray[np.integer],
    Y: NDArray[np.floating],
    tau_true: NDArray[np.floating] | None = None,
    device: str = "cpu",
) -> dict[str, CATEResult]:
    """Benchmark neural CATE estimators against baselines.
    
    Args:
        X: Covariates (n_samples, n_features)
        T: Treatment (n_samples,)
        Y: Outcome (n_samples,)
        tau_true: True CATE for evaluation (if available)
        device: 'cuda' or 'cpu'
    
    Returns:
        Dictionary of CATEResult for each method
    """
    from .cate import SLearner, TLearner
    
    input_dim = X.shape[1]
    
    # Baseline methods
    estimators = {
        "S-Learner": SLearner(),
        "T-Learner": TLearner(),
        "Siamese-T": SiameseTLearner(
            input_dim=input_dim,
            n_epochs=50,
            device=device,
            verbose=False,
        ),
        "JEPA-Causal": JEPACausalLearner(
            input_dim=input_dim,
            n_epochs=50,
            device=device,
            verbose=False,
        ),
    }
    
    results = {}
    for name, est in estimators.items():
        print(f"Training {name}...")
        result = est.fit_predict(X, T, Y)
        results[name] = result
        
        # Compute RMSE if ground truth available
        if tau_true is not None:
            rmse = np.sqrt(np.mean((result.cate - tau_true) ** 2))
            print(f"  RMSE: {rmse:.4f}")
    
    return results
