"""
Unified Model Trainer for Classical ML and PyTorch Deep Learning Models.
Handles class-weighted loss, optimization, inference latency measurement, and checkpoints.
"""

import time
import os
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score


class ModelTrainer:
    """
    Executes training and evaluation for all model families with accurate latency tracking.
    """
    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    def train_sklearn_model(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, Any]:
        """
        Trains scikit-learn / XGBoost model and measures fit time.
        """
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0
        
        # Validation predictions
        val_preds = model.predict(X_val)
        val_acc = float(accuracy_score(y_val, val_preds))
        val_f1 = float(f1_score(y_val, val_preds, average="weighted", zero_division=0))
        
        return {
            "model": model,
            "training_time_sec": fit_time,
            "val_accuracy": val_acc,
            "val_f1": val_f1
        }

    def train_pytorch_model(
        self,
        model: nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 15,
        batch_size: int = 32,
        lr: float = 0.001,
        weight_decay: float = 1e-4
    ) -> Dict[str, Any]:
        """
        Trains PyTorch deep learning model with class-weighted cross-entropy and AdamW.
        """
        model = model.to(self.device)
        
        # Compute class weights for loss
        classes, counts = np.unique(y_train, return_counts=True)
        total_samples = len(y_train)
        weights = total_samples / (len(classes) * counts)
        # Ensure 3 classes are present
        class_weight_tensor = torch.ones(3, dtype=torch.float32)
        for c, w in zip(classes, weights):
            if c < 3:
                class_weight_tensor[c] = float(w)
        class_weight_tensor = class_weight_tensor.to(self.device)
        
        criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        
        val_x = torch.from_numpy(X_val).float().to(self.device)
        val_y = torch.from_numpy(y_val).long().to(self.device)
        
        best_val_loss = float("inf")
        best_state = None
        t0 = time.perf_counter()
        
        for epoch in range(epochs):
            model.train()
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                
            # Validation step
            model.eval()
            with torch.no_grad():
                val_logits = model(val_x)
                val_loss = criterion(val_logits, val_y).item()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    
        total_train_time = time.perf_counter() - t0
        
        if best_state is not None:
            model.load_state_dict(best_state)
            
        model.eval()
        with torch.no_grad():
            val_logits = model(val_x)
            val_preds = torch.argmax(val_logits, dim=1).cpu().numpy()
            val_acc = float(accuracy_score(y_val, val_preds))
            val_f1 = float(f1_score(y_val, val_preds, average="weighted", zero_division=0))
            
        return {
            "model": model,
            "training_time_sec": total_train_time,
            "val_accuracy": val_acc,
            "val_f1": val_f1,
            "best_val_loss": best_val_loss
        }

    def measure_inference_latency(self, model: Any, sample_x: np.ndarray, num_trials: int = 100) -> Dict[str, float]:
        """
        Measures microsecond latency per single sample prediction.
        """
        latencies_ms = []
        is_torch = isinstance(model, nn.Module)
        
        if is_torch:
            model.eval()
            tensor_x = torch.from_numpy(sample_x[:1]).float().to(self.device)
            with torch.no_grad():
                # Warmup
                for _ in range(10):
                    _ = model(tensor_x)
                for _ in range(num_trials):
                    t0 = time.perf_counter()
                    _ = model(tensor_x)
                    latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        else:
            sample_1 = sample_x[:1]
            # Warmup
            for _ in range(10):
                _ = model.predict_proba(sample_1)
            for _ in range(num_trials):
                t0 = time.perf_counter()
                _ = model.predict_proba(sample_1)
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        latencies_ms = np.array(latencies_ms)
        return {
            "mean_latency_ms": float(np.mean(latencies_ms)),
            "median_latency_ms": float(np.median(latencies_ms)),
            "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
            "p99_latency_ms": float(np.percentile(latencies_ms, 99)),
            "throughput_samples_per_sec": float(1000.0 / (np.mean(latencies_ms) + 1e-6))
        }
