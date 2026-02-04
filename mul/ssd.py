from typing import Dict, Iterable
import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .base import UnlearningStrategy
from torch.func import functional_call as _functional_call
from torch.func import grad as _grad
from torch.func import vmap as _vmap


class SSD(UnlearningStrategy):
    """Selective Synaptic Dampening (SSD) unlearning."""

    def run(self, base_model: pl.LightningModule) -> pl.LightningModule:
        """Execute SSD and return the dampened model."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_model = base_model.to(device)

        alpha = float(self.cfg.unlearn.alpha)
        lambda_val = float(self.cfg.unlearn["lambda"])

        fim_full = self._compute_fim_diag(base_model, self.dm.train_eval_dataloader(), loader_name="train")
        fim_forget = self._compute_fim_diag(base_model, self.dm.forget_eval_dataloader(), loader_name="forget")

        self._apply_dampening(base_model, fim_full, fim_forget, alpha, lambda_val)
        return base_model

    def _compute_fim_diag(
        self,
        model: pl.LightningModule,
        loader: Iterable,
        loader_name: str = "data",
    ) -> Dict[str, torch.Tensor]:
        """Approximate diagonal Fisher with average squared gradients."""
        was_training = model.training
        model.eval()
        device = next(model.parameters()).device
        fim = {n: torch.zeros_like(p, device=device) for n, p in model.named_parameters() if p.requires_grad}

        n_total = self._compute_fim_diag_vmap(model, loader, fim, device, loader_name)
        
        if was_training:
            model.train()

        for n in fim:
            fim[n].div_(max(n_total, 1))
        return fim

    def _compute_fim_diag_vmap(
        self,
        model: pl.LightningModule,
        loader: Iterable,
        fim: Dict[str, torch.Tensor],
        device: torch.device,
        loader_name: str,
    ) -> int:
        """Compute per-parameter Fisher diagonals with vmap."""
        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        params_for_grad = {k: v for k, v in params.items() if v.requires_grad}
        if not params_for_grad:
            return 0

        def loss_fn(params_for_grad, x, y):
            full_params = dict(params)
            full_params.update(params_for_grad)
            logits = _functional_call(model, (full_params, buffers), (x.unsqueeze(0),))
            logp = F.log_softmax(logits, dim=1)
            loss = -logp.gather(1, y.view(1, 1)).squeeze()
            return loss

        grad_fn = _grad(loss_fn)
        microbatch = int(self.cfg.unlearn.fim_microbatch_size)

        n_total = 0
        for x, y in tqdm(loader, desc=f"SSD Fisher [{loader_name}]", leave=False):
            x, y = x.to(device), y.to(device)
            bs = x.size(0)
            mb = microbatch if microbatch > 0 else bs
            # Use microbatches to limit memory during per-sample gradients.
            for start in range(0, bs, mb):
                x_mb = x[start:start + mb]
                y_mb = y[start:start + mb]
                per_sample_grads = _vmap(grad_fn, in_dims=(None, 0, 0))(params_for_grad, x_mb, y_mb)
                for name, grad in per_sample_grads.items():
                    fim[name].add_(grad.detach().pow(2).sum(dim=0))
                n_total += x_mb.size(0)
        return n_total

    @torch.no_grad()
    def _apply_dampening(
        self,
        model: pl.LightningModule,
        fim_full: Dict[str, torch.Tensor],
        fim_forget: Dict[str, torch.Tensor],
        alpha: float,
        lambda_val: float,
    ) -> None:
        """Apply SSD dampening to model parameters in place."""
        total_params = 0
        dampened_params = 0
        total_tensors = 0
        dampened_tensors = 0
        for name, param in model.named_parameters():
            fim_f = fim_forget.get(name)
            fim_d = fim_full.get(name)
            if fim_f is None or fim_d is None:
                continue
            mask = fim_f > (alpha * fim_d)
            total_tensors += 1
            total_params += mask.numel()
            dampened = int(mask.sum().item())
            dampened_params += dampened
            if dampened:
                dampened_tensors += 1
            if not mask.any().item():
                continue
            ratio = (lambda_val * fim_d[mask]) / (fim_f[mask] + 1e-12)
            beta = torch.ones_like(fim_f)
            beta[mask] = torch.minimum(ratio, torch.ones_like(ratio))
            param.mul_(beta)
            frac = 100.0 * dampened_params / float(total_params)
            print(f"[SSD] Dampening: {dampened_params}/{total_params} params ({frac:.2f}%), tensors {dampened_tensors}/{total_tensors}")
