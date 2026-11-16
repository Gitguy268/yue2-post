"""LoRA on YuE2-3B's AR experts, applied with forward hooks so "off" is bit-identical to the base model.

Targets: model.layers.N.self_attn.{q,k,v,o}_proj and model.layers.N.mlp.{gate,up,down}_proj (AR path only;
the nar_* experts are never touched). Adapter file keys: "model.layers.N.<module>.lora_A" / ".lora_B" (fp32),
metadata {"rank", "alpha"}. delta(x) = (x @ A.T @ B.T) * alpha / rank.

SPDX-License-Identifier: Apache-2.0
"""
from __future__ import annotations
from contextlib import contextmanager
import json
import math
import torch
from torch import nn
from safetensors.torch import save_file, load_file
from safetensors import safe_open

TARGETS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


class ARLora(nn.Module):
    def __init__(self, model, rank=16, alpha=32, init=True):
        super().__init__()
        self.rank, self.alpha, self.scale = rank, alpha, alpha / rank
        self.enabled = False
        self.names, self.A, self.B = [], nn.ParameterList(), nn.ParameterList()
        self._targets = []
        for i, layer in enumerate(model.model.layers):
            for t in TARGETS:
                lin = layer.get_submodule(t)
                a = torch.zeros(rank, lin.in_features, dtype=torch.float32)
                if init:
                    nn.init.kaiming_uniform_(a, a=math.sqrt(5))
                self.A.append(nn.Parameter(a))
                self.B.append(nn.Parameter(torch.zeros(lin.out_features, rank, dtype=torch.float32)))
                self.names.append(f"model.layers.{i}.{t}")
                self._targets.append(lin)
        self._handles = []

    def attach(self):
        device = self._targets[0].weight.device
        self.to(device)
        for idx, lin in enumerate(self._targets):
            self._handles.append(lin.register_forward_hook(self._hook(idx)))
        return self

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def _hook(self, idx):
        def hook(module, inputs, output):
            if not self.enabled:
                return output
            x = inputs[0]
            delta = (x.float() @ self.A[idx].t()) @ self.B[idx].t()
            return output + (delta * self.scale).to(output.dtype)
        return hook

    @contextmanager
    def on(self):
        previous, self.enabled = self.enabled, True
        try:
            yield
        finally:
            self.enabled = previous

    def state(self):
        out = {}
        for n, a, b in zip(self.names, self.A, self.B):
            out[f"{n}.lora_A"] = a.detach().float().cpu().contiguous()
            out[f"{n}.lora_B"] = b.detach().float().cpu().contiguous()
        return out

    def save(self, path, extra=None):
        meta = {"rank": str(self.rank), "alpha": str(self.alpha), "format": "yue2-ar-lora-v1",
                "targets": json.dumps(TARGETS)}
        meta.update({k: str(v) for k, v in (extra or {}).items()})
        save_file(self.state(), str(path), metadata=meta)

    @classmethod
    def load(cls, model, path):
        with safe_open(str(path), "pt") as f:
            meta = f.metadata() or {}
        lora = cls(model, rank=int(meta.get("rank", 16)), alpha=float(meta.get("alpha", 32)), init=False)
        tensors = load_file(str(path))
        with torch.no_grad():
            for n, a, b in zip(lora.names, lora.A, lora.B):
                a.copy_(tensors[f"{n}.lora_A"])
                b.copy_(tensors[f"{n}.lora_B"])
        return lora
