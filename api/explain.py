from typing import Optional
import torch
from PIL import Image
import numpy as np

class ResNetGradCAM:
    """
    Grad-CAM that can find a target layer by dotted path, e.g. "layer4" or "backbone.layer4".
    If not found, it falls back to the last convolution-like module.
    """
    def __init__(self, model: torch.nn.Module, target_layer_name: str = "layer4"):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None

        target_layer = self._resolve_layer(target_layer_name)
        if target_layer is None:
            target_layer = self._last_conv_like(self.model)

        if target_layer is None:
            raise RuntimeError("Could not locate a suitable target layer for Grad-CAM.")

        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _resolve_layer(self, dotted: str) -> Optional[torch.nn.Module]:
        try:
            return self.model.get_submodule(dotted)  # type: ignore[attr-defined]
        except Exception:
            parts = dotted.split(".")
            mod = self.model
            for p in parts:
                if not hasattr(mod, p):
                    return None
                mod = getattr(mod, p)
            return mod

    def _last_conv_like(self, m: torch.nn.Module) -> Optional[torch.nn.Module]:
        last = None
        for _, mod in m.named_modules():
            name = mod.__class__.__name__.lower()
            if "conv" in name or "batchnorm" in name or "relu" in name:
                last = mod
        return last

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, x: torch.Tensor, img: Image.Image, alpha: float = 0.45) -> Image.Image:
        logits = self.model(x)
        score = logits[:, 1].sum()

        self.model.zero_grad(set_to_none=True)
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)[0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
        cam_np = (cam.cpu().numpy() * 255).astype(np.uint8)

        mask = Image.fromarray(cam_np).resize(img.size, Image.BILINEAR).convert("L")
        base = img.convert("RGBA")
        overlay = Image.new("RGBA", base.size, (255, 0, 0, 0))
        overlay.putalpha(mask)
        out = Image.alpha_composite(base, overlay)
        return out.convert("RGB")
