from io import BytesIO
from typing import Tuple
from PIL import Image
import requests
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import numpy as np
from sklearn.calibration import calibration_curve
from collections import OrderedDict

from .settings import settings
from .explain import ResNetGradCAM

class TemperatureScaler(nn.Module):
    def __init__(self, t: float):
        super().__init__()
        self.t = nn.Parameter(torch.tensor([t], dtype=torch.float32))
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.t.clamp(min=1e-3)

def enable_mc_dropout(m: nn.Module):
    if isinstance(m, nn.Dropout):
        m.train()

class ResNet18Fire(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        try:
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except Exception:
            self.backbone = models.resnet18(weights=None)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feats, num_classes)
    def forward(self, x):
        return self.backbone(x)

class InferenceEngine:
    def __init__(self):
        self.device = torch.device(settings.DEVICE if torch.cuda.is_available() else "cpu")
        self.model = ResNet18Fire().to(self.device)

        weights_path = settings.MODELS_DIR / settings.MODEL_WEIGHTS
        if weights_path.exists():
            self._load_flexible_resnet18(weights_path)

        self.model.eval()
        self.temp = TemperatureScaler(settings.TEMP_INIT).to(self.device)

        self.tf = T.Compose([
            T.Resize((settings.IMG_SIZE, settings.IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self._calibration_buffer = []

    def _load_flexible_resnet18(self, weights_path):
        obj = torch.load(weights_path, map_location=self.device)
        if isinstance(obj, dict) and "state_dict" in obj:
            sd = obj["state_dict"]
        elif isinstance(obj, dict):
            sd = obj
        else:
            sd = obj.state_dict()

        remapped = OrderedDict()
        for k, v in sd.items():
            k2 = k
            if k2.startswith("model."):
                k2 = k2[len("model."):]
            if not k2.startswith("backbone."):
                if k2.startswith(("conv1", "bn1", "layer", "fc", "classifier")):
                    k2 = "backbone." + k2
            if k2.startswith("backbone.classifier"):
                k2 = k2.replace("backbone.classifier", "backbone.fc")
            remapped[k2] = v

        model_sd = self.model.state_dict()
        filtered = OrderedDict()
        for k, v in remapped.items():
            if k in model_sd and model_sd[k].shape == v.shape:
                filtered[k] = v

        missing, unexpected = self.model.load_state_dict(filtered, strict=False)
        print(f"[weights] loaded={len(filtered)} | missing={len(missing)} | unexpected={len(unexpected)}")

        try:
            if isinstance(obj, dict) and "temp" in obj:
                t = float(obj["temp"])
                with torch.no_grad():
                    self.temp.t.copy_(torch.tensor([t], dtype=torch.float32, device=self.temp.t.device))
                print(f"[calibration] loaded temperature t={t:.3f}")
        except Exception:
            pass

    def fetch_image(self, url: str) -> Image.Image:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")

    def _prep(self, img: Image.Image) -> torch.Tensor:
        return self.tf(img).unsqueeze(0).to(self.device)

    def predict(self, img: Image.Image, T_mc: int = 15) -> Tuple[str, float, float, Image.Image]:
        x = self._prep(img)

        torch.set_grad_enabled(False)
        self.model.eval()
        self.model.apply(enable_mc_dropout)
        logits_list = []
        for _ in range(T_mc):
            logits = self.model(x)
            logits = self.temp(logits)
            logits_list.append(logits)
        logits_stack = torch.stack(logits_list, dim=0)
        probs = torch.softmax(logits_stack, dim=-1)[:, 0, 1]
        p_mean = probs.mean().item()

        eps = 1e-9
        ent = -(p_mean * np.log(p_mean + eps) + (1 - p_mean) * np.log(1 - p_mean + eps))
        label = "fire" if p_mean >= 0.5 else "nofire"

        torch.set_grad_enabled(True)
        gradcam = ResNetGradCAM(self.model.backbone, target_layer_name="layer4")
        x_gc = self._prep(img)
        overlay = gradcam.generate(x_gc, img)
        torch.set_grad_enabled(False)

        return label, float(p_mean), float(ent), overlay

    def reliability_curve(self, probs: np.ndarray, labels: np.ndarray, n_bins: int = 10):
        frac_pos, mean_pred = calibration_curve(labels, probs, n_bins=n_bins, strategy="uniform")
        ece = float(np.abs(frac_pos - mean_pred).mean())
        points = [dict(prob_bin_center=float(mp), accuracy=float(fp)) for mp, fp in zip(mean_pred, frac_pos)]
        return ece, points

engine = InferenceEngine()
