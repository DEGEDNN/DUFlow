"""
DUFlow 双分支模型封装，用于跨数据集推理。
网络定义来自同目录下的 duflow_model.py，不依赖外部 flow_matching。
"""
import os
import torch
from typing import Optional

from .duflow_model import DualBranchParallelFlowModel
from crackfm.data.preprocess import prepare_condition


class DUFlowInferencer:
    """
    DUFlow 推理封装。

    Args:
        weight_path: 预训练权重路径（.pth 或 .pt）
        image_size: 输入尺寸
        device: 运行设备
        sample_steps: Flow Matching ODE 积分步数
        use_cal_only: 是否只跑特征分支（cal），不跑流分支
        fusion_type: 与训练时一致的融合方式
        base_channels: 基础通道数
    """

    def __init__(
        self,
        weight_path: str,
        image_size: int = 256,
        device: str = "cuda",
        sample_steps: int = 10,
        use_cal_only: bool = False,
        fusion_type: str = "concat",
        base_channels: int = 64,
    ):
        self.weight_path = weight_path
        self.image_size = image_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.sample_steps = sample_steps
        self.use_cal_only = use_cal_only

        self.model = DualBranchParallelFlowModel(
            in_channels=10,
            base_channels=base_channels,
            fusion_type=fusion_type,
        ).to(self.device)

        self._load_weights(weight_path)
        self.model.eval()

    def _load_weights(self, weight_path: str):
        if not os.path.exists(weight_path):
            raise FileNotFoundError(
                f"Weight file not found: {weight_path}\n"
                f"请将 DUFlow 预训练权重放到该路径，或修改配置指向正确路径。"
            )
        ckpt = torch.load(weight_path, map_location=self.device, weights_only=False)
        state_dict = ckpt.get("model", ckpt)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[WARN] Missing keys: {missing}")
        if unexpected:
            print(f"[WARN] Unexpected keys: {unexpected}")
        print(f"[DUFlowInferencer] Loaded weights from {weight_path}")

    @torch.no_grad()
    def predict_cal(self, img: torch.Tensor) -> torch.Tensor:
        """
        仅使用特征分支（cal）做快速推理。
        输入：dataset 返回的图像张量 [B, C, H, W]
        输出：[B, 1, H, W] 概率图
        """
        condition = prepare_condition(img, self.image_size).to(self.device)
        B, _, H, W = condition.shape

        dummy_xt = torch.zeros(B, 3, H, W, device=self.device, dtype=condition.dtype)
        dummy_noise = torch.zeros(B, 1, H, W, device=self.device, dtype=condition.dtype)
        dummy_t = torch.zeros(B, device=self.device)

        model_input = torch.cat([dummy_xt, condition, dummy_noise], dim=1)
        v_pred, cal = self.model(model_input, dummy_t)
        return cal

    @torch.no_grad()
    def predict_flow(self, img: torch.Tensor) -> torch.Tensor:
        """
        使用完整 Flow Matching 分支做推理。
        输入：dataset 返回的图像张量 [B, C, H, W]
        输出：[B, 1, H, W] 概率图
        """
        condition = prepare_condition(img, self.image_size).to(self.device)
        return euler_ode_sample(
            self.model, condition, self.sample_steps, self.image_size, self.device
        )

    def predict(self, img: torch.Tensor) -> torch.Tensor:
        if self.use_cal_only:
            return self.predict_cal(img)
        return self.predict_flow(img)


def euler_ode_sample(
    model: torch.nn.Module,
    condition: torch.Tensor,
    num_steps: int,
    image_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    使用 Euler ODE solver 从标准高斯噪声采样到目标掩码。
    model_input = [x_t(3ch), condition(6ch), noise_channel(1ch)] -> 10ch
    """
    B = condition.shape[0]
    dtype = condition.dtype

    noise_channel = torch.randn(B, 1, image_size, image_size, device=device, dtype=dtype)
    x = torch.randn(B, 3, image_size, image_size, device=device, dtype=dtype)
    dt = 1.0 / num_steps

    model.eval()
    with torch.no_grad():
        for i in range(num_steps):
            t_val = i * dt
            t_tensor = torch.full((B,), t_val, device=device, dtype=dtype)
            model_input = torch.cat([x, condition, noise_channel], dim=1)
            v_pred, _ = model(model_input, t_tensor)
            x = x + v_pred * dt

    # 将 [-1, 1] 映射回 [0, 1]
    pred = (x[:, 0:1] + 1.0) / 2.0
    return torch.clamp(pred, 0.0, 1.0)
