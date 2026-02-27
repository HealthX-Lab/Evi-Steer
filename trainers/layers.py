import torch
import torch.nn as nn
import torch.nn.functional as F


class EviTextAdapter(nn.Module):
    def __init__(self, embed_dim: int, dimension: int = 8):
        super().__init__()
        self.dimension = int(dimension)

        self.down = nn.Linear(embed_dim, 2 * self.dimension, bias=False)
        self.up = nn.Linear(self.dimension, embed_dim, bias=False)
        nn.init.zeros_(self.up.weight)

        self.conf_gate = nn.Sequential(
            nn.Linear(self.dimension, 1, bias=False),
            nn.Sigmoid()
        )
        self.global_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor):
        """
        x: (C, T, D)
        Returns:
          txt_delta: (C, T, D)
          txt_alpha_cls: (C, r)   <-- for KL reg
        """
        z = self.down(x)                      # (C, T, 2r)
        z_mu, z_unc = z.chunk(2, dim=-1)      # (C, T, r), (C, T, r)

        # evidential concentration
        energy = z_unc ** 2
        evidence = F.softplus(energy) + 1e-6
        alpha = evidence + 1.0                # (C, T, r)
        uncertainty = 1.0 / alpha             # (C, T, r)

        confidence = self.conf_gate(1.0 - uncertainty)  # (C, T, 1)
        delta = self.up(z_mu)                           # (C, T, D)
        global_gate = torch.sigmoid(self.global_gate)

        txt_delta = global_gate * confidence * delta
        txt_alpha_cls = alpha[:, 0, :]                  # (C, r)

        return txt_delta, txt_alpha_cls


class EviVisionAdapter(nn.Module):
    def __init__(self, embed_dim: int, dimension: int = 8):
        super().__init__()
        self.dimension = int(dimension)

        self.down = nn.Linear(embed_dim, 2 * self.dimension, bias=False)
        self.up = nn.Linear(self.dimension, embed_dim, bias=False)
        nn.init.zeros_(self.up.weight)

        self.conf_gate = nn.Sequential(
            nn.Linear(self.dimension, 1, bias=False),
            nn.Sigmoid()
        )
        self.global_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, text_alpha_cls: torch.Tensor):
        """
        x: (B, N, D)
        text_alpha_cls: (C, r)  (only used to form DS gate)
        Returns:
          img_delta: (B, N, D)
          img_alpha: (B, N, r)   <-- for KL reg
        """
        z = self.down(x)                    # (B, N, 2r)
        z_mu, z_unc = z.chunk(2, dim=-1)    # (B, N, r), (B, N, r)
        delta = self.up(z_mu)               # (B, N, D)

        # vision evidential concentration
        energy_v = z_unc ** 2
        evidence_v = F.softplus(energy_v) + 1e-6
        alpha_v = evidence_v + 1.0          # (B, N, r)
        uncertainty_v = 1.0 / alpha_v       # (B, N, r)

        # Use text alpha to derive text uncertainty for DS
        # (keeps DS behavior without returning uncertainty tensors)
        u_t = (1.0 / text_alpha_cls).mean(dim=0)  # (r,)

        m_v_update = 1.0 - uncertainty_v
        m_v_omega = uncertainty_v

        m_t_update = (1.0 - u_t).view(1, 1, -1)
        m_t_omega = u_t.view(1, 1, -1)

        conflict = m_t_update * m_v_omega + m_t_omega * m_v_update
        fused_conf = (m_t_update * m_v_update) / (1.0 - conflict + 1e-6)  # (B, N, r)

        ds_conf = self.conf_gate(fused_conf)  # (B, N, 1)
        global_gate = torch.sigmoid(self.global_gate)

        img_delta = global_gate * ds_conf * delta

        return img_delta, alpha_v


class EviSteer(nn.Module):
    def __init__(self, img_dim: int, text_dim: int, dimension: int = 8):
        super().__init__()
        self.text_adapter = EviTextAdapter(text_dim, dimension)
        self.vision_adapter = EviVisionAdapter(img_dim, dimension)

    def forward(self, x_img: torch.Tensor, x_txt: torch.Tensor):
        """
        Returns:
          img_delta: (B, N, D)
          txt_delta: (C, T, D)
          txt_alpha_cls: (C, r)
          img_alpha: (B, N, r)
        """
        txt_delta, txt_alpha_cls = self.text_adapter(x_txt)
        img_delta, img_alpha = self.vision_adapter(x_img, txt_alpha_cls)
        return img_delta, txt_delta, txt_alpha_cls, img_alpha