import torch
import torch.nn as nn
from open_clip import create_model_from_pretrained, get_tokenizer
from biomedclip.vision_modules import Block
from biomedclip.text_modules import BertLayer
from typing import Optional, Tuple, Union

from .layers import EviSteer


def load_biomedclip_to_device():
    model, preprocess = create_model_from_pretrained(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )

    vision_target_network = nn.Sequential(*[Block(768, 12) for _ in range(12)]).to("cuda")
    vision_network = model.visual.trunk.blocks.to("cuda")

    text_target_network = nn.ModuleList([BertLayer() for _ in range(12)]).to("cuda")
    text_network = model.text.transformer.encoder.layer.to("cuda")

    for target_param, param in zip(vision_target_network.parameters(), vision_network.parameters()):
        target_param.data.copy_(param.data)

    for target_param, param in zip(text_target_network.parameters(), text_network.parameters()):
        target_param.data.copy_(param.data)

    model.visual.trunk.blocks = vision_target_network.to("cuda")
    model.text.transformer.encoder.layer = text_target_network.to("cuda")

    return model.to("cuda").eval(), preprocess


def gamma_kl_to_unit_prior(alpha: torch.Tensor) -> torch.Tensor:
    """
    KL(Gamma(alpha, 1) || Gamma(1, 1)) elementwise, averaged.

    For same-rate=1:
      KL = (alpha - 1) * digamma(alpha) - lgamma(alpha)

    Minimum at alpha=1.
    """
    alpha = alpha.clamp_min(1e-6).float()
    return ((alpha - 1.0) * torch.digamma(alpha) - torch.lgamma(alpha)).mean()


class CustomCLIP(nn.Module):
    """
    Custom BiomedCLIP wrapper with Evi-Steer adapters + KL regularization.

    Model outputs (no hidden states):
      image_features, text_features, kl_reg

    Model outputs (with hidden states):
      image_features, hidden_states, text_features, kl_reg
    """

    def __init__(self, cfg, clip_model, output_hidden_states: bool = False):
        super().__init__()
        self.vision_model = clip_model.visual
        self.text_model = clip_model.text
        self.logit_scale = clip_model.logit_scale

        self.img_embed_dim = 768
        self.txt_embed_dim = 768

        self.output_hidden_states = output_hidden_states
        self.dtype = self.text_model.transformer.dtype
        self.device = "cuda"

        self.tokenizer = get_tokenizer(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )

        self.adapter_dimension = cfg.adapter_dimension
        self.adapter_depth = cfg.adapter_depth
        self.cfg = cfg

        self.evi_steer_adapters = nn.ModuleList(
            [
                EviSteer(
                    img_dim=self.img_embed_dim,
                    text_dim=self.txt_embed_dim,
                    dimension=self.adapter_dimension,
                ).to(self.device)
                for _ in range(self.adapter_depth)
            ]
        )

    def encode_text_image(
        self,
        tokenized_prompts: torch.Tensor,
        text_prompts: torch.Tensor,
        image: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
        Tuple[torch.Tensor, list, torch.Tensor, Optional[torch.Tensor]],
    ]:
        """
        Returns (no hidden states):
          image_features, text_features, kl_reg

        Returns (with hidden states):
          image_features, hidden_states, text_features, kl_reg
        """
        if attention_mask is None:
            attention_mask = (tokenized_prompts != self.text_model.config.pad_token_id).long()

        # Text embeddings from precomputed word embeddings
        x_txt = self.text_model.transformer.embeddings(inputs_embeds=text_prompts)

        # Build extended attention mask (BERT style)
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.dtype).min

        # Vision patch embedding
        trunk = self.vision_model.trunk
        x_img = trunk.patch_embed(image)  # (B, HW, D)
        x_img = trunk._pos_embed(x_img)
        x_img = trunk.norm_pre(x_img)

        hidden_states = []

        # Collect per-layer alphas for KL
        img_alpha_list = []  # each: (B, N, r)
        txt_alpha_list = []  # each: (C, r)

        for i, (block, layer) in enumerate(zip(trunk.blocks, self.text_model.transformer.encoder.layer)):
            if i == 0:
                x_img = block(x_img)
                x_txt = layer(x_txt, attention_mask=extended_attention_mask)

            elif i < self.adapter_depth:

                vis_delta, txt_delta, txt_alpha, img_alpha = self.evi_steer_adapters[i](x_img, x_txt)

                x_txt = x_txt + txt_delta
                x_img = x_img + vis_delta

                txt_alpha_list.append(txt_alpha)   # (C, r)
                img_alpha_list.append(img_alpha)   # (B, N, r)

                x_img = block(x_img)
                x_txt = layer(x_txt, attention_mask=extended_attention_mask)

            else:
                x_img = block(x_img)
                x_txt = layer(x_txt, attention_mask=extended_attention_mask)

            hidden_states.append(x_img)

            if isinstance(x_txt, (tuple, list)):
                x_txt = x_txt[0]

        # Vision pooling / projection
        x_img = x_img[:, 0, :]
        x_img = trunk.norm(x_img)
        image_features = self.vision_model.head(x_img)  # 768 -> 512

        # Text pooling / projection
        pooled_out = x_txt[:, 0, :]
        text_features = self.text_model.proj(pooled_out)  # 768 -> 512

        # KL regularization (average over adapted layers)
        kl_reg: Optional[torch.Tensor] = None
        if len(img_alpha_list) > 0:
            kl_img = torch.stack([gamma_kl_to_unit_prior(a) for a in img_alpha_list]).mean()
            kl_txt = torch.stack([gamma_kl_to_unit_prior(a) for a in txt_alpha_list]).mean()
            kl_reg = kl_img + kl_txt

        if self.output_hidden_states:
            return image_features, hidden_states, text_features, kl_reg
        else:
            return image_features, text_features, kl_reg

    def forward(self, image: torch.Tensor, text: list):
        tokenized_prompts = self.tokenizer(text).to(self.device)

        with torch.no_grad():
            prompts = self.text_model.transformer.embeddings.word_embeddings(tokenized_prompts).type(self.dtype)

        out = self.encode_text_image(tokenized_prompts, prompts, image)

        if self.output_hidden_states:
            image_features, hidden_states, text_features, kl_reg = out
        else:
            image_features, text_features, kl_reg = out

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        if self.output_hidden_states:
            return image_features, hidden_states, text_features, kl_reg
        else:
            return image_features, text_features, kl_reg


def build_evi_steer(cfg):
    print(f"Loading BiomedCLIP (backbone: {cfg.backbone})")
    clip_model, preprocess = load_biomedclip_to_device()
    clip_model.float()

    print("Building custom BiomedCLIP")
    model = CustomCLIP(cfg, clip_model)

    print("Turning off gradients in both the image and the text encoder")
    for name, param in model.named_parameters():
        if "evi_steer_adapters" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    return model, preprocess