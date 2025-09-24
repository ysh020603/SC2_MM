# model.py

import torch
import torch.nn as nn
from transformers import Qwen2ForCausalLM, Qwen2Config
from typing import Optional, Tuple, Union

# class MLPProjector(nn.Module):
#     """A simple two-layer MLP to project the entity vector into the LLM's embedding space."""
#     def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
#         super().__init__()
#         self.layer1 = nn.Linear(input_dim, hidden_dim)
#         self.activation = nn.GELU()
#         self.layer2 = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x):
#         return self.layer2(self.activation(self.layer1(x)))


class MLPProjector(nn.Module):
    """
    方案二：增加网络深度的MLP (三层结构)。
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.activation1 = nn.GELU()
        self.layer2 = nn.Linear(hidden_dim, hidden_dim) # <-- 新增的中间层
        self.activation2 = nn.GELU()
        self.layer3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.activation1(self.layer1(x))
        x = self.activation2(self.layer2(x))
        x = self.layer3(x)
        return x

# ... (Qwen2ForSC2Fusion 类的代码保持不变)
# 注意：使用这个方案时，Qwen2ForSC2Fusion 的 __init__ 不需要修改 hidden_dim 参数
# 它会直接使用 embedding_dim*2 作为 hidden_dim 传入

class Qwen2ForSC2Fusion(Qwen2ForCausalLM):
    """
    A Qwen2 model that supports fusion with StarCraft entity vectors.
    """
    def __init__(self, config: Qwen2Config, entity_vector_dim: int):
        super().__init__(config)
        embedding_dim = config.hidden_size
        self.projector = MLPProjector(
            input_dim=entity_vector_dim,
            hidden_dim=embedding_dim*8,
            output_dim=embedding_dim
        )
        self.sc2_entity_token_id = -1

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        entity_vectors: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, torch.Tensor]:

        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("You have to specify either input_ids or inputs_embeds")
            
            inputs_embeds = self.model.embed_tokens(input_ids)

            if entity_vectors is not None and self.sc2_entity_token_id != -1:
                projected_entity_embeds = self.projector(entity_vectors)
                batch_indices, token_indices = (input_ids == self.sc2_entity_token_id).nonzero(as_tuple=True)
                if batch_indices.numel() > 0:
                    inputs_embeds[batch_indices, token_indices] = projected_entity_embeds.to(inputs_embeds.dtype)
        
        # --- MODIFIED: The final call to the parent method is updated ---
        # We must pass input_ids as None because we are providing our custom inputs_embeds.
        # This resolves the "specify exactly one" error.
        return super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            labels=labels,
            inputs_embeds=inputs_embeds,
            **kwargs
        )