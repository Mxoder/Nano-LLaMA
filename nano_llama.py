import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from typing import Optional, Tuple
from torch.nn import CrossEntropyLoss

from nano_tokenizer import NanoTokenizer
from nano_llama_config import NanoLlamaConfig

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class NanoRMSNorm(nn.Module):
    """
    均方根层归一化 (Root Mean Square Layer Normalization)。

    参数:
        dimension (int): 输入张量的维度。
        epsilon (float, 可选): 一个小值，用于避免除零错误。默认值为 1e-6。
    """
    def __init__(self, dimension: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dimension))

    def _compute_norm(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        计算均方根归一化。

        参数:
            tensor (torch.Tensor): 需要归一化的输入张量。

        返回:
            torch.Tensor: 归一化后的张量。
        """
        norm = torch.rsqrt(tensor.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return tensor * norm

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        RMSNorm 的前向传播。

        参数:
            input_tensor (torch.Tensor): 输入到该层的张量。

        返回:
            torch.Tensor: 归一化并缩放后的张量。
        """
        normalized_tensor = self._compute_norm(input_tensor.float()).type_as(input_tensor)
        return normalized_tensor * self.weight

def precompute_freqs_cis(dimension: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    预计算频率的复数表示。

    参数:
        dimension (int): 频率的维度。
        end (int): 时间步的结束值。
        theta (float, 可选): 频率缩放因子，默认值为 10000.0。

    返回:
        torch.Tensor: 预计算的频率复数表示张量。
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dimension, 2)[: (dimension // 2)].float() / dimension))
    time_steps = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(time_steps, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    freqs_cis = freqs_cis.to(device)
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    调整频率张量以便广播。

    参数:
        freqs_cis (torch.Tensor): 频率的复数表示张量。
        x (torch.Tensor): 输入张量。

    返回:
        torch.Tensor: 调整后的频率复数表示张量。
    """
    num_dimensions = x.ndim
    assert 0 <= 1 < num_dimensions, "输入张量的维度应大于1。"
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), "频率张量的形状应与输入张量的形状匹配。"
    new_shape = [dim if i == 1 or i == num_dimensions - 1 else 1 for i, dim in enumerate(x.shape)]
    return freqs_cis.view(new_shape)

def apply_rotary_emb(
    query_tensor: torch.Tensor,
    key_tensor: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    应用旋转嵌入到查询和键张量。

    参数:
        query_tensor (torch.Tensor): 查询张量。
        key_tensor (torch.Tensor): 键张量。
        freqs_cis (torch.Tensor): 频率的复数表示张量。

    返回:
        Tuple[torch.Tensor, torch.Tensor]: 经过旋转嵌入处理后的查询和键张量。
    """
    # 将查询向量转换为复数形式。首先将张量转换为浮点数，然后调整其形状以保证最后一个维度为2（代表复数的实部和虚部），
    # 最后使用torch.view_as_complex将这些实虚部对视为复数。
    query_complex = torch.view_as_complex(query_tensor.float().reshape(*query_tensor.shape[:-1], -1, 2))
    
    # 同样的方式处理键向量
    key_complex = torch.view_as_complex(key_tensor.float().reshape(*key_tensor.shape[:-1], -1, 2))
    
    # 调用reshape_for_broadcast函数调整freqs_cis的形状，使其可以广播到query_complex的形状
    freqs_cis = reshape_for_broadcast(freqs_cis, query_complex)
    
    # 对查询向量和键向量应用旋转嵌入，即通过复数相乘来实现旋转。
    # 结果使用torch.view_as_real转换回实数张量（将复数的实部和虚部分开），并将最后两维合并起来。
    query_out = torch.view_as_real(query_complex * freqs_cis).flatten(3)
    key_out = torch.view_as_real(key_complex * freqs_cis).flatten(3)
    
    # 返回的张量类型与输入的张量类型相同
    return query_out.type_as(query_tensor), key_out.type_as(key_tensor)

class NanoLlamaAttention(nn.Module):
    """
    NanoLlama 注意力机制类。

    参数:
        cfg (NanoLlamaConfig): NanoLlama 配置实例。
    """
    def __init__(self, cfg: NanoLlamaConfig):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.num_attention_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.num_attention_heads * self.head_dim, cfg.hidden_size, bias=False)

        self.dropout = cfg.dropout

    def forward(
        self,
        input_tensor: torch.Tensor,
        freqs_cis: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播方法。

        参数:
            input_tensor (torch.Tensor): 输入张量，形状为 (bsz, seq_len, hidden_size)。
            freqs_cis (torch.Tensor): 频率的复数表示张量。
            padding_mask (Optional[torch.Tensor]): 用于屏蔽pad位置的掩码，形状为 (bsz, seq_len)。

        返回:
            torch.Tensor: 输出张量。
        """
        bsz, seq_len, _ = input_tensor.shape

        # QKV 投影
        query = self.q_proj(input_tensor)
        key = self.k_proj(input_tensor)
        value = self.v_proj(input_tensor)

        # 重排张量形状
        query = rearrange(query, 'b s (h d) -> b s h d', h=self.num_attention_heads)
        key = rearrange(key, 'b s (h d) -> b s h d', h=self.num_attention_heads)
        value = rearrange(value, 'b s (h d) -> b s h d', h=self.num_attention_heads)

        # 应用旋转嵌入 (RoPE) 相对位置嵌入
        query, key = apply_rotary_emb(query, key, freqs_cis)

        # 将 heads 转换为 batch 维度
        query = rearrange(query, 'b s h d -> b h s d')
        key = rearrange(key, 'b s h d -> b h s d')
        value = rearrange(value, 'b s h d -> b h s d')

        scores = torch.matmul(query, rearrange(key, 'b h s d -> b h d s')) / math.sqrt(self.head_dim)

        # 应用padding_mask
        if padding_mask is not None:
            pad_mask = torch.full((bsz, seq_len), -10000.0).to(device)
            pad_mask.masked_fill_(padding_mask.to(torch.bool), 0.0)

            scores = scores + rearrange(pad_mask, "b s -> b 1 1 s")

        scores = scores + self.casual_mask[:, :, :seq_len, :seq_len]
        scores = F.softmax(scores.float(), dim=-1).type_as(query)
        attention_output = torch.matmul(scores, value)  # (bs, n_local_heads, seqlen, head_dim)

        # 恢复时间维度为 batch 维度并连接 heads
        attention_output = rearrange(attention_output, 'b h s d -> b s (h d)')

        # 最终投影到残差流中
        output = self.o_proj(attention_output)

        return output

class NanoLlamaMLP(nn.Module):
    """
    NanoLlama FFN 层。

    参数:
        cfg (NanoLlamaConfig): NanoLlama 配置实例。
    """
    def __init__(self, cfg: NanoLlamaConfig):
        super().__init__()
        self.config = cfg
        self.hidden_size = cfg.hidden_size
        self.intermediate_size = cfg.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=cfg.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=cfg.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=cfg.mlp_bias)
        self.act_fn = nn.SiLU()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        前向传播方法。

        参数:
            input_tensor (torch.Tensor): 输入张量，形状为 (bsz, seq_len, hidden_size)。

        返回:
            torch.Tensor: 输出张量。
        """
        gate_output = self.gate_proj(input_tensor)
        up_output = self.up_proj(input_tensor)
        activated_output = self.act_fn(gate_output) * up_output
        down_output = self.down_proj(activated_output)
        return down_output

class NanoLlamaDecoderLayer(nn.Module):
    """
    NanoLlama 解码层类。

    参数:
        cfg (NanoLlamaConfig): NanoLlama 配置实例。
        layer_idx (int): 层索引。
    """
    def __init__(self, cfg: NanoLlamaConfig, layer_idx: int):
        super().__init__()
        self.num_attention_heads = cfg.num_attention_heads
        self.hidden_size = cfg.hidden_size
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads

        self.attention = NanoLlamaAttention(cfg)
        self.mlp = NanoLlamaMLP(cfg)
        self.layer_idx = layer_idx

        self.input_layernorm = NanoRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = NanoRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(
        self,
        input_tensor: torch.Tensor,
        freqs_cis: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        前向传播方法。

        参数:
            input_tensor (torch.Tensor): 输入张量，形状为 (bsz, seq_len, hidden_size)。
            freqs_cis (torch.Tensor): 频率张量（包含正余弦）。
            padding_mask (Optional[torch.Tensor]): 填充掩码，形状为 (bsz, seq_len)

        返回:
            torch.Tensor: 输出张量。
        """
        # 输入层归一化
        normed_input = self.input_layernorm(input_tensor)
        # 注意力层前向传播
        attention_output = self.attention(normed_input, freqs_cis, padding_mask)
        # 残差连接
        residual_connection1 = input_tensor + attention_output

        # 后注意力层归一化
        normed_residual = self.post_attention_layernorm(residual_connection1)
        # MLP层前向传播
        mlp_output = self.mlp(normed_residual)
        # 残差连接
        output = residual_connection1 + mlp_output

        return output

class NanoLlamaForCasualLM(nn.Module):
    """
    NanoLlamaForCasualLM 模型类。

    参数:
        cfg (NanoLlamaConfig): NanoLlama 配置实例。
    """
    def __init__(self, cfg: NanoLlamaConfig):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [NanoLlamaDecoderLayer(cfg, layer_idx) for layer_idx in range(cfg.num_hidden_layers)]
        )
        self.norm = NanoRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self._init_weights()

        self.freqs_cis = precompute_freqs_cis(
            cfg.hidden_size // cfg.num_attention_heads,
            cfg.max_position_embeddings * 2,
            cfg.rope_theta,
        )
        self.freqs_cis.to(self.device)

    def forward(
        self, 
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        前向传播方法。

        参数:
            input_ids (torch.LongTensor): 输入张量，形状为 (bsz, seq_len)。
            attention_mask (Optional[torch.LongTensor]): 注意力掩码（for padding），形状为 (bsz, seq_len)
            labels (Optional[torch.LongTensor]): 标签张量，形状为 (bsz, seq_len)。

        返回:
            Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]: 损失、logits 和输出张量。
        """
        bsz, seq_len = input_ids.shape
        embedded_input = self.embed_tokens(input_ids)
        freqs_cis = self.freqs_cis[:seq_len]

        hidden_states = embedded_input
        for layer in self.layers:
            hidden_states = layer(hidden_states, freqs_cis, padding_mask=attention_mask)
        outputs = self.norm(hidden_states)
        
        logits = self.lm_head(outputs)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.cfg.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        
        return loss, logits, outputs

    def generate(
        self,
        tokenizer: NanoTokenizer,       # tokenizer
        test_text: str,                 # 输入文本
        max_new_tokens: int = 128,      # 生成的新token的最大数量
        do_sample: bool = False,        # 是否进行采样
        top_k: int = 40,                # top-k采样的k值
        top_p: float = 0.85,            # top-p采样的p值
        temperature: float = 0.7,       # 温度系数
    ):
        def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
            """
            对概率分布执行top-p（核）采样。

            参数:
                probs (torch.Tensor): 概率分布张量。
                p (float): top-p采样的概率阈值。

            返回:
                torch.Tensor: 采样的token索引。

            注意:
                Top-p采样选择累积概率质量超过阈值p的最小token集合。
                分布将基于所选token重新归一化。
            """
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)  # 对概率分布进行排序
            probs_sum = torch.cumsum(probs_sort, dim=-1)  # 计算累积概率
            mask = probs_sum - probs_sort > p  # 创建掩码，标记超过阈值的部分
            probs_sort[mask] = 0.0  # 将超过阈值的部分置为0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))  # 重新归一化概率分布
            next_token = torch.multinomial(probs_sort, num_samples=1)  # 从归一化后的分布中采样
            next_token = torch.gather(probs_idx, -1, next_token)  # 获取采样的token索引
            return next_token  # 返回采样的token索引
        
        input_ids = tokenizer([test_text], return_tensors='pt')['input_ids'].to(device)  # 将输入文本转换为张量并移动到设备上
        output_ids = input_ids  # 初始化输出张量
        for i in range(max_new_tokens):  # 循环生成新token
            loss, logits, outputs = self(output_ids)  # 获取模型的输出
            if not do_sample:  # 如果不进行采样
                next_token = torch.argmax(logits[:, -1], dim=-1)  # 选择概率最大的token
            else:  # 如果进行采样
                if temperature > 0:  # 如果温度系数大于0
                    probs = torch.softmax(logits[:, -1] / temperature, dim=-1)  # 计算softmax概率
                    next_token = sample_top_p(probs, top_p)  # 进行top-p采样
                else:
                    next_token = torch.argmax(logits[:, -1], dim=-1)  # 选择概率最大的token

            next_token = next_token.reshape(-1)  # 将next_token转换为一维张量
            next_token = next_token.unsqueeze(0)  # 将next_token扩展为二维张量
            output_ids = torch.cat((output_ids, next_token), dim=1)  # 将next_token拼接到output_ids

        decoded = tokenizer.decode(output_ids[0])  # 解码生成的token序列
        return decoded  # 返回生成的文本

    def _init_weights(self):
        """
        初始化模型权重。
        """
        num_layers = len(self.layers)
        for name, param in self.named_parameters():
            if name.endswith('o_proj.weight'):
                nn.init.normal_(param, mean=0.0, std=0.02 / (2 * num_layers) ** 0.5)
            elif any(s in name for s in ["proj", "embed", "lm_head"]):
                nn.init.normal_(param, mean=0.0, std=0.02)

    def print_model_parameters(model, verbose=True):
        """
        打印模型的每一层及其参数大小
        """
        print("Layer Name & Parameters")
        print("----------------------------")
        total_params = 0
        for name, parameter in model.named_parameters():
            param_size = parameter.size()
            param_count = torch.prod(torch.tensor(param_size)).item()
            total_params += param_count
            if verbose:
                print(f"{name:50} | Size: {str(param_size):30} | Count: {str(param_count):20}")
        if verbose:
            print("----------------------------")
        print(f"Total Parameters: {total_params} ({total_params / 1000000:.1f} M)")
