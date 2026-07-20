from einops import repeat, rearrange, einsum, reduce
import torch
import torch.nn as nn
import math
from cs336_basics.nn_utils import softmax
from collections import OrderedDict

class linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.device: torch.device | None = device
        self.dtype: torch.dtype | None = dtype

        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype = dtype).to(device))
        std = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean = 0, std = std, a = -3 * std, b = 3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weight, x, "... d_out d_in, ... d_in-> ... d_out")
    
class embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.num_embeddings: int = num_embeddings
        self.embedding_dim: int = embedding_dim
        self.device: torch.device | None = device
        self.dtype: torch.dtype | None = dtype

        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim).to(device))
        nn.init.trunc_normal_(self.weight, mean = 0, std = 1, a = -3, b = 3)
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: (batch, seq_length)
        output: (batch, seq_length, embedding_dim)
        """
        return self.weight[token_ids]
    
class rmsnorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model: int = d_model
        self.eps: float = eps
        self.device: torch.device | None = device
        self.dtype: torch.dtype | None = dtype

        self.weight = nn.Parameter(torch.ones(d_model).to(device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, ..., seq_length, d_model)
        output: same shape
        """
        in_type: torch.dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt((reduce(x**2, "... seq_len d_model -> ... seq_len ()", "sum") / self.d_model) + self.eps) # (batch, seq_length)
        result = (x / rms) * self.weight

        return result.to(in_type)

class positionwise_feedforward(nn.Module):
    def __init__(self, d_ff: int, d_model: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_ff: int = d_ff
        self.d_model: int = d_model
        self.device: torch.device | None = device
        self.dtype: torch.dtype | None = dtype

        self.w1 = linear(in_features=d_model, out_features=d_ff, device = device, dtype = dtype)
        self.w2 = linear(in_features=d_ff, out_features=d_model, device = device, dtype = dtype)
        self.w3 = linear(in_features=d_model, out_features=d_ff, device = device, dtype = dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_sizem, seq_len, d_model)
        output: same shape
        W1: (d_ff, d_model) 
        W3: (d_ff, d_model)
        W2: (d_model, d_ff)
        """
        term1 = self.w1(x)
        term1 = term1 * torch.sigmoid(term1) 
        term2 = self.w3(x)

        return self.w2(term1 * term2)


#Do not have to re-compute it for all batches and across layers
class ROPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        self.theta: float = theta
        self.d_k: int = d_k
        self.max_seq_len: int = max_seq_len
        self.device: torch.device | None = device

        max_token_positions = torch.arange(max_seq_len).to(device)
        d_range_half = torch.arange(1, self.d_k // 2 + 1).to(device)
        d_range = torch.arange(1, self.d_k + 1).to(device)
        R1 = repeat(torch.cos(max_token_positions.unsqueeze(-1) / self.theta**((2 * d_range_half - 2) / self.d_k)), "... max_seq_length d_k_half -> ... max_seq_length (d_k_half k)", k =2)
        R2 = repeat(torch.sin(max_token_positions.unsqueeze(-1) / self.theta**((2 * d_range_half - 2) / self.d_k)), "... max_seq_length d_k_half -> ... max_seq_length (d_k_half k)", k =2)

        self.register_buffer("R1", R1, persistent=False)
        self.register_buffer("R2", R2, persistent=False)
        self.register_buffer("d_range", d_range, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (batch_size, ..., seq_length, d_k)
        token_positions: (..., seq_length), not always [1, 2, ..., seq_length], since a sentence can be much longer than seq_length 
        """
        seq_len = x.size(-2)
        if token_positions is not None:
            token_positions = token_positions.to(x.device) #keep in same device 
            R1 = self.R1[token_positions]
            R2 = self.R2[token_positions]
        
        else:
            R1 = self.R1[:seq_len,:]
            R2 = self.R2[:seq_len,:]
        x_variant = rearrange((-1)**(self.d_range - 1) * x, "... seq_length (d_k_half k) -> ... seq_length d_k_half k", k = 2)[..., [1, 0]]
        x_variant = rearrange(x_variant, "... seq_length d_k_half k -> ... seq_length (d_k_half k)")
        return x * R1 + x_variant * R2

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    K, Q: (batch_size, ..., seq_len, d_k)
    V: (batch_size, ..., seq_len, d_v)
    mask: (seq_len, seq_len)
    """
    d_k = K.shape[-1]
    QK = einsum(Q, K, "batch_size ... seq_len_q d_k, batch_size ... seq_len_k d_k -> batch_size ... seq_len_q seq_len_k") / math.sqrt(d_k)
    QK[~mask] = -torch.inf 

    return einsum(softmax(QK, -1), V, "batch_size ... seq_len_q seq_len_k, batch_size ... seq_len_k d_v -> batch_size ... seq_len_q d_v")

class multihead_self_attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, pos_embedding: ROPE | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model: int = d_model
        self.num_heads: int = num_heads
        self.device: torch.device | None = device
        self.dtype: torch.dtype | None = dtype

        assert d_model % num_heads == 0

        """
        x: (batch_size, seq_len, d_model)
        Q: (batch_size, num_heads, seq_len_q, d_k)
        K: (batch_size, num_heads, seq_len_q, d_k)
        V: (batch_size, num_heads, seq_len_q, d_v)

        Wq: (d_model, d_model)
        Wk: (d_model, d_model)
        Wv: (d_model, d_model)
        
        Q = Wq x
        K = Wk x
        V = Wv x
        """
        self.q_proj = linear(in_features=d_model, out_features=d_model, device = device, dtype = dtype)
        self.k_proj = linear(in_features=d_model, out_features=d_model, device = device, dtype = dtype)
        self.v_proj = linear(in_features=d_model, out_features=d_model, device = device, dtype = dtype)
        self.output_proj = linear(in_features=d_model, out_features=d_model, device = device, dtype = dtype)

        self.rope: ROPE | None = pos_embedding   

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        Wq = rearrange(self.q_proj.weight, "(num_heads d_k) d_model -> num_heads d_k d_model", num_heads = self.num_heads)
        Wk = rearrange(self.k_proj.weight, "(num_heads d_k) d_model -> num_heads d_k d_model", num_heads = self.num_heads)
        Wv = rearrange(self.v_proj.weight, "(num_heads d_v) d_model -> num_heads d_v d_model", num_heads = self.num_heads)

        Q = einsum(x, Wq, "... seq_len d_model, num_h d_k d_model -> ... num_h seq_len d_k")
        K = einsum(x, Wk, "... seq_len d_model, num_h d_k d_model -> ... num_h seq_len d_k")
        V = einsum(x, Wv, "... seq_len d_model, num_h d_v d_model -> ... num_h seq_len d_v")

        seq_len = Q.shape[-2]
        if self.rope is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        mask = torch.triu(torch.ones(seq_len, seq_len, dtype = torch.bool), diagonal = 1).to(self.device)
        mask = repeat(mask, "l1 l2 -> b num_h l1 l2", b = x.shape[0], num_h = self.num_heads)
        total_QKV = rearrange(scaled_dot_product_attention(Q, K, V, ~mask), "batch_size num_h seq_len_q d_v -> batch_size seq_len_q (num_h d_v)")
        return einsum(total_QKV, self.output_proj.weight, "batch_size seq_len_q d_1, d_model d_1 -> batch_size seq_len_q d_model")

class transformer_block(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, pos_embedding: ROPE | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.pos_embedding = pos_embedding
        self.device = device
        self.dtype = dtype
        if self.pos_embedding is not None:
            self.attn = multihead_self_attention(d_model=d_model, num_heads=num_heads, pos_embedding=pos_embedding, device=device, dtype=dtype)
        else:
            self.attn = multihead_self_attention(d_model=d_model, num_heads=num_heads, device=device, dtype=dtype)
        self.ln1 = rmsnorm(d_model=d_model, device=device, dtype=dtype)
        self.ln2 = rmsnorm(d_model=d_model, device=device, dtype=dtype)
        self.ffn = positionwise_feedforward(d_ff=d_ff, d_model=d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        y1 = x + self.attn(self.ln1(x), token_positions = token_positions)
        return y1 + self.ffn(self.ln2(y1))
    
class transformer_lm(nn.Module):
    def __init__(self, vocab_size : int, d_model: int, num_heads: int, d_ff: int, context_length: int, num_layers: int, pos_embedding: ROPE | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.context_length = context_length
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype

        self.token_embeddings = embedding(num_embeddings = vocab_size, embedding_dim = d_model, device = device, dtype = dtype)
        layers = [(f"{i}", transformer_block(d_model=d_model, num_heads=num_heads, d_ff=d_ff, pos_embedding = pos_embedding, device=device, dtype=dtype)) for i in range(num_layers)]
        self.layers = nn.Sequential(OrderedDict(layers))
        self.ln_final = rmsnorm(d_model = d_model, device = device, dtype = dtype)
        self.lm_head = linear(in_features=d_model, out_features=vocab_size, device = device, dtype = dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.ln_final(self.layers(self.token_embeddings(x))))
        







