import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from fairseq.modules import SamePad

KVCache = Tuple[torch.Tensor, torch.Tensor]

# @torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config, pos_emb=False):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.max_seq_len = config.block_size
        
        if config.context_width != -1:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            attn_mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
            # Local attention
            window_mask = torch.triu(torch.ones(config.block_size, config.block_size, dtype=torch.bool), 
                                        -(config.context_width - 1))
            attn_mask *= window_mask
            attn_mask = attn_mask.view(1, 1, config.block_size, config.block_size)
            self.register_buffer("bias", attn_mask)
        else:
            self.bias = None
        
        if pos_emb:
            d_head = config.n_embd // config.n_head
            # Positional encoding
            position = torch.arange(config.block_size).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_head, 2) * (-math.log(10000.0) / d_head))
            pe = torch.zeros(1, config.block_size, d_head)
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
            pe = pe.view(1, 1, config.block_size, d_head)
            self.register_buffer('pe', pe)
        else:
            self.pe = None

    def forward(self, x, 
                input_pos: Optional[torch.Tensor] = None, 
                kv_cache: Optional[KVCache] = None):
        assert kv_cache is None or (input_pos is not None and kv_cache is not None), "For kv caching input_pos must be provided"
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if self.pe is not None:
            if kv_cache is None:
                pe = self.pe[:, :, :T]
            else:
                pe = self.pe.index_select(2, input_pos)
            k = k + pe
            q = q + pe

        if kv_cache is not None:
            cache_k, cache_v = kv_cache # (b, n_head, block_size, head_size)
            kv_cache = cache_k.index_copy(2, input_pos, k), cache_v.index_copy(2, input_pos, v)
            T = input_pos[-1] + 1
            k = kv_cache[0][:, :, :T, :]
            v = kv_cache[1][:, :, :T, :]
            attn_mask = None if self.bias is None else self.bias.index_select(2, input_pos)[:,:,:,:T]
        else:
            attn_mask = None if self.bias is None else self.bias[:,:,:T,:T]
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, 
                                                                dropout_p=self.dropout if self.training else 0, 
                                                                is_causal=True if (self.bias is None and kv_cache is None) else False)
        y = y.transpose(1, 2).contiguous().view(B, -1, C) # re-assemble all head outputs side by side
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, kv_cache

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config, pos_emb=False):
        super().__init__()
        self.attention_only = config.attention_only
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config, pos_emb)
        if not self.attention_only:
            self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
            self.mlp = MLP(config)

    def forward(self, x, 
                input_pos: Optional[torch.Tensor] = None, 
                kv_cache: Optional[KVCache] = None):
        h, new_kv_cache = self.attn(self.ln_1(x), input_pos, kv_cache)
        x = x + h
        if not self.attention_only:
            x = x + self.mlp(self.ln_2(x))
        if new_kv_cache is not None:
            return x, new_kv_cache
        else:
            return x

@dataclass
class GPTConfig:
    attention_only: bool = True
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    context_width: int = -1

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if config.attention_only:
            layers = dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                drop = nn.Dropout(config.dropout)
            )
            if config.n_layer > 0:
                layers["h"] = nn.ModuleList([Block(config, pos_emb=True if l == 0 else False) for l in range(config.n_layer)])
            self.transformer = nn.ModuleDict(layers)
        else:
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = nn.Embedding(config.block_size, config.n_embd),
                drop = nn.Dropout(config.dropout),
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f = LayerNorm(config.n_embd, bias=config.bias),
            ))
            # with weight tying when using torch.compile() some warnings get generated:
            # "UserWarning: functional_call was passed multiple values for tied weights.
            # This behavior is deprecated and will be an error in future versions"
            # not 100% sure what this is, so far seems to be harmless. TODO investigate
            self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        self.kv_caches: List[KVCache] = []

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("LM number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            if self.config.attention_only:
                if self.config.n_layer > 0:
                    n_params -= self.transformer.h[0].attn.pe.numel()
            else:
                n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_probs, return_embeddings=False, return_logits=False, return_posteriograms=False, input_pos: Optional[torch.Tensor] = None):
        assert (
            (return_embeddings ^ return_logits ^ return_posteriograms) or
            (not return_embeddings and not return_logits and not return_posteriograms)
        ), "Choose only one of return_embeddings, return_logits, or return_entropy as LM output"
        device = token_probs.device
        b, t, vocab_size = token_probs.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        # forward the GPT model itself
        tok_emb = (token_probs.unsqueeze(2) @ self.transformer.wte.weight).squeeze(2) # b x t x n_embd
        if not self.config.attention_only: 
            pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)
            # token_probs has size b x t x vocab_size, wte.weight has size vocab_size x n_embd 
            pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)
        else:
            x = self.transformer.drop(tok_emb)
        if self.config.n_layer > 0:
            if input_pos is None:  # proxy for use_cache=False
                for block in self.transformer.h:
                    x = block(x)
            else:
                if not self.kv_caches:
                    head_size = self.config.n_embd // self.config.n_head
                    cache_shape = (b, self.config.n_head, self.config.block_size, head_size)
                    self.kv_caches = [
                        (torch.zeros(cache_shape, device=x.device, dtype=x.dtype), torch.zeros(cache_shape, device=x.device, dtype=x.dtype))
                        for _ in range(self.config.n_layer)
                    ]
                for i, block in enumerate(self.transformer.h):
                    x, self.kv_caches[i] = block(x, input_pos, self.kv_caches[i])
        if "ln_f" in self.transformer:
            x = self.transformer.ln_f(x)
        if return_embeddings:
            return x, None, None, None
        logits = self.lm_head(x)
        if return_logits:
            return x, logits, None, None
        probs = F.softmax(logits, dim=-1)
        if return_posteriograms:
            return x, logits, probs, None
        entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1, keepdim=True)
        return x, logits, probs, entropy
    
    def reset_cache(self) -> None:
        self.kv_caches.clear()

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, mask=None, min_len=None, eos_token=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        assert min_len is None or (eos_token is not None), "If min_len is set then eos_token must be specified"
        for i in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            idx_cond = F.one_hot(idx_cond, num_classes=self.config.vocab_size).float()
            # forward the model to get the logits for the index in the sequence
            _, logits, _, _ = self(idx_cond, return_logits=True)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # optionally forbid some tokens by adding a bias to the logits
            if mask is not None:
                logits += mask
            if i < min_len: # Forbid from making 
                logits[:, eos_token] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    @torch.no_grad()
    def fast_generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, mask=None, min_len=None, eos_token=None):    
        assert idx.size(1) + max_new_tokens <= self.config.block_size, "Generation of sequences larger than the context window is not supported"
        assert min_len is None or (eos_token is not None), "If min_len is set then eos_token must be specified"
        t = idx.size(1)
        input_pos = torch.arange(0, t, device=idx.device)
        for i in range(max_new_tokens):
            x = idx.index_select(1, input_pos)
            idx_cond = F.one_hot(x, num_classes=self.config.vocab_size).float()
            # forward the model to get the logits for the index in the sequence
            _, logits, _, _ = self(idx_cond, return_logits=True, input_pos=input_pos)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # optionally forbid some tokens by adding a bias to the logits
            if mask is not None:
                logits += mask
            if i < min_len: # Forbid from making 
                logits[:, eos_token] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            # advance index pointer
            input_pos = input_pos[-1:] + 1
        self.reset_cache()
        return idx

class ConvLM(nn.Module):
    def __init__(self, 
                 vocab_size,
                 lm_dim=384,
                 lm_kernel=8,
                 lm_causal=True,
                 lm_max_pool=False,
                 lm_weight_norm=False,
                 lm_spectral_norm=False,
                 lm_dilation= 1
                ):
        super().__init__()

        self.n_embd = n_embd = lm_dim
        kernel = lm_kernel
        dilation = lm_dilation
        self.max_pool = lm_max_pool

        if lm_causal:
            padding = kernel - 1
        else:
            padding = kernel // 2

        def make_conv(in_d, out_d, k, p=0, has_dilation=True):
            conv = nn.Conv1d(
                in_d,
                out_d,
                kernel_size=k,
                padding=p,
                dilation=dilation if has_dilation else 1,
                bias=False
            )
            if lm_spectral_norm:
                conv = nn.utils.spectral_norm(conv)
            elif lm_weight_norm:
                conv = nn.utils.weight_norm(conv)
            return conv
        
        self.pad = SamePad(kernel_size=kernel, causal=lm_causal)
        self.emb_net = nn.Embedding(vocab_size, n_embd)
        self.ln_1 = LayerNorm(n_embd, bias=True)
        self.conv1 = make_conv(n_embd, n_embd, kernel, padding, has_dilation=False)
        self.ln_2 = LayerNorm(n_embd, bias=True)
        self.conv2 = make_conv(n_embd, n_embd, kernel, padding, has_dilation=False)
        self.ln_3 = LayerNorm(n_embd, bias=True)
        self.conv3 = make_conv(n_embd, n_embd, kernel, padding, has_dilation=False)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.emb_net.weight = self.lm_head.weight

    def forward(self, token_probs, return_logits=False):
        x = (token_probs.unsqueeze(2) @ self.emb_net.weight).squeeze(2) # b x t x n_embd
        x = F.gelu(self.pad(self.conv1(self.ln_1(x).transpose(1, 2)))).transpose(1, 2)
        x = F.gelu(self.pad(self.conv2(self.ln_2(x).transpose(1, 2)))).transpose(1, 2)
        x = F.gelu(self.pad(self.conv3(self.ln_3(x).transpose(1, 2)))).transpose(1, 2)
        logits = self.lm_head(x)
        if return_logits:
            return logits
        probs = F.softmax(logits, dim=-1)
        return probs
        # return -torch.sum(probs * torch.log(probs), dim=-1, keepdim=True)