import base64
import gzip
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
import torch.nn.init as init
from .decoding import decode as decode_function
from .decoding import detect_language as detect_language_function
from .transcribe import transcribe as transcribe_function


@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        #modify here qk to w
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk

class MultiHeadAttentionPE(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)
        self.query_cs = Linear(n_state, n_state)
        self.key_cs = Linear(n_state, n_state, bias=False)
        self.gate = nn.Parameter(torch.Tensor(12))  # Create a trainable parameter for the gate
        nn.init.uniform_(self.gate, 0, 1)
        # self.gate = Linear(n_state,1)  # Create a trainable parameter for the gate
        # nn.init.xavier_uniform_(self.query_cs.weight)
        # nn.init.constant_(self.query_cs.bias, 0)  # Optional: initialize bias with zeros

        # Initialize key_cs with Xavier initialization
        # nn.init.xavier_uniform_(self.key_cs.weight)
        # self.gate = nn.Sigmoid()

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)
        q_cs = self.query_cs(x)
        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            k_cs = self.key_cs(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]
        # sigmoid_gate = torch.sigmoid(self.gate(x))
        # sigmoid_gate = sigmoid_gate.unsqueeze(1).repeat(1, self.n_head, 1, 1)
        sigmoid_gate = torch.sigmoid(self.gate)
        # sigmoid_gate = sigmoid_gate.unsqueeze(1).repeat(1, self.n_head)
        wv, qk = self.qkv_attention(q, k, v, q_cs, k_cs, sigmoid_gate, mask)
        return self.out(wv), qk
    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, q_cs: Tensor, k_cs:Tensor, sigmoid_gate:Tensor, mask: Optional[Tensor] = None
    ):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        q_cs = q_cs.view(*q_cs.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k_cs = k_cs.view(*k_cs.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        
        qk = q @ k
        qk_cs = q_cs @ k_cs 
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
            qk_cs = qk_cs + mask[:n_ctx, :n_ctx]
            # sigmoid_gate = sigmoid_gate + mask[:n_ctx, :n_ctx]
        qk = qk.float()
        qk_cs = qk_cs.float()
        # make the shape to be the same as qk
        sigmoid_gate = sigmoid_gate.view(1, self.n_head, 1, 1)
        qk_combined = (1 - sigmoid_gate) *  qk + sigmoid_gate * qk_cs
        w = F.softmax(qk_combined, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), w

class Adapter(nn.Module):
    def __init__(self, idim, bottleneck_dim=None) -> None:
        super().__init__()
        bottleneck_dim = bottleneck_dim if bottleneck_dim else int(idim//4)
        self.model = nn.Sequential(
             #nn.Linear(10, 10),
             nn.Linear(idim, bottleneck_dim),
             nn.GELU(),
             nn.Linear(bottleneck_dim, idim),
        )

    def forward(self,input):
        output = self.model(input)
        return input + output
class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, adapter: bool = False, pe_whisper: bool = False, cross_attention: bool = False):
        super().__init__()
        self.adapter_flag = adapter
        if pe_whisper:
            self.attn = MultiHeadAttentionPE(n_state, n_head)
        else:
            self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)
        if self.adapter_flag:
            self.adapter_attn = Adapter(n_state)
            self.adapter_attn_ln = LayerNorm(n_state) 
        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        # if cross_attention and self.adapter_flag:
        #     self.adapter_cross_attn = Adapter(n_state)
        #     self.adapter_cross_attn_ln = LayerNorm(n_state) 
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)
        if self.adapter_flag:
            self.adapter_mlp = Adapter(n_state)
            self.adapter_mlp_ln = LayerNorm(n_state) 
            

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        attn_output = self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)
        x = x + attn_output[0]
        if self.adapter_flag:
            x = self.adapter_attn(x)
            x = self.adapter_attn_ln(x)
            # x = x + self.adapter_attn_ln(self.adapter_attn(x))
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
            # if self.adapter_flag:
            #     x = self.adapter_cross_attn(x)
            #     x = self.adapter_cross_attn_ln(x)
        x = x + self.mlp(self.mlp_ln(x))
        if self.adapter_flag:
            x = self.adapter_mlp(x)
            x = self.adapter_mlp_ln(x)
            # x = x + self.adapter_mlp_ln(self.adapter_mlp(x))
        return x,attn_output[1]


class AudioEncoder(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int, adapter: bool=False, pe_whisper: bool=False
    ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))
        if pe_whisper:
            self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
                [ResidualAttentionBlock(n_state, n_head, pe_whisper=True) for _ in range(n_layer)]
            )
        elif adapter:
            self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
                [ResidualAttentionBlock(n_state, n_head, adapter=True) for _ in range(n_layer)]
            )
        else:    
            self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
                [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
            )
        self.ln_post = LayerNorm(n_state)
        self.n_layer = n_layer

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)

        for block in self.blocks:
            x,_ = block(x)

        x = self.ln_post(x)
        return x


class TextDecoder(nn.Module):
    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int, pe_whisper: bool = False, adapter: bool = False
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        if pe_whisper:
            self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
                [
                    ResidualAttentionBlock(n_state, n_head, pe_whisper=True ,cross_attention=True)
                    for _ in range(n_layer)
                ]
            )
        elif adapter:
            self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
                [ResidualAttentionBlock(n_state, n_head, adapter=True, cross_attention=True) for _ in range(n_layer)]
            )
        else:
            self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
                [
                    ResidualAttentionBlock(n_state, n_head, cross_attention=True)
                    for _ in range(n_layer)
                ]
            )
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)
        self.n_layer = n_layer
    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
            the encoded audio features to be attended on
        """
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = (
            self.token_embedding(x)
            + self.positional_embedding[offset : offset + x.shape[-1]]
        )
        x = x.to(xa.dtype)

        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

        x = self.ln(x)
        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()

        return logits

class AudioEncoderSideNetwork(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int, input_dim: int, output_dim :int
    ):
        super().__init__()
        # self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        # self.downsample_conv_1 = Linear(input_dim,n_state)
        # self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        # self.downsample_conv_2 = Linear(input_dim,n_state)
        # self.sigmoid_gate_conv = nn.ParameterList(
        #     [nn.Parameter(torch.Tensor(1)) for _ in range(2)]
        # )
        # self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))
        self.downsample_input = Linear(input_dim,n_state)
        self.downsample_intermediate_layers = nn.ModuleList(
            [Linear(input_dim, n_state) for _ in range(n_layer)]
        )
        # self.sigmoid_gate_intermediate_layers = nn.ModuleList(
        #     [Linear(n_state, 1) for _ in range(n_layer)]
        # )
        self.sigmoid_gate_intermediate_layers = nn.ParameterList(
            [nn.Parameter(torch.Tensor(1)) for _ in range(n_layer)]
        )
        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        # self.blocks: Iterable[Adapter] = nn.ModuleList(
        #     [Adapter(n_state,n_state//2) for _ in range(n_layer)]
        # )
        # self.ln_post = LayerNorm(n_state)
        self.upsample_output = Linear(n_state,output_dim)
        self.ln_post = LayerNorm(output_dim)
        self.sigmoid_gate_output = nn.Parameter(torch.Tensor(1))
        
        self.n_layer = n_layer
        # for layer in self.sigmoid_gate_conv:
        #     nn.init.uniform_(layer, 0, 0)
        # Initialize sigmoid_gate_intermediate_layers with Xavier initialization
        for layer in self.sigmoid_gate_intermediate_layers:
            # init.xavier_uniform_(layer.weight)
            # init.constant_(layer.bias, 0.0)
            nn.init.uniform_(layer, -1, 1)
        nn.init.uniform_(self.sigmoid_gate_output, -1, 1)
        
        # for layer in self.downsample_intermediate_layers:
        #     init.xavier_uniform_(layer.weight)
        #     init.constant_(layer.bias, 0.0)
        # init.xavier_uniform_(self.downsample_input.weight)
        # init.constant_(self.downsample_input.bias, 0.0)
        # init.xavier_uniform_(self.upsample_output.weight)
        # init.constant_(self.upsample_output.bias, 0.0)
        
    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)
        x_downsampled=self.downsample_input(x)
        for i in range(self.n_layer):
            x_downsampled,_ = self.blocks[i](x_downsampled)
            # x_downsampled = x_outputfromoriginalmodel + self.sigmoid_gate_intermediate_layers[i](x_downsampled) * x_downsampled
        x_downsampled = self.ln_post(x_downsampled)
        # x_upsampled = self.upsample_output(x_downsampled)
        return x_downsampled
class TextDecoderSideNetwork(nn.Module):
    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int, input_dim: int, output_dim :int
    ):
        super().__init__()

        self.downsample_input = Linear(input_dim,n_state)
        self.downsample_encoder_input = Linear(input_dim,n_state)
        self.downsample_intermediate_layers = nn.ModuleList(
            [Linear(input_dim, n_state) for _ in range(n_layer)]
        )
        # self.sigmoid_gate_intermediate_layers = nn.ModuleList(
        #     [Linear(n_state, 1) for _ in range(n_layer)]
        # )
        self.sigmoid_gate_intermediate_layers = nn.ParameterList(
            [nn.Parameter(torch.Tensor(1)) for _ in range(n_layer)]
        )
        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
                [ResidualAttentionBlock(n_state, n_head, cross_attention=True) for _ in range(n_layer)]
            )
        # self.blocks: Iterable[Adapter] = nn.ModuleList(
        #     [Adapter(n_state,n_state//2) for _ in range(n_layer)]
        # )
        self.upsample_output = Linear(n_state,output_dim)
        self.ln = LayerNorm(output_dim)
        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)
        #  # Initialize sigmoid_gate_intermediate_layers with Xavier initialization
        for layer in self.sigmoid_gate_intermediate_layers:
            # init.xavier_uniform_(layer.weight)
            # init.constant_(layer.bias, 0.0)
            nn.init.uniform_(layer, -1, 1)
        # for layer in self.downsample_intermediate_layers:
        #     init.xavier_uniform_(layer.weight)
        #     init.constant_(layer.bias, 0.0)
        # init.xavier_uniform_(self.downsample_input.weight)
        # init.constant_(self.downsample_input.bias, 0.0)
        # # init.xavier_uniform_(self.downsample_encoder_input.weight)
        # # init.constant_(self.downsample_encoder_input.bias, 0.0)
        # init.xavier_uniform_(self.upsample_output.weight)
        # init.constant_(self.upsample_output.bias, 0.0)
    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
            the encoded audio features to be attended on
        """
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = (
            self.token_embedding(x)
            + self.positional_embedding[offset : offset + x.shape[-1]]
        )
        x = x.to(xa.dtype)
        x_downsampled=self.downsample_input(x)
        # xa_downsampled=self.downsample_encoder_input(xa_downsampled)
        for i in range(self.n_layer):
            x_downsampled = self.blocks[i](x_downsampled, xa, mask=self.mask, kv_cache=kv_cache)
            # x_downsampled = x_outputfromoriginalmodel + self.sigmoid_gate_intermediate_layers[i](x_downsampled) * x_downsampled
        x_downsampled = self.ln(x_downsampled)
        x_upsampled = self.upsample_output(x_downsampled)
        logits = (
            x_upsampled @ torch.transpose(self.token_embedding.weight.to(x_upsampled.dtype), 0, 1)
        ).float()

        return logits
class Whisper(nn.Module):
    def __init__(self, dims: ModelDimensions, pe_whisper:bool=False, adapter:bool=False, side_network:bool=False,side_network_conf:dict=None):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
            pe_whisper = pe_whisper,
            adapter = adapter
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
            pe_whisper=pe_whisper,
            adapter = adapter
        )
        if(side_network):
            self.encoder_sidenetwork = AudioEncoderSideNetwork(
                self.dims.n_mels,
                self.dims.n_audio_ctx,
                side_network_conf['n_dim'],
                side_network_conf['n_head'],
                len(side_network_conf['layers']),
                self.dims.n_audio_state,
                self.dims.n_audio_state,
            )
            
            self.decoder_sidenetwork = TextDecoderSideNetwork(
                self.dims.n_vocab,
                self.dims.n_text_ctx,
                side_network_conf['n_dim'],
                side_network_conf['n_head'],
                len(side_network_conf['layers']),
                self.dims.n_text_state,
                self.dims.n_text_state,
            )
        # use the last half layers for alignment by default; see `set_alignment_heads()` below
        all_heads = torch.zeros(
            self.dims.n_text_layer, self.dims.n_text_head, dtype=torch.bool
        )
        all_heads[self.dims.n_text_layer // 2 :] = True
        self.register_buffer("alignment_heads", all_heads.to_sparse(), persistent=False)

    def set_alignment_heads(self, dump: bytes):
        array = np.frombuffer(
            gzip.decompress(base64.b85decode(dump)), dtype=bool
        ).copy()
        mask = torch.from_numpy(array).reshape(
            self.dims.n_text_layer, self.dims.n_text_head
        )
        self.register_buffer("alignment_heads", mask.to_sparse(), persistent=False)

    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder(tokens, audio_features)

    def forward(
        self, mel: torch.Tensor, tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return self.decoder(tokens, self.encoder(mel))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab == 51865

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Module, torch.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.dims.n_text_ctx:
                # save as-is, for the first token or cross attention
                cache[module] = output
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = decode_function
