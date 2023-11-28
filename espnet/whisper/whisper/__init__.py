import hashlib
import io
import os
import urllib
import warnings
from typing import List, Optional, Union

import torch
from tqdm import tqdm
import random
from .audio import load_audio, log_mel_spectrogram, pad_or_trim
from .decoding import DecodingOptions, DecodingResult, decode, detect_language
from .model import ModelDimensions, Whisper
from .transcribe import transcribe
from .version import __version__

_MODELS = {
    "tiny.en": "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt",
    "tiny": "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
    "base.en": "https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt",
    "base": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
    "small.en": "https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt",
    "small": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
    "medium.en": "https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt",
    "medium": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
    "large-v1": "https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt",
    "large-v2": "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
    "large": "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
}

# base85-encoded (n_layers, n_heads) boolean arrays indicating the cross-attention heads that are
# highly correlated to the word-level timing, i.e. the alignment between audio and text tokens.
_ALIGNMENT_HEADS = {
    "tiny.en": b"ABzY8J1N>@0{>%R00Bk>$p{7v037`oCl~+#00",
    "tiny": b"ABzY8bu8Lr0{>%RKn9Fp%m@SkK7Kt=7ytkO",
    "base.en": b"ABzY8;40c<0{>%RzzG;p*o+Vo09|#PsxSZm00",
    "base": b"ABzY8KQ!870{>%RzyTQH3`Q^yNP!>##QT-<FaQ7m",
    "small.en": b"ABzY8>?_)10{>%RpeA61k&I|OI3I$65C{;;pbCHh0B{qLQ;+}v00",
    "small": b"ABzY8DmU6=0{>%Rpa?J`kvJ6qF(V^F86#Xh7JUGMK}P<N0000",
    "medium.en": b"ABzY8usPae0{>%R7<zz_OvQ{)4kMa0BMw6u5rT}kRKX;$NfYBv00*Hl@qhsU00",
    "medium": b"ABzY8B0Jh+0{>%R7}kK1fFL7w6%<-Pf*t^=N)Qr&0RR9",
    "large-v1": b"ABzY8r9j$a0{>%R7#4sLmoOs{s)o3~84-RPdcFk!JR<kSfC2yj",
    "large-v2": b"ABzY8zd+h!0{>%R7=D0pU<_bnWW*tkYAhobTNnu$jnkEkXqp)j;w1Tzk)UH3X%SZd&fFZ2fC2yj",
    "large": b"ABzY8zd+h!0{>%R7=D0pU<_bnWW*tkYAhobTNnu$jnkEkXqp)j;w1Tzk)UH3X%SZd&fFZ2fC2yj",
}


def _download(url: str, root: str, in_memory: bool) -> Union[bytes, str]:
    os.makedirs(root, exist_ok=True)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, os.path.basename(url))

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        with open(download_target, "rb") as f:
            model_bytes = f.read()
        if hashlib.sha256(model_bytes).hexdigest() == expected_sha256:
            return model_bytes if in_memory else download_target
        else:
            warnings.warn(
                f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file"
            )

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(
            total=int(source.info().get("Content-Length")),
            ncols=80,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    model_bytes = open(download_target, "rb").read()
    if hashlib.sha256(model_bytes).hexdigest() != expected_sha256:
        raise RuntimeError(
            "Model has been downloaded but the SHA256 checksum does not not match. Please retry loading the model."
        )

    return model_bytes if in_memory else download_target


def available_models() -> List[str]:
    """Returns the names of available models"""
    return list(_MODELS.keys())

def project_weights(original_weight, side_n_dims):
    # Calculate the projection factor for the weight
    projection_factor = original_weight.shape[-1] / side_n_dims

    # Perform weight projection
    projected_weight = original_weight.view(-1, side_n_dims).mean(dim=0)
    projected_weight = projected_weight.view(1, -1)
    projected_weight /= projection_factor

    return projected_weight

def attention_copy_weights_and_biases(original_weights, original_biases, new_dims, new_heads, num_heads=12):
    original_weights_dim = original_weights.shape[0]
    head_size = original_weights_dim // num_heads
    original_weights_reshaped = original_weights.view(num_heads, head_size, -1)
    
    # Choose random head indices
    random_heads = random.sample(range(num_heads), new_heads)
    
    # Get the weights and biases for the randomly chosen heads
    selected_weights = torch.cat([original_weights_reshaped[head_idx, :, torch.randperm(new_dims)[:new_dims]] for head_idx in random_heads])
    # selected_weights = torch.cat([original_weights_reshaped[head_idx, :, :] for head_idx in random_heads])
    
    selected_biases = None
    if original_biases is not None:
        selected_biases = torch.cat([original_biases[head_idx * head_size: (head_idx + 1) * head_size] for head_idx in random_heads])
    
    return selected_weights, selected_biases
def mlp_copy_weights_and_biases(original_weights, original_biases, new_dims):
    original_in_features, original_out_features = original_weights.shape
    
    # Choose random rows and columns indices
    selected_rows = torch.randperm(original_in_features)[:new_dims[0]]
    selected_columns = torch.randperm(original_out_features)[:new_dims[1]]
    
    side_weights = original_weights[selected_rows][:, selected_columns]
    side_biases = original_biases[selected_rows]
    
    # side_weights = original_weights[:][:, :]
    # side_biases = original_biases[:]
    
    return side_weights, side_biases

def copy_weights_with_projection(original_block, side_block, side_n_dims, side_n_head, decoder:bool=False):
     # Copy weights and biases for the attention layers
    side_query_weights, side_query_biases = attention_copy_weights_and_biases(original_block.attn.query.weight, original_block.attn.query.bias, side_n_dims, side_n_head)
    side_key_weights, _ = attention_copy_weights_and_biases(original_block.attn.key.weight, None, side_n_dims, side_n_head)
    side_value_weights, side_value_biases = attention_copy_weights_and_biases(original_block.attn.value.weight, original_block.attn.value.bias, side_n_dims, side_n_head)
    side_out_weights, side_out_biases = attention_copy_weights_and_biases(original_block.attn.out.weight, original_block.attn.out.bias, side_n_dims, side_n_head)
    
    # Copy weights and biases for the side_block
    side_block.attn.query.weight.data.copy_(side_query_weights)
    side_block.attn.key.weight.data.copy_(side_key_weights)
    side_block.attn.value.weight.data.copy_(side_value_weights)
    side_block.attn.out.weight.data.copy_(side_out_weights)
    
    side_block.attn.query.bias.data.copy_(side_query_biases)
    side_block.attn.value.bias.data.copy_(side_value_biases)
    side_block.attn.out.bias.data.copy_(side_out_biases)
    
    if decoder:
        # Copy weights for the cross-attention layers
        side_cross_query_weights, side_cross_query_biases = attention_copy_weights_and_biases(original_block.cross_attn.query.weight, original_block.cross_attn.query.bias, side_n_dims, side_n_head)
        side_cross_key_weights, _ = attention_copy_weights_and_biases(original_block.cross_attn.key.weight, None, side_n_dims, side_n_head)
        side_cross_value_weights, side_cross_value_biases = attention_copy_weights_and_biases(original_block.cross_attn.value.weight, original_block.cross_attn.value.bias, side_n_dims, side_n_head)
        side_cross_out_weights, side_cross_out_biases = attention_copy_weights_and_biases(original_block.cross_attn.out.weight, original_block.cross_attn.out.bias, side_n_dims, side_n_head)
        
        # Copy weights and biases for the side_block cross-attention
        side_block.cross_attn.query.weight.data.copy_(side_cross_query_weights)
        side_block.cross_attn.key.weight.data.copy_(side_cross_key_weights)
        side_block.cross_attn.value.weight.data.copy_(side_cross_value_weights)
        side_block.cross_attn.out.weight.data.copy_(side_cross_out_weights)
        
        side_block.cross_attn.query.bias.data.copy_(side_cross_query_biases)
        side_block.cross_attn.value.bias.data.copy_(side_cross_value_biases)
        side_block.cross_attn.out.bias.data.copy_(side_cross_out_biases)
    # Copy weights and biases for the MLP layers
    side_mlp_0_weights, side_mlp_0_biases = mlp_copy_weights_and_biases(original_block.mlp[0].weight, original_block.mlp[0].bias, [4*side_n_dims, side_n_dims])
    side_mlp_2_weights, side_mlp_2_biases = mlp_copy_weights_and_biases(original_block.mlp[2].weight, original_block.mlp[2].bias, [side_n_dims, 4*side_n_dims])
    
    # Copy weights and biases for the side_block MLP
    side_block.mlp[0].weight.data.copy_(side_mlp_0_weights)
    side_block.mlp[2].weight.data.copy_(side_mlp_2_weights)
    side_block.mlp[0].bias.data.copy_(side_mlp_0_biases)
    side_block.mlp[2].bias.data.copy_(side_mlp_2_biases)

def load_model(
    name: str,
    adapter: bool = False,
    pe_whisper: bool = False,
    side_network: bool = False,
    side_network_conf : dict = None,
    device: Optional[Union[str, torch.device]] = None,
    download_root: str = None,
    in_memory: bool = False,
) -> Whisper:
    """
    Load a Whisper ASR model

    Parameters
    ----------
    name : str
        one of the official model names listed by `whisper.available_models()`, or
        path to a model checkpoint containing the model dimensions and the model state_dict.
    device : Union[str, torch.device]
        the PyTorch device to put the model into
    download_root: str
        path to download the model files; by default, it uses "~/.cache/whisper"
    in_memory: bool
        whether to preload the model weights into host memory

    Returns
    -------
    model : Whisper
        The Whisper ASR model instance
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if download_root is None:
        default = os.path.join(os.path.expanduser("~"), ".cache")
        download_root = os.path.join(os.getenv("XDG_CACHE_HOME", default), "whisper")

    if name in _MODELS:
        checkpoint_file = _download(_MODELS[name], download_root, in_memory)
        alignment_heads = _ALIGNMENT_HEADS[name]
    elif os.path.isfile(name):
        checkpoint_file = open(name, "rb").read() if in_memory else name
        alignment_heads = None
    else:
        raise RuntimeError(
            f"Model {name} not found; available models = {available_models()}"
        )

    with (
        io.BytesIO(checkpoint_file) if in_memory else open(checkpoint_file, "rb")
    ) as fp:
        checkpoint = torch.load(fp, map_location=device)
    del checkpoint_file

    dims = ModelDimensions(**checkpoint["dims"])
    model = Whisper(dims,pe_whisper,adapter,side_network,side_network_conf)
    if(pe_whisper):
        model.load_state_dict(checkpoint["model_state_dict"],strict=False)
        for block in model.encoder.blocks:
            block.attn.query_cs.weight.data.copy_(block.attn.query.weight)
            block.attn.query_cs.bias.data.copy_(block.attn.query.bias)
            block.attn.key_cs.weight.data.copy_(block.attn.key.weight)
        for block in model.decoder.blocks:
            block.attn.query_cs.weight.data.copy_(block.attn.query.weight)
            block.attn.query_cs.bias.data.copy_(block.attn.query.bias)
            block.attn.key_cs.weight.data.copy_(block.attn.key.weight)
    elif(adapter):
        model.load_state_dict(checkpoint["model_state_dict"],strict=False)
    elif(side_network):
        model.load_state_dict(checkpoint["model_state_dict"],strict=False)
        side_block=0
        # for layer in side_network_conf['layers']:
        #     copy_weights_with_projection(model.encoder.blocks[layer], model.encoder_sidenetwork.blocks[side_block], side_network_conf['n_dim'],  side_network_conf['n_head'])
        #     # copy_weights_with_projection(model.decoder.blocks[layer], model.decoder_sidenetwork.blocks[side_block], side_network_conf['n_dim'],  side_network_conf['n_head'],True)
        #     side_block+=1
            # print('check')
        # for i in range(model.decoder.n_layer):
        #     copy_weights_with_projection(model.decoder.blocks[i], model.decoder_sidenetwork.blocks[i], side_network_conf['n_dim'],True)
        #     # print('check')
            
    else:
        model.load_state_dict(checkpoint["model_state_dict"])

    if alignment_heads is not None:
        model.set_alignment_heads(alignment_heads)

    return model.to(device)
