import copy
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from typeguard import check_argument_types

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.specaug.specaug import SpecAug


class OpenAIWhisperEncoder(AbsEncoder):
    """Transformer-based Speech Encoder from OpenAI's Whisper Model:

    URL: https://github.com/openai/whisper
    """

    def __init__(
        self,
        input_size: int = 1,
        dropout_rate: float = 0.0,
        whisper_model: str = "small",
        download_dir: str = None,
        use_specaug: bool = False,
        specaug_conf: Union[dict, None] = None,
        do_pad_trim: bool = False, # should be false
        pe_whisper: bool = False,
        adapter: bool = False,
        side_network : bool = False,
        side_network_conf = None
    ):
        try:
            # from whisper import whisper
            import whisper
            from whisper.audio import HOP_LENGTH, N_FFT, N_MELS, N_SAMPLES
        except Exception as e:
            print("Error: whisper is not properly installed.")
            print(
                "Please install whisper with: cd ${MAIN_ROOT}/tools &&",
                "./installers/install_whisper.sh",
            )
            raise e

        assert check_argument_types()
        super().__init__()

        self.n_fft = N_FFT
        self.win_length = N_FFT
        self.hop_length = HOP_LENGTH
        self.n_mels = N_MELS

        self.mel_filters = whisper.audio.mel_filters

        # note that originally Whisper doesn't use dropouts
        self.dropout = torch.nn.Dropout(dropout_rate)

        assert whisper_model in whisper.available_models()
        _model = whisper.load_model(whisper_model, adapter, pe_whisper, side_network,side_network_conf, download_root=download_dir)
        self.sidenetwork = side_network
        self.encoders = copy.deepcopy(_model.encoder)
        if(self.sidenetwork):
            # for param in self.encoders.parameters():
            #     param.requires_grad = False
            self.encoders_sidenetwork = copy.deepcopy(_model.encoder_sidenetwork)
            self.encoders_sidenetwork.train()
            self.sidenetwork_layers = side_network_conf['layers']
            self.sidenetwork_outputsize=self.encoders_sidenetwork.ln_post.normalized_shape[-1]
        else:
            self.encoders.train()
        del _model

        if use_specaug:
            self.specaug = SpecAug(**specaug_conf)
        else:
            self.specaug = None

        self.do_pad_trim = do_pad_trim
        self.pad_samples = N_SAMPLES
        
    def output_size(self) -> int:
        return self.encoders.ln_post.normalized_shape[-1]

    def pad_or_trim(
        self,
        array: torch.Tensor,
        length: int,
        axis: int = -1,
    ) -> torch.Tensor:
        """Pad or trim the audio array to N_SAMPLES.

        Used in zero-shot inference cases.
        """
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length).to(array.device)
            )

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])

        return array

    def log_mel_spectrogram(
        self,
        audio: torch.Tensor,
        ilens: torch.Tensor = None,
    ) -> torch.Tensor:
        """Use log-mel spectrogram computation native to Whisper training"""
        window = torch.hann_window(self.win_length).to(audio.device)
        stft = torch.stft(
            audio, self.n_fft, self.hop_length, window=window, return_complex=True
        )

        # whisper deletes the last frame by default (Shih-Lun)
        magnitudes = stft[..., :-1].abs() ** 2

        filters = self.mel_filters(audio.device, self.n_mels)
        mel_spec = filters @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()

        if ilens is not None:
            olens = ilens // self.hop_length
        else:
            olens = None

        log_spec = torch.maximum(
            log_spec,
            log_spec.view(audio.size(0), -1).max(dim=-1)[0][:, None, None] - 8.0,
        )
        log_spec = (log_spec + 4.0) / 4.0

        return log_spec, olens

    def whisper_encode(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor = None,
    ) -> torch.Tensor:
        x = F.gelu(self.encoders.conv1(input))
        # if(self.sidenetwork):
        #     x_intermediate_downsample=F.gelu(self.encoders_sidenetwork.downsample_conv_1(x.permute(0, 2, 1))).permute(0,2,1)
        #     x_downsampled=F.gelu(self.encoders_sidenetwork.conv1(input))
        #     sigmoid_val =  torch.sigmoid(self.encoders_sidenetwork.sigmoid_gate_conv[0])
        #     x_downsampled = (1.0-sigmoid_val)* x_intermediate_downsample + sigmoid_val * x_downsampled
        x = F.gelu(self.encoders.conv2(x))
        # if(self.sidenetwork):
        #     x_intermediate_downsample=F.gelu(self.encoders_sidenetwork.downsample_conv_2(x.permute(0, 2, 1))).permute(0,2,1)
        #     x_downsampled=F.gelu(self.encoders_sidenetwork.conv2(x_downsampled))
        #     sigmoid_val =  torch.sigmoid(self.encoders_sidenetwork.sigmoid_gate_conv[1])
        #     x_downsampled = (1.0-sigmoid_val)* x_intermediate_downsample + sigmoid_val * x_downsampled
        x = x.permute(0, 2, 1)
        # x_downsampled = x_downsampled.permute(0, 2, 1)

        n_frames = x.size(1)
        max_pos = self.encoders.positional_embedding.size(0)
        if n_frames <= max_pos:
            x = (x + self.encoders.positional_embedding[: x.size(1), :]).to(x.dtype)
            # if(self.sidenetwork):
            #     x_downsampled = (x_downsampled + self.encoders_sidenetwork.positional_embedding[: x_downsampled.size(1), :]).to(x_downsampled.dtype)
        else:
            # due to positional encoding, audios >30 sec won't be accepted
            x = x[:, :max_pos, :] + self.encoders.positional_embedding
            # if(self.sidenetwork):
            #     x_downsampled = x_downsampled[:, :x_downsampled, :] + self.encoders.positional_embedding

        x = self.dropout(x)
        attention_scores = []
        # additional
        if(self.sidenetwork):
            x_downsampled = self.encoders_sidenetwork.downsample_input(x)
            side_block = 0
            for i in range(self.encoders.n_layer):
                # forward original model
                x,_ = self.encoders.blocks[i](x)
                # Get the intermediate downsampled output from the original model
                if(i in self.sidenetwork_layers):
                    ## Downsample intermediate model output
                    x_intermediate_downsample = self.encoders_sidenetwork.downsample_intermediate_layers[side_block](x)
                    # x_intermediate_downsample = self.encoders_sidenetwork.downsample_intermediate_layers[side_block](x)
                    ## Apply sigmoid gate to combine the outputs
                    # sigmoid_val =  torch.sigmoid(self.encoders_sidenetwork.sigmoid_gate_intermediate_layers[side_block](x_downsampled))
                    sigmoid_val =  torch.sigmoid(self.encoders_sidenetwork.sigmoid_gate_intermediate_layers[side_block]).to(x.dtype)
                    x_downsampled = (1.0-sigmoid_val)* x_intermediate_downsample + sigmoid_val * x_downsampled
                    ## Forward side model
                    x_downsampled,_ = self.encoders_sidenetwork.blocks[side_block](x_downsampled)
                    # x_downsampled = self.encoders_sidenetwork.blocks[side_block](x_downsampled)
                    side_block+=1
            
        else:
            for layer, block in enumerate(self.encoders.blocks):
                x,_ = block(x)
                # x, att_map = block(x)
                # attention_scores.append(att_map.cpu())
                if layer < len(self.encoders.blocks) - 1:
                    x = self.dropout(x)
        
        x = self.encoders.ln_post(x)
        if(self.sidenetwork):
            x_downsampled = self.encoders_sidenetwork.upsample_output(x_downsampled)
            x_downsampled = self.encoders_sidenetwork.ln_post(x_downsampled)
            sigmoid_val =  torch.sigmoid(self.encoders_sidenetwork.sigmoid_gate_output).to(x.dtype)
            x = (1.0-sigmoid_val)* x + sigmoid_val * x_downsampled
        if ilens is not None:
            olens = (
                1
                + (
                    ilens
                    - self.encoders.conv2.kernel_size[0]
                    + 2 * self.encoders.conv2.padding[0]
                )
                // self.encoders.conv2.stride[0]
            )
            olens = torch.clamp(olens, max=max_pos)
        else:
            olens = None
        if self.sidenetwork:
            return x, olens, x_downsampled
        else:
            return x, olens

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if self.do_pad_trim:
            xs_pad = self.pad_or_trim(xs_pad, self.pad_samples)

        feats, feats_lens = self.log_mel_spectrogram(xs_pad, ilens)

        if self.specaug is not None and self.encoders.training:
            feats, feats_lens = self.specaug(feats, feats_lens)

        if self.sidenetwork:           
            xs_pad, olens, x_downsampled = self.whisper_encode(feats, feats_lens)
            return xs_pad, olens, x_downsampled
        else:
            xs_pad, olens = self.whisper_encode(feats, feats_lens)
            return xs_pad, olens, None
