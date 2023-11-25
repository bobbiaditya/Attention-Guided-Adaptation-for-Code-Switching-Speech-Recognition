import copy
from typing import Any, List, Tuple

import torch
from typeguard import check_argument_types
import torch.nn as nn
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet.nets.scorer_interface import BatchScorerInterface
import torch.nn.functional as F


class OpenAIWhisperDecoder(AbsDecoder, BatchScorerInterface):
    """Transformer-based Speech-to-Text Decoder from OpenAI's Whisper Model:

    URL: https://github.com/openai/whisper
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        dropout_rate: float = 0.0,
        whisper_model: str = "small",
        download_dir: str = None,
        src_layer : int = 12,
        whisper_cs: bool = False,
        pe_whisper: bool = False,
        adapter: bool = False,
        side_network : bool = False,
        side_network_conf = None,
        c_val_attention: float = 0.6,
        estimate_c : bool = False
    ):
        try:
            # from whisper import whisper
            import whisper
        except Exception as e:
            print("Error: whisper is not properly installed.")
            print(
                "Please install whisper with: cd ${MAIN_ROOT}/tools && "
                "./installers/install_whisper.sh"
            )
            raise e

        assert check_argument_types()
        super().__init__()

        assert whisper_model in whisper.available_models()
        _model = whisper.load_model(whisper_model, adapter, pe_whisper, side_network,side_network_conf, download_root=download_dir)
        self.sidenetwork = side_network
        self.decoders = copy.deepcopy(_model.decoder)
        if(self.sidenetwork):
            # for param in self.decoders.parameters():
            #     param.requires_grad = False
            self.decoders_sidenetwork = copy.deepcopy(_model.decoder_sidenetwork)
            self.decoders_sidenetwork.train()
            self.sidenetwork_layers = side_network_conf['layers']
        else:
            self.decoders.train()
        del _model
        attention_dim = self.decoders.token_embedding.embedding_dim
        # note that originally Whisper doesn't use dropouts
        self.dropout = torch.nn.Dropout(dropout_rate)
        # vocab size mismatch -> reinitialize embedding
        # orig vocab size (multilingual): 51865
        # orig vocab size (english): 51864
        if vocab_size != self.decoders.token_embedding.num_embeddings:
            orig_emb_std, orig_emb_mean = torch.std_mean(
                self.decoders.token_embedding.weight
            )
            self.decoders.token_embedding = torch.nn.Embedding(
                vocab_size, attention_dim
            )
            # init new embedding with original embedding with a normal distribution with original embedding mean and std. 
            torch.nn.init.normal_(
                self.decoders.token_embedding.weight,
                orig_emb_mean.item(),
                orig_emb_std.item(),
            )
        self.whisper_cs = whisper_cs
        self.src_layer = src_layer - 1
        self.att_map = None
        self.estimate_c = estimate_c
        self.c_val_attention = c_val_attention
        if self.estimate_c:
            self.decoders.estimated_c_val = nn.Parameter(torch.Tensor([self.c_val_attention]))
            
        
    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        side_encoder_output:torch.Tensor=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward decoder.

        Args:
            hs_pad: encoded memory, float32  (batch, maxlen_in, feat)
            hlens: (batch)
            ys_in_pad:
                input token ids, int64 (batch, maxlen_out)
                if input_layer == "embed"
                input tensor (batch, maxlen_out, #mels) in the other cases
            ys_in_lens: (batch)
        Returns:
            (tuple): tuple containing:

            x: decoded token score before softmax (batch, maxlen_out, token)
                if use_output_layer is True,
            olens: (batch, )
        """
        # fix ys_in_pad
        # tensor_without_first = ys_in_pad[:, 1:]
        # ys_in_pad = torch.cat((tensor_without_first, torch.full((ys_in_pad.shape[0], 1), 50257, device=ys_in_pad.device)), dim=1)
        tgt, memory = ys_in_pad, hs_pad
        tgt = (
            self.decoders.token_embedding(tgt)
            + self.decoders.positional_embedding[: tgt.size(1)]
        )
        tgt = self.dropout(tgt)
            
        x = tgt.to(memory.dtype)
        attention_scores = []
        if(self.sidenetwork):
            x_downsampled = self.decoders_sidenetwork.downsample_input(x)
            # x_downsampled = self.decoders_sidenetwork.downsample_input_3(self.decoders_sidenetwork.downsample_input_2(F.gelu(self.decoders_sidenetwork.downsample_input_1(x))))
            side_encoder_output = self.decoders_sidenetwork.downsample_encoder_input(memory)
            side_block = 0
            for i in range(self.decoders.n_layer):
                # forward original model
                x,attention_map = self.decoders.blocks[i](x, memory, mask=self.decoders.mask)
                if(i in self.sidenetwork_layers):
                    ## Get the intermediate downsampled output from the original model
                    x_intermediate_downsample = self.decoders_sidenetwork.downsample_intermediate_layers[side_block](x)
                    # x_intermediate_downsample = self.decoders_sidenetwork.downsample_intermediate_layers_3[side_block](self.decoders_sidenetwork.downsample_intermediate_layers_2[side_block](F.gelu(self.decoders_sidenetwork.downsample_intermediate_layers_1[side_block](x))))
                    ## Apply sigmoid gate to combine the outputs
                    # sigmoid_val =  torch.sigmoid(self.decoders_sidenetwork.sigmoid_gate_intermediate_layers[side_block](x_downsampled))
                    sigmoid_val =  torch.sigmoid(self.decoders_sidenetwork.sigmoid_gate_intermediate_layers[side_block]).to(x.dtype)
                    x_downsampled = (1.0-sigmoid_val)* x_intermediate_downsample + sigmoid_val * x_downsampled
                    ## forward side model
                    x_downsampled,_ = self.decoders_sidenetwork.blocks[side_block](x_downsampled, side_encoder_output, mask=self.decoders_sidenetwork.mask)
                    # x_downsampled = self.decoders_sidenetwork.blocks[side_block](x_downsampled)
                    side_block+=1
                if(self.whisper_cs and i >= self.src_layer ):
                    # if layer!=11:
                    # attention_map = attention_map.detach()
                    attention_scores.append(attention_map)
        else:
            for layer, block in enumerate(self.decoders.blocks):
                x,attention_map = block(x, memory, mask=self.decoders.mask)
                if layer < len(self.decoders.blocks) - 1:
                    x = self.dropout(x)
                if(self.whisper_cs and layer >= self.src_layer ):
                    # if layer!=11:
                    # attention_map = attention_map.detach()
                    attention_scores.append(attention_map)
        x = self.decoders.ln(x)
        if(self.sidenetwork):
            x_downsampled = self.decoders_sidenetwork.upsample_output(x_downsampled)
            x = self.decoders_sidenetwork.ln(x_downsampled)
           
        x = (
            x @ torch.transpose(self.decoders.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()
        if self.whisper_cs:
            return x, torch.stack(attention_scores)
        else:
            return x,attention_scores

    def forward_one_step(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory: torch.Tensor,
        cache: List[torch.Tensor] = None,
        side_encoder_output:torch.Tensor=None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward one step.

        Args:
            tgt: input token ids, int64 (batch, maxlen_out)
            tgt_mask: input token mask,  (batch, maxlen_out)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            cache: cached output list of (batch, max_time_out-1, size)
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
            y.shape` is (batch, maxlen_out, token)
        NOTE (Shih-Lun):
            cache implementation is ignored for now
            for simplicity & correctness
        """
        # print(tgt.size(dim=1))
        if(tgt.size(dim=1)>448):
            tgt=tgt[:, :448]
        #     print(tgt.shape)
        x = (
            self.decoders.token_embedding(tgt)
            + self.decoders.positional_embedding[: tgt.size(1)]
        )
        x = self.dropout(x)
        x = x.to(memory.dtype)
        ## additional
        attention_scores = []
        if(self.sidenetwork):
            x_downsampled = self.decoders_sidenetwork.downsample_input(x)
            memory_sidenetwork = self.decoders_sidenetwork.downsample_encoder_input(memory)
            side_block=0
            for i in range(self.decoders.n_layer):
                # forward original model
                x,_ = self.decoders.blocks[i](x, memory, mask=self.decoders.mask)
                # Get the intermediate downsampled output from the original model
                if i in self.sidenetwork_layers:
                    x_intermediate_downsample = self.decoders_sidenetwork.downsample_intermediate_layers[side_block](x)
                    ## Apply sigmoid gate to combine the outputs
                    # sigmoid_val =  torch.sigmoid(self.decoders_sidenetwork.sigmoid_gate_intermediate_layers[side_block](x_downsampled))
                    sigmoid_val =  torch.sigmoid(self.decoders_sidenetwork.sigmoid_gate_intermediate_layers[side_block]).to(x.dtype)
                    x_downsampled = (1.0-sigmoid_val) * x_intermediate_downsample + sigmoid_val * x_downsampled
                    ## forward side model
                    x_downsampled,att_map = self.decoders_sidenetwork.blocks[side_block](x_downsampled, memory_sidenetwork, mask=self.decoders_sidenetwork.mask)
                    # x_downsampled = self.decoders_sidenetwork.blocks[side_block](x_downsampled)
                    side_block+=1
                    # attention_scores.append(att_map.cpu())
        else:
            for layer, block in enumerate(self.decoders.blocks):
                x, att_map = block(x, memory, mask=self.decoders.mask)
                attention_scores.append(att_map.cpu())
                if layer < len(self.decoders.blocks) - 1:
                    x = self.dropout(x)
        x = self.decoders.ln(x)
        if(self.sidenetwork):
            x_downsampled = self.decoders_sidenetwork.upsample_output(x_downsampled)
            x = self.decoders_sidenetwork.ln(x_downsampled)
        y = x[:, -1]
        y = (
            y @ torch.transpose(self.decoders.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()
        y = torch.log_softmax(y, dim=-1)
        if torch.argmax(y).cpu() == 50257:
              print('end')
        return y, None

    def score(self, ys, state, x):
        """Score."""
        logp, state = self.forward_one_step(
            ys.unsqueeze(0), torch.empty(0), x.unsqueeze(0), cache=state  # dummy mask
        )
        return logp.squeeze(0), state

    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor,x_enc:torch.Tensor=None
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch.

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).

        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.

        """
        # batch decoding, dummy mask is passed
        logp, states = self.forward_one_step(ys, torch.empty(0), xs, cache=None,side_encoder_output=x_enc)

        return logp, None
