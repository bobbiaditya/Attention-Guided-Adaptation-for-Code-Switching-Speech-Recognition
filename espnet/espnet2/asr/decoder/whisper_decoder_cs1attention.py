import copy
from typing import Any, List, Tuple

import torch
from typeguard import check_argument_types

from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet.nets.scorer_interface import BatchScorerInterface


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
        whisper_cs: bool = False
    ):
        try:
            from whisper import whisper
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
        _model = whisper.load_model(whisper_model, download_root=download_dir)
        self.decoders = copy.deepcopy(_model.decoder)
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
        self.src_layer = src_layer
        self.decoders.train()
        self.att_map = None
        del _model
        
    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
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
        att_map = None
        if(self.whisper_cs):
            def get_attention_hook(module, input, output):
                nonlocal att_map
                att_map = torch.mean(output[1], dim=1)
            handle = self.decoders.blocks[self.src_layer-1].attn.register_forward_hook(get_attention_hook)
        for layer, block in enumerate(self.decoders.blocks):
            x = block(x, memory, mask=self.decoders.mask)
            if layer < len(self.decoders.blocks) - 1:
                x = self.dropout(x)

        x = self.decoders.ln(x)
        x = (
            x @ torch.transpose(self.decoders.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()
        if(self.whisper_cs):
            handle.remove()
        return x, att_map

    def forward_one_step(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory: torch.Tensor,
        cache: List[torch.Tensor] = None,
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
        # attention_scores = []
        # cross_attention_scores = []
        # def get_attention_hook(module, input, output):
        #     attention_scores.append(output[1].cpu())
        # def get_cross_attention_hook(module, input, output):
        #     cross_attention_scores.append(output[1].cpu())
        for layer, block in enumerate(self.decoders.blocks):
            # handle = block.attn.register_forward_hook(get_attention_hook) 
            # handle2 = block.cross_attn.register_forward_hook(get_cross_attention_hook) 
            x = block(x, memory, mask=self.decoders.mask)
            if layer < len(self.decoders.blocks) - 1:
                x = self.dropout(x)
        x = self.decoders.ln(x)
        y = x[:, -1]
        y = (
            y @ torch.transpose(self.decoders.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()
        y = torch.log_softmax(y, dim=-1)
        # if torch.argmax(y).cpu() == 50257:
        #     print('akhir')
        return y, None

    def score(self, ys, state, x):
        """Score."""
        logp, state = self.forward_one_step(
            ys.unsqueeze(0), torch.empty(0), x.unsqueeze(0), cache=state  # dummy mask
        )
        return logp.squeeze(0), state

    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
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
        logp, states = self.forward_one_step(ys, torch.empty(0), xs, cache=None)

        return logp, None
