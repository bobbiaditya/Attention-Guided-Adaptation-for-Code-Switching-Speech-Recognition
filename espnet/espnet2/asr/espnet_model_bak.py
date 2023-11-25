import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union
import string
import torch
import torch.nn as nn
import pickle
import numpy as np
from packaging.version import parse as V
from typeguard import check_argument_types

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.transducer.error_calculator import ErrorCalculatorTransducer
from espnet2.asr_transducer.utils import get_transducer_task_io
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (  # noqa: H301
    LabelSmoothingLoss,
)

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class ESPnetASRModel(AbsESPnetModel):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        encoder: AbsEncoder,
        postencoder: Optional[AbsPostEncoder],
        decoder: Optional[AbsDecoder],
        ctc: CTC,
        joint_network: Optional[torch.nn.Module],
        aux_ctc: dict = None,
        ctc_weight: float = 0.5,
        cs_weight: float = 0.0,
        interctc_weight: float = 0.0,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        # In a regular ESPnet recipe, <sos> and <eos> are both "<sos/eos>"
        # Pretrained HF Tokenizer needs custom sym_sos and sym_eos
        sym_sos: str = "<sos/eos>",
        sym_eos: str = "<sos/eos>",
        extract_feats_in_collect_stats: bool = True,
        lang_token_id: int = -1,
        c_val_attention: float = 0.6,
        head_percentage: float = 100.0,
    ):
        assert check_argument_types()
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        assert 0.0 <= interctc_weight < 1.0, interctc_weight

        super().__init__()
        # NOTE (Shih-Lun): else case is for OpenAI Whisper ASR model,
        #                  which doesn't use <blank> token
        if sym_blank in token_list:
            self.blank_id = token_list.index(sym_blank)
        else:
            self.blank_id = 0
        if sym_sos in token_list:
            self.sos = token_list.index(sym_sos)
        else:
            self.sos = vocab_size - 1
        if sym_eos in token_list:
            self.eos = token_list.index(sym_eos)
        else:
            self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.cs_weight = cs_weight
        self.interctc_weight = interctc_weight
        self.aux_ctc = aux_ctc
        self.token_list = token_list.copy()

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.postencoder = postencoder
        self.encoder = encoder

        if not hasattr(self.encoder, "interctc_use_conditioning"):
            self.encoder.interctc_use_conditioning = False
        if self.encoder.interctc_use_conditioning:
            self.encoder.conditioning_layer = torch.nn.Linear(
                vocab_size, self.encoder.output_size()
            )

        self.use_transducer_decoder = joint_network is not None

        self.error_calculator = None

        if self.use_transducer_decoder:
            from warprnnt_pytorch import RNNTLoss

            self.decoder = decoder
            self.joint_network = joint_network

            self.criterion_transducer = RNNTLoss(
                blank=self.blank_id,
                fastemit_lambda=0.0,
            )

            if report_cer or report_wer:
                self.error_calculator_trans = ErrorCalculatorTransducer(
                    decoder,
                    joint_network,
                    token_list,
                    sym_space,
                    sym_blank,
                    report_cer=report_cer,
                    report_wer=report_wer,
                )
            else:
                self.error_calculator_trans = None

                if self.ctc_weight != 0:
                    self.error_calculator = ErrorCalculator(
                        token_list, sym_space, sym_blank, report_cer, report_wer
                    )
        else:
            # we set self.decoder = None in the CTC mode since
            # self.decoder parameters were never used and PyTorch complained
            # and threw an Exception in the multi-GPU experiment.
            # thanks Jeff Farris for pointing out the issue.
            if ctc_weight < 1.0:
                assert (
                    decoder is not None
                ), "decoder should not be None when attention is used"
            else:
                decoder = None
                logging.warning("Set decoder to none as ctc_weight==1.0")

            self.decoder = decoder

            self.criterion_att = LabelSmoothingLoss(
                size=vocab_size,
                padding_idx=ignore_id,
                smoothing=lsm_weight,
                normalize_length=length_normalized_loss,
            )

            if report_cer or report_wer:
                self.error_calculator = ErrorCalculator(
                    token_list, sym_space, sym_blank, report_cer, report_wer
                )

        if ctc_weight == 0.0:
            self.ctc = None
        else:
            self.ctc = ctc

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

        self.is_encoder_whisper = "Whisper" in type(self.encoder).__name__
        self.c_val_attention = c_val_attention
        # init count for each head in each layer
        self.attention_count = {}
        
        # NEED TO CHANGE HERE
        
        num_layers = 12  # Set the number of layers
        num_heads = 12  # Set the number of heads

        # Loop over the layers
        for layer in range(1, num_layers + 1):
            self.attention_count[layer] = {}

            # Loop over the heads within each layer
            for head in range(1, num_heads + 1):
                self.attention_count[layer][head] = 0
        if(self.cs_weight):
            self.head_percentage=head_percentage
            with open("/home/espnet/egs2/seame/asr1/attention_count_whispernoft_new.pkl", "rb") as file:
                attention_count = pickle.load(file)
            # Step 1: Flatten the nested dictionary into a list of tuples
            frequency_list = []
            for head, layer_dict in attention_count.items():
                for layer, frequency in layer_dict.items():
                    frequency_list.append((head, layer, frequency))

            # Step 2: Sort the list of tuples based on frequency in descending order
            sorted_list = sorted(frequency_list, key=lambda x: x[2], reverse=True)
            
            output_array = torch.zeros((12, 12),dtype=torch.float)  # Adjust the dimensions according to your data
            num_selected_heads = int(110 * self.head_percentage/100)
            for layer, head, num in sorted_list[:num_selected_heads]:  # Modify the range if needed
                # print(num)
                if num >0:
                    output_array[layer-1][head-1] = 1
                # print(layer,head)
            # print(output_array)
            self.selected_heads = output_array

        if self.is_encoder_whisper:
            assert (
                self.frontend is None
            ), "frontend should be None when using full Whisper model"
            from transformers import WhisperTokenizer
            self.tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="chinese", task="transcribe")

        if lang_token_id != -1:
            self.lang_token_id = torch.tensor([[lang_token_id]])
        else:
            self.lang_token_id = None
        # self.estimate_c = estimate_c
        # if self.estimate_c:
        #     self.estimated_c_val = nn.Parameter(torch.Tensor([self.c_val_attention]))
        #     self.estimated_c_val.requires_grad_()
            
    # Function to check if a token is English or not
    def is_english(self,token):
        return all(char in string.ascii_letters for char in token)
    ## Calculate 1 layer only
    # def calculate_cs_loss(self, attention_map, ground_truth_token):        
    #     attention_pattern = torch.stack([self.create_attention_pattern(tensor) for tensor in ground_truth_token])
    #     lid_attention_map = attention_map[:,:,1:3]
    #     # lid_attention_map[torch.isinf(lid_attention_map)] = 3
    #     lid_attention_map[torch.isinf(attention_pattern)] = 0.0
    #     attention_pattern[torch.isinf(attention_pattern)] = 0.0
    #     mse_loss = nn.MSELoss()(attention_pattern, lid_attention_map)
    #     return mse_loss
    # # calculate from 4th layer until last
    # def calculate_cs_loss(self, attention_maps, ground_truth_token):
    #     # Create attention pattern for each ground truth token
    #     attention_pattern = torch.stack([self.create_attention_pattern(tensor) for tensor in ground_truth_token])

    #     # Expand attention pattern to match the shape of attention maps
    #     attention_pattern_expanded = attention_pattern.unsqueeze(0).repeat(attention_maps.shape[0], 1, 1, 1)

    #     # Clone the attention maps and mask the lid_attention_maps based on the expanded attention pattern
    #     lid_attention_maps = attention_maps[:, :, :, 1:3].clone()
    #     lid_attention_maps[torch.isinf(attention_pattern_expanded)] = 0.0
    #     attention_pattern_expanded[torch.isinf(attention_pattern_expanded)] = 0.0
    #     # change shape to batchsize,layer,attention_map
    #     attention_pattern_permute = torch.permute(attention_pattern_expanded,(1,0,2,3))
    #     lid_attention_permute = torch.permute(lid_attention_maps,(1,0,2,3))
    #     # Calculate MSE loss for each attention map
    #     att_map_loss = torch.mean((attention_pattern_permute - lid_attention_permute)**2, dim=-1)
    #     per_layer_loss = torch.mean(att_map_loss,dim=-1) # not sure should be mean or sum
    #     # calculate each loss per batch
    #     per_batch_loss = torch.sum(per_layer_loss,dim=-1)
    #     # sum for each layer, then mean for batch
    #     loss = torch.mean(per_batch_loss)
    #     return loss
    def create_attention_pattern(self, ground_truth_token, attention_default=0.6):
        # Tokenize ground truth
        token_list = self.tokenizer.convert_ids_to_tokens(ground_truth_token)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize variables
        prompt_index = 5
        prompt_length = prompt_index
        lid_length = 0
        
        # estimate c
        if self.decoder.estimate_c:
            attention_default = self.decoder.decoders.estimated_c_val
        # Iterate over word tokens and calculate lid_length
        lid_tensor_list = []
        for token in token_list[prompt_index:]:
            if token == '<|endoftext|>':
                lid_tensor_list.append([attention_default, attention_default])
                lid_length += 1
                break
            elif token.replace("Ġ", "") == '':
                lid_tensor_list.append([attention_default,attention_default])
            else:
                is_english = 1 if self.is_english(token.replace("Ġ", "")) else 0
                # lid_tensor_list.append([-1.0*attention_default if is_english else attention_default, attention_default if is_english else -1.0*attention_default])
                lid_tensor_list.append([0.0 if is_english else attention_default, attention_default if is_english else 0.0])
                # lid_tensor_list.append([attention_default,attention_default])
            lid_length += 1
        
        # Create tensors directly
        # prompt_tensor = torch.full((prompt_length, 2), attention_default, dtype=torch.float)
        prompt_tensor = torch.tensor([[0.0000, 0.0000],
                                     [attention_default, 0.0000],
                                     [0.0000, attention_default],
                                     [0.0000, 0.0000],
                                     [0.0000, 0.0000]], dtype=torch.float)
        lid_tensor = torch.tensor(lid_tensor_list, dtype=torch.float)
        pad_length = len(ground_truth_token) - prompt_length - lid_length
        pad_tensor = torch.full((pad_length, 2), torch.inf, dtype=torch.float)
        # Concatenate tensors
        lid_attention = torch.cat((prompt_tensor, lid_tensor, pad_tensor))
        # Move lid_attention to device
        lid_attention = lid_attention.to(device)
        lid_attention.requires_grad=True
        return lid_attention
    def create_zero_mask(self, num_layers, proportion):
        torch.manual_seed(2022)  # Set the random seed
        matrix = torch.zeros(num_layers, 12)
        num_ones = int(12 * proportion)
        for i in range(num_layers):
            ones_indices = torch.randperm(12)[:num_ones]
            matrix[i, ones_indices] = 1
        return(matrix)
    def new_check_attention_language(self, attention_maps):
        # Modify to be batchsize, layer, head, attention map
        attention_maps = attention_maps.permute(1, 0, 2, 3, 4)
        num_head = attention_maps.shape[1]
        num_layer = attention_maps.shape[2]
        seq_len = attention_maps.shape[3]
        # Loop for each data in the batch size
        for data in attention_maps:
            selected_heads = {}
            selected_layers = []
            for layer in range(num_layer):
                for head in range(num_head):
                    sum_1 = sum(sum(data[layer][head][:,1:3]))
                    sum_2 = sum(data[layer][head][:,0]) + sum(sum(data[layer][head][:,3:]))
                    if(sum_1 > sum_2):
                        if layer not in selected_heads:
                            selected_heads[layer] = []
                        selected_heads[layer].append(head)
                        if layer not in selected_layers:
                            selected_layers.append(layer)
                    # print('a')
        # Update the attention_count dictionary based on the selected heads
            for layer, heads in selected_heads.items():
                for head in heads:
                    self.attention_count[layer+1][head+1] += 1
        return 
    def check_attention_language(self, attention_maps):
        # Modify to be batchsize, layer, head, attention map
        attention_maps = attention_maps.permute(1, 0, 2, 3, 4)
        num_head = attention_maps.shape[1]
        num_layer = attention_maps.shape[2]
        seq_len = attention_maps.shape[3]
        k = 2

        # Loop for each data in the batch size
        for data in attention_maps:
            selected_heads = {}
            selected_layers = []

            # Reshape the tensor to combine the layer and head dimensions
            reshaped_temp = data.reshape(-1, seq_len, seq_len)

            # Calculate the argsort output along the last dimension (seq_len) for each head within each layer
            argsort_output = torch.argsort(reshaped_temp, dim=-1, descending=True)

            # Reshape the argsort output to restore the layer and head dimensions
            argsort_output = argsort_output.reshape(num_layer, num_head, seq_len, seq_len)

            # Iterate over all layers and heads
            for layer in range(num_layer):
                for head in range(num_head):
                    # Get the argsort output for the current head and layer
                    current_output = argsort_output[layer, head]

                    # Get the unique elements and their counts
                    unique_elements, counts = torch.unique(current_output[:, :k].flatten(), return_counts=True)

                    # Create a dictionary with unique elements as keys and counts as values
                    count_dict = {element.item(): count.item() for element, count in zip(unique_elements, counts)}

                    # Sort the dictionary items by their values in descending order
                    sorted_items = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)

                    # Get the top k keys with the biggest values
                    top_keys = [key for key, _ in sorted_items[:k]]

                    # Check if values 1 and 2 are present in the top_keys
                    if 1 in top_keys and 2 in top_keys:
                        if layer not in selected_heads:
                            selected_heads[layer] = []
                        selected_heads[layer].append(head)
                        if layer not in selected_layers:
                            selected_layers.append(layer)

            # Update the attention_count dictionary based on the selected heads
            for layer, heads in selected_heads.items():
                for head in heads:
                    self.attention_count[layer+1][head+1] += 1

    def calculate_cs_loss_onlydistance(self, attention_maps):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        criterion = nn.MSELoss(reduction='none')
        attention_maps = attention_maps.permute(1, 0, 2, 3, 4)
        all_layer_att_map_loss = criterion(attention_maps[:,:,:,:,2],attention_maps[:,:,:,:,1])
        all_layer_head_mse = torch.sum(all_layer_att_map_loss,dim=-1) / torch.count_nonzero(all_layer_att_map_loss,dim=-1)
        random_onezero=torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1.],
                            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                            [1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 0.],
                            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                            [1., 1., 1., 1., 1., 0., 0., 0., 1., 0., 1., 1.]]).to(device)
        all_layer_head_mse_masked = random_onezero * all_layer_head_mse
        final_loss = torch.mean(torch.sum(all_layer_head_mse_masked,dim=[-1,-2]))
        
        
        return -1*final_loss
    
    def getlid(self,ground_truth_token):
        # Tokenize ground truth
        token_list = self.tokenizer.convert_ids_to_tokens(ground_truth_token)
        prompt_index = 5
        prompt_length = prompt_index
        lid_length = 0
        #  1-> zh 2->en
        lid_token=[]
        for token in token_list[prompt_index:]:
            if token == '<|endoftext|>':
                break
            elif token.replace("Ġ", "") == '':
                lid_token.append(2)
            else:
                is_english = 2 if self.is_english(token.replace("Ġ", "")) else 1
                lid_token.append(is_english)
            lid_length += 1
        prompt_tensor = torch.tensor([torch.inf,1,2,torch.inf,torch.inf], dtype=torch.float)
        lid_tensor = torch.tensor(lid_token, dtype=torch.float)
        pad_length = len(ground_truth_token) - prompt_length - lid_length
        pad_tensor = torch.full((pad_length,), torch.inf, dtype=torch.float)
        # Concatenate tensors
        lid_sentence = torch.cat((prompt_tensor, lid_tensor, pad_tensor))
        # Move lid_attention to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lid_sentence = lid_sentence.to(device)
        return lid_sentence
    def calculate_cs_loss_lid_ce(self, attention_maps, ground_truth_token, ground_truth_len):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lid_sentences = torch.stack([self.getlid(tensor) for tensor in ground_truth_token])
        # modify to be batchsize, layer, head, attention map
        attention_maps = attention_maps.permute(1, 0, 2, 3, 4).to(torch.float)
        seq_len = attention_maps.shape[-1]
        bs = attention_maps.shape[0]
        ground_truth_labels = lid_sentences
        # Define a mask for invalid entries
        mask = torch.isfinite(ground_truth_labels).float()
        # Replace 'inf' with 0 for valid labels
        ground_truth_labels[~mask.bool()] = 0
        # Expand dimensions to match the shape of attention maps
        expanded_ground_truth = ground_truth_labels.unsqueeze(1).unsqueeze(2)
        # Repeat the expanded ground truth labels to match the shape of attention maps
        expanded_ground_truth = expanded_ground_truth.repeat(1, 12, 12, 1).to(torch.long)
        
        # mask inf attention maps
        mask = torch.isinf(attention_maps)
        attention_maps[mask] = 0
        # Reshape 'attention_maps' to [7*12*12, 29, 29]
        attention_maps = attention_maps.reshape(-1, seq_len, seq_len)

        # Flatten 'expanded_ground_truth' to [7*12*12, 29]
        expanded_ground_truth = expanded_ground_truth.view(-1, seq_len)
        # Create an instance of nn.CrossEntropyLoss()
        criterion = nn.CrossEntropyLoss(reduction='none',label_smoothing=0.1)
        reshaped_attention_maps = attention_maps.permute(0,2,1)
        # Calculate the cross-entropy loss
        loss = criterion(reshaped_attention_maps, expanded_ground_truth)

        # Reshape the loss back to the original shape [7, 12, 12, 29]
        loss = loss.view(bs, 12, 12, seq_len)
        
        indices_to_mask = [0, 3, 4]
        # Create a mask for the specified indices
        mask_indices = torch.tensor(indices_to_mask, device=loss.device)
        # Create a mask based on 'ground_truth_len' to mark valid tokens
        # delete eot
        ground_truth_len = ground_truth_len -1
        expanded_ground_truth_len = ground_truth_len.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # Create a mask that marks valid tokens based on 'ground_truth_len'
        mask_sequence_length = torch.arange(attention_maps.shape[-1], device=loss.device)[None, None, None, :] < expanded_ground_truth_len

        # Create a final mask for the specified indices masking
        mask_indices = mask_indices.view(1, 1, 1, -1)
        mask_specific_indices = torch.ones_like(mask_sequence_length)
        mask_specific_indices[:, :, :, indices_to_mask] = 0

        # Combine the masks by performing element-wise multiplication
        mask = mask_sequence_length * mask_specific_indices

        # Apply the mask to the 'loss' tensor to ignore tokens beyond the sequence length and specific indices
        masked_loss = loss * mask.float()
        # next mask based on head layer and then finished
        random_onezero = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 1., 1., 1., 0., 1., 1., 0., 0., 1., 1., 1.],
                            [0., 0., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1.],
                            [1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 0., 1.],
                            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
                            [0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                            [1., 0., 0., 1., 1., 0., 1., 0., 1., 0., 1., 0.],
                            [1., 1., 1., 1., 0., 0., 1., 0., 0., 0., 1., 0.],
                            [1., 1., 1., 1., 1., 0., 1., 0., 1., 0., 0., 1.],
                            [0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1.]]).to(device) # 50% selected head 144/2 
        layer_head_loss = masked_loss.nansum(dim=-1)
        cs_masked_layer_head = random_onezero * layer_head_loss
        final_loss = torch.mean(torch.sum(cs_masked_layer_head,dim=[-1,-2]))
        return final_loss
         
    def calculate_cs_loss(self, attention_maps, ground_truth_token,attention_default=0.6):
        # Create attention pattern for each ground truth token
        fourth_layer_attention_pattern = torch.stack([self.create_attention_pattern(tensor,attention_default) for tensor in ground_truth_token])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Create a mask to check if the 2nd value is not inf
        mask = torch.isinf(fourth_layer_attention_pattern[:, :, 1])
        # create pattern for last layer
        last_layer_attention_pattern = torch.zeros(attention_maps.shape[1], attention_maps.shape[-1], attention_maps.shape[-1])
        last_layer_attention_pattern[:, :, 1:3] = fourth_layer_attention_pattern
        # last_layer_attention_pattern[:, :, 1:3] = fourth_layer_attention_pattern/0.5*0.333
        # last_layer_attention_pattern[:, :, 3][~mask] = 0.333 # last layer should be 0.3 for all
        last_layer_attention_pattern=last_layer_attention_pattern.to(device)
        
        # fourth layer pattern extension
        fourth_layer_attention_pattern_extended = torch.zeros(attention_maps.shape[1], attention_maps.shape[-1], attention_maps.shape[-1])
        fourth_layer_attention_pattern_extended[:, :, 1:3] = fourth_layer_attention_pattern
        fourth_layer_attention_pattern_extended=fourth_layer_attention_pattern_extended.to(device)
        #create pattern for early layer
        early_layer_attention_pattern = torch.zeros(attention_maps.shape[1], attention_maps.shape[-1], attention_maps.shape[-1])
        early_layer_attention_pattern[:, :, 0][~mask] = 1.0 # 1 for sot
        early_layer_attention_pattern=early_layer_attention_pattern.to(device)
        
        # attention_pattern for all layer
        attention_pattern = torch.zeros(attention_maps.shape[1], attention_maps.shape[0], attention_maps.shape[-1], attention_maps.shape[-1])
        attention_pattern[:,:2,:,:] = early_layer_attention_pattern.unsqueeze(1).repeat(1,2,1,1)
        # attention_pattern[:,3,:,:] = fourth_layer_attention_pattern_extended # only layer 4
        attention_pattern[:,2:-1,:,:] = fourth_layer_attention_pattern_extended.unsqueeze(1).repeat(1,9,1,1)
        attention_pattern[:,-1,:,:] = last_layer_attention_pattern
        
        # Calculate MSE loss for each attention map and sum it for different layers
        # modify to be batchsize, layer, head, attention map
        attention_maps = attention_maps.permute(1, 0, 2, 3, 4)
        repeated_attention_pattern = attention_pattern.unsqueeze(2).repeat(1, 1, attention_maps.shape[1], 1, 1)
        repeated_attention_pattern = repeated_attention_pattern.to(device)
        # dismiss inf value in attention_ maps
        attention_maps[torch.isinf(repeated_attention_pattern)] = 0.0
        attention_maps[torch.isinf(attention_maps)] = 0.0
        # dismiss inf value in pattern attention map
        repeated_attention_pattern[torch.isinf(repeated_attention_pattern)] = 0.0
        
        criterion = nn.MSELoss(reduction='none')
        # Compare with the early layer attention pattern
        # early_att_map_loss = criterion(attention_maps[:,:3,:,:,:1],repeated_attention_pattern[:,:3,:,:,:1])
        # fourth_att_map_loss = criterion(attention_maps[:,2:3,:,:,1:3],repeated_attention_pattern[:,2:3,:,:,1:3]) # only layer 4
        # last_att_map_loss = criterion(attention_maps[:,-1:,:,:,1:4],repeated_attention_pattern[:,-1:,:,:,1:4])
        fourth_att_map_loss = criterion(attention_maps[:,:,:,:,1:3],repeated_attention_pattern[:,:,:,:,1:3]) 
        all_layer_att_map_loss = torch.concat([
            # torch.sum(early_att_map_loss,dim=-1),
            torch.sum(fourth_att_map_loss,dim=-1),
            # torch.sum(last_att_map_loss,dim=-1)
        ], dim =1)
        all_layer_head_mse = torch.sum(all_layer_att_map_loss,dim=-1) / torch.count_nonzero(all_layer_att_map_loss,dim=-1)
        proportion = 0.5  # proportion of 1s
        # random_onezero=torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #                     [0., 1., 1., 1., 0., 1., 1., 0., 0., 1., 1., 1.],
        #                     [1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
        #                     [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1.],
        #                     [0., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 0.],
        #                     [0., 1., 1., 0., 1., 1., 0., 1., 1., 1., 1., 1.],
        #                     [1., 0., 1., 1., 1., 0., 0., 0., 1., 0., 1., 0.],
        #                     [1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #                     [0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0.],
        #                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]).to(device) # previous_best
        # random_onezero = torch.tensor([[1., 1., 1., 0., 0., 1., 1., 0., 1., 0., 0., 0.],
        #                 [1., 0., 1., 0., 1., 0., 0., 1., 0., 1., 0., 1.],
        #                 [1., 1., 1., 0., 0., 0., 1., 0., 1., 0., 1., 0.],
        #                 [0., 1., 0., 1., 1., 0., 1., 1., 0., 0., 0., 1.],
        #                 [1., 1., 0., 1., 0., 0., 1., 0., 1., 0., 0., 1.],
        #                 [0., 0., 0., 1., 1., 0., 0., 1., 1., 0., 1., 1.],
        #                 [0., 1., 1., 1., 1., 0., 0., 1., 0., 0., 0., 1.],
        #                 [1., 0., 1., 0., 0., 1., 1., 0., 1., 0., 0., 1.],
        #                 [0., 0., 0., 1., 1., 0., 0., 1., 1., 0., 1., 1.],
        #                 [0., 1., 1., 1., 1., 0., 0., 1., 0., 0., 0., 1.],
        #                 [1., 0., 1., 0., 0., 1., 1., 0., 1., 0., 0., 1.],
        #                 [1., 1., 1., 0., 0., 1., 0., 0., 0., 0., 1., 1.]]).to(device) # random 50%
        # random_onezero = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1.],
        #                     [0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0.],
        #                     [0., 1., 0., 1, 0., 0., 0., 0., 0., 0., 0., 0.],
        #                     [0.,0., 0., 0., 0., 1., 0., 0., 0., 1., 1., 1.],
        #                     [0., 0., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1.],
        #                     [1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 0., 1.],
        #                     [0., 0., 1., 1., 0., 1., 1., 1., 1., 1., 1., 0.],
        #                     [0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
        #                     [1., 0., 0., 1., 1., 0., 1., 0., 1., 0., 1., 0.],
        #                     [1., 1., 1., 1., 0., 0., 1., 0., 0., 1., 1., 0.],
        #                     [1., 1., 1., 1., 1., 0., 1., 0., 1., 0., 0., 1.],
        #                     [0., 1., 0., 0., 1., 1., 0., 1., 0., 0., 0., 1.]]).to(device) # random 50%_1
        # random_onezero = torch.tensor([[0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1.],
        #                     [0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 0., 0.],
        #                     [0., 1., 0., 1, 0., 0., 1., 0., 0., 0., 0., 0.],
        #                     [0.,0., 0., 0., 0., 1., 0., 0., 0., 1., 1., 1.],
        #                     [0., 0., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1.],
        #                     [1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 0., 1.],
        #                     [0., 0., 1., 1., 0., 1., 1., 1., 1., 1., 1., 0.],
        #                     [0., 1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1.],
        #                     [1., 0., 0., 1., 1., 0., 1., 0., 1., 0., 1., 0.],
        #                     [1., 1., 0., 1., 0., 0., 1., 0., 0., 1., 1., 0.],
        #                     [1., 1., 0., 1., 1., 0., 1., 0., 1., 0., 0., 1.],
        #                     [0., 1., 0., 0., 1., 1., 0., 1., 0., 1., 0., 1.]]).to(device) # random 50%_2
        # random_onezero = torch.tensor([[0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1.],
        #                     [0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 0., 0.],
        #                     [0., 1., 0., 1, 0., 0., 1., 0., 0., 0., 0., 0.],
        #                     [0.,0., 0., 0., 0., 1., 0., 0., 0., 1., 1., 1.],
        #                     [0., 0., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1.],
        #                     [1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 0., 1.],
        #                     [0., 0., 1., 1., 0., 1., 1., 1., 1., 1., 1., 0.],
        #                     [0., 1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1.],
        #                     [1., 0., 0., 1., 1., 0., 1., 0., 1., 0., 1., 0.],
        #                     [1., 1., 0., 1., 1., 0., 1., 0., 0., 1., 1., 0.],
        #                     [1., 1., 1., 1., 1., 0., 1., 0., 1., 0., 0., 1.],
        #                     [0., 1., 0., 0., 1., 1., 0., 0., 0., 1., 0., 1.]]).to(device) # random 50%_3
        # random_onezero = torch.tensor([[0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 1.],
        #                     [0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 0., 0.],
        #                     [0., 1., 0., 1, 0., 0., 1., 0., 0., 0., 1., 0.],
        #                     [0.,0., 0., 0., 0., 1., 0., 0., 0., 1., 1., 1.],
        #                     [0., 0., 1., 0., 0., 1., 0., 1., 1., 1., 1., 1.],
        #                     [1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 0., 1.],
        #                     [0., 0., 1., 1., 0., 1., 1., 1., 1., 1., 1., 0.],
        #                     [0., 1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1.],
        #                     [1., 0., 0., 1., 1., 0., 1., 0., 1., 0., 1., 0.],
        #                     [1., 1., 0., 0., 1., 0., 1., 0., 0., 1., 1., 0.],
        #                     [1., 1., 0., 0., 1., 0., 1., 0., 1., 0., 0., 1.],
        #                     [0., 1., 0., 0., 1., 1., 0., 0., 0., 1., 0., 1.]]).to(device) # random 50%_4
        # random_onezero = torch.tensor([[1., 1., 0., 1., 1., 0., 0., 1., 0., 0., 1., 0.],
        #                     [0., 1., 0., 1., 1., 0., 0., 0., 1., 1., 1., 1.],
        #                     [1., 1., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0.],
        #                     [0., 1., 1., 1., 1., 1., 1., 0., 1., 0., 0., 0.],
        #                     [0., 0., 0., 0., 1., 1., 1., 1., 1., 0., 0., 1.],
        #                     [0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 1., 0.],
        #                     [1., 1., 1., 1., 1., 0., 0., 1., 0., 1., 0., 0.],
        #                     [0., 1., 0., 1., 0., 0., 0., 1., 1., 1., 0., 0.],
        #                     [0., 1., 1., 0., 0., 0., 0., 1., 0., 1., 0., 1.],
        #                     [0., 1., 0., 1., 0., 1., 1., 0., 0., 1., 0., 1.],
        #                     [0., 0., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1.],
        #                     [1., 0., 0., 1., 0., 1., 0., 1., 1., 0., 1., 1.]]).to(device) # random 50%_5
        # random_onezero = torch.tensor([[1., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0.],
        #                 [1., 0., 1., 0., 1., 0., 0., 1., 0., 1., 0., 1.],
        #                 [1., 1., 1., 0., 0., 0., 1., 0., 1., 0., 1., 0.],
        #                 [0., 1., 0., 1., 1., 0., 1., 1., 0., 0., 0., 1.],
        #                 [1., 1., 0., 1., 0., 0., 1., 0., 1., 0., 0., 1.],
        #                 [0., 0., 0., 1., 1., 0., 0., 1., 1., 0., 1., 1.],
        #                 [0., 1., 1., 1., 1., 0., 0., 1., 0., 0., 0., 1.],
        #                 [1., 0., 1., 0., 0., 1., 1., 0., 1., 0., 0., 1.],
        #                 [0., 0., 0., 1., 1., 0., 0., 1., 1., 0., 1., 1.],
        #                 [0., 1., 1., 1., 1., 0., 0., 1., 0., 1., 0., 1.],
        #                 [1., 0., 1., 0., 0., 1., 1., 0., 1., 0., 0., 1.],
        #                 [1., 1., 1., 0., 0., 1., 0., 0., 0., 1., 1., 1.]]).to(device) # random 50%_6
        # random_onezero = torch.tensor([[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
        #                     [0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.],
        #                     [0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.],
        #                     [1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.],
        #                     [1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.],
        #                     [1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.],
        #                     [1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.],
        #                     [1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.],
        #                     [1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.],
        #                     [1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.],
        #                     [1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.],
        #                     [1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.]]).to(device) # all selected head
        random_onezero = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 1., 1., 1., 0., 1., 1., 0., 0., 1., 1., 1.],
                            [0., 0., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1.],
                            [1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 0., 1.],
                            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
                            [0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                            [1., 0., 0., 1., 1., 0., 1., 0., 1., 0., 1., 0.],
                            [1., 1., 1., 1., 0., 0., 1., 0., 0., 0., 1., 0.],
                            [1., 1., 1., 1., 1., 0., 1., 0., 1., 0., 0., 1.],
                            [0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1.]]).to(device) # 50% selected head 144/2 
        # random_onezero = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #                     [0., 1., 1., 1., 0., 1., 1., 0., 0., 0., 1., 1.],
        #                     [0., 0., 1., 1., 0., 1., 0., 1., 1., 1., 1., 1.],
        #                     [0., 1., 1., 1., 1., 1., 0., 1., 1., 1., 0., 1.],
        #                     [1., 0., 1., 1., 1., 1., 1., 1., 1., 0., 1., 0.],
        #                     [0., 0., 1., 1., 1., 1., 1., 1., 1., 0., 0., 1.],
        #                     [1., 0., 0., 1., 1., 0., 0., 0., 1., 0., 1., 0.],
        #                     [1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        #                     [1., 1., 1., 0., 1., 0., 1., 0., 1., 0., 0., 0.],
        #                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]).to(device) # 50% selected head 110/2 
        
        all_layer_head_mse_masked = random_onezero * all_layer_head_mse
        # all_layer_head_mse_masked = self.selected_heads.to(device) * all_layer_head_mse
        
        final_loss = torch.mean(torch.sum(all_layer_head_mse_masked,dim=[-1,-2]))
        # Compare with the fourth layer attention pattern
        
        # att_map_loss = criterion(attention_maps[3, :, :, 1:3], fourth_layer_attention_pattern)
        # loss += att_map_loss
        # # Compare with the last layer attention pattern
        # att_map_loss = criterion(attention_maps[-1, :, :, 1:4], last_layer_attention_pattern)
        # loss += att_map_loss
        # # Compare with the early layer attention pattern
        # early_layer_att_maps = attention_maps[:3, :, :, :1]
        # att_map_loss = nn.MSELoss(reduction='none')(early_layer_att_maps, early_layer_attention_pattern.expand_as(early_layer_att_maps))
        # att_map_loss = torch.sum(torch.mean(torch.mean(torch.sum(att_map_loss,dim=-1),dim=-1),dim=-1))
        # loss += att_map_loss
        return final_loss 


    
    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
            kwargs: "utt_id" is among the input.
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        batch_size = speech.shape[0]

        text[text == -1] = self.ignore_id

        # for data-parallel
        text = text[:, : text_lengths.max()]

        # 1. Encoder
        if self.encoder.sidenetwork:
            encoder_out, encoder_out_lens, encoder_sidenetwork_out = self.encode(speech, speech_lengths)
        else:    
            encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        loss_att, acc_att, cer_att, wer_att = None, None, None, None
        loss_ctc, cer_ctc = None, None
        loss_transducer, cer_transducer, wer_transducer = None, None, None
        stats = dict()

        # 1. CTC branch
        if self.ctc_weight != 0.0:
            if(self.encoder.sidenetwork):
                loss_ctc, cer_ctc = self._calc_ctc_loss(
                    encoder_sidenetwork_out, encoder_out_lens, text, text_lengths
                )
            else:
                loss_ctc, cer_ctc = self._calc_ctc_loss(
                    encoder_out, encoder_out_lens, text, text_lengths
                )

            # Collect CTC branch stats
            stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
            stats["cer_ctc"] = cer_ctc

        # Intermediate CTC (optional)
        loss_interctc = 0.0
        if self.interctc_weight != 0.0 and intermediate_outs is not None:
            for layer_idx, intermediate_out in intermediate_outs:
                # we assume intermediate_out has the same length & padding
                # as those of encoder_out

                # use auxillary ctc data if specified
                loss_ic = None
                if self.aux_ctc is not None:
                    idx_key = str(layer_idx)
                    if idx_key in self.aux_ctc:
                        aux_data_key = self.aux_ctc[idx_key]
                        aux_data_tensor = kwargs.get(aux_data_key, None)
                        aux_data_lengths = kwargs.get(aux_data_key + "_lengths", None)

                        if aux_data_tensor is not None and aux_data_lengths is not None:
                            loss_ic, cer_ic = self._calc_ctc_loss(
                                intermediate_out,
                                encoder_out_lens,
                                aux_data_tensor,
                                aux_data_lengths,
                            )
                        else:
                            raise Exception(
                                "Aux. CTC tasks were specified but no data was found"
                            )
                if loss_ic is None:
                    loss_ic, cer_ic = self._calc_ctc_loss(
                        intermediate_out, encoder_out_lens, text, text_lengths
                    )
                loss_interctc = loss_interctc + loss_ic

                # Collect Intermedaite CTC stats
                stats["loss_interctc_layer{}".format(layer_idx)] = (
                    loss_ic.detach() if loss_ic is not None else None
                )
                stats["cer_interctc_layer{}".format(layer_idx)] = cer_ic

            loss_interctc = loss_interctc / len(intermediate_outs)

            # calculate whole encoder loss
            loss_ctc = (
                1 - self.interctc_weight
            ) * loss_ctc + self.interctc_weight * loss_interctc

        if self.use_transducer_decoder:
            # 2a. Transducer decoder branch
            (
                loss_transducer,
                cer_transducer,
                wer_transducer,
            ) = self._calc_transducer_loss(
                encoder_out,
                encoder_out_lens,
                text,
            )

            if loss_ctc is not None:
                loss = loss_transducer + (self.ctc_weight * loss_ctc)
            else:
                loss = loss_transducer

            # Collect Transducer branch stats
            stats["loss_transducer"] = (
                loss_transducer.detach() if loss_transducer is not None else None
            )
            stats["cer_transducer"] = cer_transducer
            stats["wer_transducer"] = wer_transducer

        else:
            # 2b. Attention decoder branch
            if self.ctc_weight != 1.0:
                if(self.encoder.sidenetwork):
                    loss_att, acc_att, cer_att, wer_att, loss_cs = self._calc_att_loss(
                        encoder_out, encoder_out_lens, text, text_lengths, encoder_sidenetwork_out
                    )
                else:
                    loss_att, acc_att, cer_att, wer_att, loss_cs = self._calc_att_loss(
                        encoder_out, encoder_out_lens, text, text_lengths
                    )

            # 3. CTC-Att loss definition
            if self.ctc_weight == 0.0:
                loss = loss_att
            elif self.ctc_weight == 1.0:
                loss = loss_ctc
            else:
                loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att
            
            if self.cs_weight != 0.0:
                # loss = self.cs_weight * loss_cs + (1 - self.cs_weight) * loss_att
                # if loss_att > 80.0:
                #     cs_weight = 4.0
                # elif loss_att > 40.0:
                #     cs_weight = 2.0
                # else :
                #     cs_weight=1.0
                loss = self.cs_weight * loss_cs + loss_att
                stats['loss_cs'] = loss_cs.detach()
            # modify here later

            # Collect Attn branch stats
            stats["loss_att"] = loss_att.detach() if loss_att is not None else None
            stats["acc"] = acc_att
            stats["cer"] = cer_att
            stats["wer"] = wer_att
            # CKPT 1
            # stats['loss_cs'] = loss_cs.detach()
        # Collect total loss stats
        stats["loss"] = loss.detach()

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)

            # 2. Data augmentation
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

        # Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            feats, feats_lengths = self.preencoder(feats, feats_lengths)

        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        if self.encoder.interctc_use_conditioning:
            encoder_out, encoder_out_lens, _ = self.encoder(
                feats, feats_lengths, ctc=self.ctc
            )
        elif self.encoder.sidenetwork:
            encoder_out, encoder_out_lens, encoder_sidenetwork_out = self.encoder(feats, feats_lengths)
        else:
            encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        # Post-encoder, e.g. NLU
        if self.postencoder is not None:
            encoder_out, encoder_out_lens = self.postencoder(
                encoder_out, encoder_out_lens
            )

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        if (
            getattr(self.encoder, "selfattention_layer_type", None) != "lf_selfattn"
            and not self.is_encoder_whisper
        ):
            assert encoder_out.size(-2) <= encoder_out_lens.max(), (
                encoder_out.size(),
                encoder_out_lens.max(),
            )

        if intermediate_outs is not None:
            return (encoder_out, intermediate_outs), encoder_out_lens
        if self.encoder.sidenetwork:
            return encoder_out, encoder_out_lens, encoder_sidenetwork_out
        else:    
            return encoder_out, encoder_out_lens

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths

    def nll(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute negative log likelihood(nll) from transformer-decoder

        Normally, this function is called in batchify_nll.

        Args:
            encoder_out: (Batch, Length, Dim)
            encoder_out_lens: (Batch,)
            ys_pad: (Batch, Length)
            ys_pad_lens: (Batch,)
        """
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1


        decoder_out, _ = self.decoder(
                encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
            )  # [batch, seqlen, dim]
        batch_size = decoder_out.size(0)
        decoder_num_class = decoder_out.size(2)
        # nll: negative log-likelihood
        nll = torch.nn.functional.cross_entropy(
            decoder_out.view(-1, decoder_num_class),
            ys_out_pad.view(-1),
            ignore_index=self.ignore_id,
            reduction="none",
        )
        nll = nll.view(batch_size, -1)
        nll = nll.sum(dim=1)
        assert nll.size(0) == batch_size
        return nll

    def batchify_nll(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        batch_size: int = 100,
    ):
        """Compute negative log likelihood(nll) from transformer-decoder

        To avoid OOM, this fuction seperate the input into batches.
        Then call nll for each batch and combine and return results.
        Args:
            encoder_out: (Batch, Length, Dim)
            encoder_out_lens: (Batch,)
            ys_pad: (Batch, Length)
            ys_pad_lens: (Batch,)
            batch_size: int, samples each batch contain when computing nll,
                        you may change this to avoid OOM or increase
                        GPU memory usage
        """
        total_num = encoder_out.size(0)
        if total_num <= batch_size:
            nll = self.nll(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)
        else:
            nll = []
            start_idx = 0
            while True:
                end_idx = min(start_idx + batch_size, total_num)
                batch_encoder_out = encoder_out[start_idx:end_idx, :, :]
                batch_encoder_out_lens = encoder_out_lens[start_idx:end_idx]
                batch_ys_pad = ys_pad[start_idx:end_idx, :]
                batch_ys_pad_lens = ys_pad_lens[start_idx:end_idx]
                batch_nll = self.nll(
                    batch_encoder_out,
                    batch_encoder_out_lens,
                    batch_ys_pad,
                    batch_ys_pad_lens,
                )
                nll.append(batch_nll)
                start_idx = end_idx
                if start_idx == total_num:
                    break
            nll = torch.cat(nll)
        assert nll.size(0) == total_num
        return nll

    # def whisper_pad(self,ys_pad, sos, eos, ignore_id):
    #     from espnet.nets.pytorch_backend.nets_utils import pad_list
    #     _sos = ys_pad.new([sos])
    #     ys = [y[y != ignore_id] for y in ys_pad]  # parse padded ys
    #     ys_in = [torch.cat([_sos, y], dim=0) for y in ys]
    #     return pad_list(ys_in, eos), pad_list(ys_pad, ignore_id)
    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        encoder_sidenetwork_out:torch.Tensor=None,
    ):
        if hasattr(self, "lang_token_id") and self.lang_token_id is not None:
            ys_pad = torch.cat(
                [
                    self.lang_token_id.repeat(ys_pad.size(0), 1).to(ys_pad.device),
                    ys_pad,
                ],
                dim=1,
            )
            ys_pad_lens += 1
            
        # if self.is_encoder_whisper:
        #     ys_in_pad, ys_out_pad = self.whisper_pad(ys_pad, self.sos, self.eos, self.ignore_id)
        # else:
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1
        # decoder output here
        # 1. Forward decoder
        # torch.autograd.set_detect_anomaly(True)
        if(self.encoder.sidenetwork):
            decoder_out, att_map = self.decoder(
                encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens, encoder_sidenetwork_out
            )
        else:
            decoder_out, att_map = self.decoder(
                encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
            )
            
        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        
        # compute additional loss here
        loss_cs = None
        # CKPT 1
        if self.is_encoder_whisper and self.cs_weight!=0:
            # att_map = torch.mean(att_map,dim=2)
            # self.check_attention_language(att_map)
            # self.new_check_attention_language(att_map)
            # loss_cs = self.calculate_cs_loss(att_map, ys_in_pad,self.c_val_attention)
            loss_cs = self.calculate_cs_loss_lid_ce(att_map, ys_in_pad,ys_in_lens)
            # loss_cs = self.calculate_cs_loss_onlydistance(att_map)
        
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )
        
        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())
        #add additional return value
        return loss_att, acc_att, cer_att, wer_att, loss_cs
    # define new function to calculate additional loss
    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc

    def _calc_transducer_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        labels: torch.Tensor,
    ):
        """Compute Transducer loss.

        Args:
            encoder_out: Encoder output sequences. (B, T, D_enc)
            encoder_out_lens: Encoder output sequences lengths. (B,)
            labels: Label ID sequences. (B, L)

        Return:
            loss_transducer: Transducer loss value.
            cer_transducer: Character error rate for Transducer.
            wer_transducer: Word Error Rate for Transducer.

        """
        decoder_in, target, t_len, u_len = get_transducer_task_io(
            labels,
            encoder_out_lens,
            ignore_id=self.ignore_id,
            blank_id=self.blank_id,
        )

        self.decoder.set_device(encoder_out.device)
        decoder_out = self.decoder(decoder_in)

        joint_out = self.joint_network(
            encoder_out.unsqueeze(2), decoder_out.unsqueeze(1)
        )

        loss_transducer = self.criterion_transducer(
            joint_out,
            target,
            t_len,
            u_len,
        )

        cer_transducer, wer_transducer = None, None
        if not self.training and self.error_calculator_trans is not None:
            cer_transducer, wer_transducer = self.error_calculator_trans(
                encoder_out, target
            )

        return loss_transducer, cer_transducer, wer_transducer

    def _calc_batch_ctc_loss(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ):
        if self.ctc is None:
            return
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)

        # for data-parallel
        text = text[:, : text_lengths.max()]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        # Calc CTC loss
        do_reduce = self.ctc.reduce
        self.ctc.reduce = False
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, text, text_lengths)
        self.ctc.reduce = do_reduce
        return loss_ctc
