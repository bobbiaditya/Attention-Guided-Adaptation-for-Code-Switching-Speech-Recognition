from typing import Iterable, List, Union

import numpy as np
from typeguard import check_argument_types

# <sos> and <eos> for Whisper multilingual ---
# '<|startoftranscript|>': 50258
# '<|endoftext|>':         50257

# <sos> and <eos> for Whisper english ---
# '<|startoftranscript|>': 50257
# '<|endoftext|>':         50256


class OpenAIWhisperTokenIDConverter:
    def __init__(
        self,
        model_type: str = "whisper_multilingual",
    ):
        assert check_argument_types()

        try:
            import whisper.tokenizer
            from transformers import WhisperTokenizer
        except Exception as e:
            print("Error: whisper is not properly installed.")
            print(
                "Please install whisper with: cd ${MAIN_ROOT}/tools && "
                "./installers/install_whisper.sh"
            )
            raise e

        if model_type == "whisper_en":
            self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual=False)
        # TODO(Shih-Lun): should support feeding in
        #                  different languages (default is en)
        elif model_type == "whisper_multilingual":
            # self.tokenizer = whisper.tokenizer.get_tokenizer(
            #     multilingual=True, language=None
            # )
            self.tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="chinese", task="transcribe")
        else:
            raise ValueError("tokenizer unsupported:", model_type)

    def get_num_vocabulary_size(self) -> int:
        return self.tokenizer.vocab_size + len(
            self.tokenizer.get_added_vocab()
        )

    def ids2tokens(self, integers: Union[np.ndarray, Iterable[int]]) -> List[str]:
        # return  self.tokenizer.convert_ids_to_tokens(integers)
        # return self.tokenizer.decode(integers)
        return self.tokenizer.convert_ids_to_tokens(
            integers, skip_special_tokens=True
        )

    def tokens2ids(self, tokens: Iterable[str]) -> List[int]:
        # print(tokens)
        # return list(self.tokenizer(tokens).input_ids)
        new_prefix_tokens = [50260, 50259, 50359, 50363] # zh en asr notimestamps
        # new_prefix_tokens = [50260, 50359, 50363] # zh asr notimestamps
        return list(
            new_prefix_tokens
        ) + self.tokenizer.convert_tokens_to_ids(tokens) + [self.tokenizer.eos_token_id]
        # return list(
        #     self.tokenizer.prefix_tokens
        # ) + self.tokenizer.convert_tokens_to_ids(tokens) + [self.tokenizer.eos_token_id]
        # return list(
        #     self.tokenizer.sot_sequence_including_notimestamps[1:]
        # ) + self.tokenizer.convert_tokens_to_ids(tokens)
