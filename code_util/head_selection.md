# Attention Head Selection

To perform attention head selection, we need to do forward pass over all dataset without updating the model parameters and check the attention map output for every heads in every layers for each data. 

The head selection is done by utilizing debugging process of python, here is the step by step process:
1. Comment the backward process in `espnet/espnet2/train/trainer.py` at line 644
2. Uncomment the new_check_attention_language function call in `espnet/espnet2/asr/espnet_model.py` at line 944. This function will check the attention map output for each data and save the calculation to `self.attention_count` under class `espnet_model`.
3. Run `espnet/espnet2/bin/asr_train.py` under debugging process with the configuration following `code_util/debug_config.json` put the breakpoint in `espnet/espnet2/train/trainer.py` at line 309, after finish calling the `train_one_epoch` function.
4. After finish, we can save the `self.attention_count` dictionary into a pkl file by running this code
```
# Save attention_count dictionary to a file
with open("[filename here]", "wb") as file:
    pickle.dump(self.attention_count, file)

```
5. After we have the attention count, we need to modify the `espnet/espnet2/asr/espnet_model.py` at line 200 where we define the path for attention count.