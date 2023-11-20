# Printing Attention Map

To print attention map output of the model, we take the attention map output of the model given a data until it produce the eot token that indicated end of transcript.

We utilize debugging function from the inference process of the model, we can utilize the utility function `whisper_check.py`. In the code, we will give a speech input to the model and it will produce the text transcription by doing forward pass to the model.

Here is the step by step to print the attention map:
1. Check the `espnet/espnet2/asr/decoder/whisper_decoder.py` at the `forward_one_step` function. In the function, we have this piece of code
```python
if torch.argmax(y).cpu() == 50257:
    print('end')
```
2. The condition above will indicate that the inference process produce the EOT, and we can print the attention map output by running this code on the debug console:
```python
#### self attention
# Define your custom color scale
import os
import plotly.express as px
import plotly.io as pio
from transformers import WhisperTokenizer
color_scale = [
    [0.0, 'rgb(255, 250, 252)'],      
    [0.25, 'rgb(255, 235, 242)'],            
    [0.5, 'rgb(86, 182, 245)'],      
    [0.75, 'rgb(0, 132, 219)'],      
    [1.0, 'rgb(1, 98, 162)']         
]
tkn = WhisperTokenizer.from_pretrained("openai/whisper-small", language="id", task="transcribe")
token_list = tkn.convert_ids_to_tokens(tgt[0].cpu())
token_list = [tkn.convert_tokens_to_string(x) for x in token_list]
# token_list[0]='<|sot|>'
# token_list[3]='<|tran|>'
# token_list[4]='<|nots|>'
# token_list[7]='<|blnk|>'
# token_list[10]='<|blnk|>'
# token_list[12]='<|blnk|>'
# Create a directory for the heads
base_dir = 'temp' # put the directory path where you want to store the attention map here
os.makedirs(base_dir, exist_ok=True)
# Per head
for layer, scores in enumerate(attention_scores):
    # Create a directory for the layer
    layer_dir = os.path.join(base_dir, f'layer_{layer + 1}')
    os.makedirs(layer_dir, exist_ok=True)

    for head in range(scores.shape[1]):
        fig = px.imshow(scores[0][head].numpy(), color_continuous_scale=color_scale)
        fig.update_layout(
            coloraxis_showscale=True,  # Hide the color scale
            autosize=False,  # Turn off autosizing
            width=1080,  # Set the width of the plot
            xaxis=dict(tickmode='array', tickvals=list(range(len(token_list))), ticktext=token_list),
            yaxis=dict(tickmode='array', tickvals=list(range(len(token_list))), ticktext=token_list),
            font=dict(family="思源宋體-bold", size=60),
            height=1080  # Set the height of the plot

        )
        filename = f'layer{layer + 1}_head{head + 1}.png'
        filepath = os.path.join(layer_dir, filename)
        pio.write_image(fig, filepath)
```
3. The code above will take the attention map output of each head in every layer and save the picture on the directory `base_dir`.
4. The attention map can be opened in the `base_dir` directory.

