# Taiwan Accent Mandarin English Code-Switching

## Environments
- python version: `3.9.16 (main, Dec  7 2022, 01:11:51)  [GCC 9.4.0]`
- espnet version: `espnet 202301`
- pytorch version: `pytorch 1.13.1+cu117`

# Installation
Basically follow the installation process of espnet following `https://espnet.github.io/espnet/installation.html`

Based on the ESPnet library and Whisper library, we modify the code to build our model.

Step by step installation
1. Add deadsnake repo `add-apt-repository -y 'ppa:deadsnakes/ppa'`
2. Install python3.9 `apt install python3.9 python3.9-venv python3.9-dev`
3. Create python3.9 environment `python3.9 -m venv env39`
4. Activate the environment `source env39/bin/activate`
5. Go to tools directory and run `rm -f activate_python.sh && touch activate_python.sh`
5. Go to tools directory and install the espnet by `make TH_VERSION=1.13.1 CUDA_VERSION=11.7`
6. Install transformers tools by run `installers/install_transformers.sh`
6. Go to whisper directory `cd ../whisper` and then install the whisper library by `pip install -e .`


# Inference Model
We can utilize the code `whisper_check.py` in the code_util folder. But first we need to download the model weights from `https://mllab.asuscomm.com/s/L6oowFsT6ApsSHt`.

Make sure to modify the path for the model, config, and the audio file path.

# Head Selection process
Follow the instructions at `/code_util/head_selection.md` or just utilize the `/espnet/egs2/seame/asr1/attention_count_whispernoft_new.pkl` as the head selection result.

# Training Process
Example of training process utilizing SEAME Recipe

First make sure to put the dataset in the correct folder path, for SEAME, put the data under the `seame` folder

1. Run the `run.sh` to do the data preprocessing
2. Run the `run_whisper1ststage.sh` to run the 1st stage training
2. Run the `run_whisper2ndstage.sh` to run the 2nd stage training, make sure that the pretrained weight path is correct

# Print Attention Map 
Follow the instructions at `/code_util/attention_map.md`.

