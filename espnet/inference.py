import torch
import yaml
from espnet2.bin.asr_inference import Speech2Text
import soundfile
import librosa
import librosa.display
from datetime import datetime
# model_path = r"/home/espnet/egs2/shangen/asr1/exp/asr_TL_att_ctc_new/valid.acc.ave.pth"
# config_path = r"/home/espnet/egs2/shangen/asr1/exp/asr_TL_att_ctc_new/config.yaml"
# lm_config_path = r"/home/espnet/egs2/shangen/asr1/exp/lm_train_lm_transformer_zh_char/config.yaml"
# lm_model_path = r"/home/espnet/egs2/shangen/asr1/exp/lm_train_lm_transformer_zh_char/valid.loss.ave.pth"

model_path = r"/home/espnet/tryhere/asr_TL_att_ctc_new/valid.acc.ave_10best.pth"
config_path = r"/home/espnet/tryhere/asr_TL_att_ctc_new/config.yaml"
lm_config_path = r"/home/espnet/tryhere/lm_train_lm_transformer_zh_char/config.yaml"
lm_model_path = r"/home/espnet/tryhere/lm_train_lm_transformer_zh_char/valid.loss.ave_10best.pth"
speech2text = Speech2Text(  asr_model_file=model_path,
                            asr_train_config=config_path,
                            # lm_train_config=lm_config_path,
                            # lm_file= lm_model_path,
                            device='cuda',
                            minlenratio=0.0,
                            maxlenratio=0.0,
                            lm_weight=0.3,
                            ctc_weight=0.6,
                            beam_size=2,  
                            batch_size=0                   
                          )

# speech2text = Speech2Text("EspASRModel/exp/asihell2/config.yaml", "EspASRModel/exp/asihell2/valid.acc.best.pth")
# print(speech2text)
# audio1, rate1 = soundfile.read("/home/espnet/shCPXGY1L_8_3.wav")
# print(audio1,rate1)
# audio2, rate2 = soundfile.read("/home/espnet/input_user.wav")
# print(audio2,rate2)
start=datetime.now()
audio3, rate3 = librosa.load('/home/espnet/not_clear.wav', sr=16000)
nbests = speech2text(audio3)
text, *_ = nbests[0]
print(text)
print ("Processing Time",datetime.now()-start)


