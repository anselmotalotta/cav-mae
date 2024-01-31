import torch,timm
from models import CAVMAE

assert timm.__version__ == '0.4.5' # it is important to have right version of timm

model_path = '/storage/models/cavmae/audio_model.21.pth'
# CAV-MAE model with decoder
audio_model = CAVMAE(audio_length=1024,  # all models trained with 10s audio
                     modality_specific_depth=11,  # all models trained with 11 modality-specific layers and 1 shared layer
                     norm_pix_loss=True, tr_pos=False) # most models are trained with pixel normalization and non-trainabe positional embedding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mdl_weight = torch.load(model_path, map_location=device)
audio_model = torch.nn.DataParallel(audio_model) # it is important to convert the model to dataparallel object as all weights are saved in dataparallel format (i.e., in module.xxx)
miss, unexpected = audio_model.load_state_dict(mdl_weight, strict=False)
print(miss, unexpected) # check if all weights are correctly loaded