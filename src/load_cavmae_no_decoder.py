import torch,timm
from models import CAVMAEFT

assert timm.__version__ == '0.4.5' # it is important to have right version of timm

model_path = '/storage/models/cavmae/as-full-51.2.pth'
# CAV-MAE model without decoder

n_class = 527 # 527 for audioset finetuned models, 309 for vggsound finetuned models

audio_model = CAVMAEFT(label_dim=n_class, 
                     modality_specific_depth=11) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mdl_weight = torch.load(model_path, map_location=device)
audio_model = torch.nn.DataParallel(audio_model) # it is important to convert the model to dataparallel object as all weights are saved in dataparallel format (i.e., in module.xxx)
miss, unexpected = audio_model.load_state_dict(mdl_weight, strict=False)
# check if all weights are correctly loaded
print("miss", miss) 
print("unexpected", unexpected) 
