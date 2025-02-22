import torch
import json
from models.rnnvqe_last import DeepVQES

checkpoint_path = "/home/node25_tmpdata/xcli/percepnet/upload_scripts_2024/checkpoints/rnnvqe_last.pt.tar"
cpt = torch.load(checkpoint_path, map_location="cpu")
model = DeepVQES(in_dim=112, out_dim=100, casual=True, bidirectional=False)
state_dict = cpt['model_state_dict']
# 去掉 'module.' 前缀
new_state_dict = {}
for key, value in state_dict.items():
    # 只去掉 'module.' 前缀，但保留 'rnnoise_module.' 前缀中的 'module'
    if key.startswith('module.') and not key.startswith('rnnoise_module.'):
        new_key = key.replace('module.', '', 1)  # 只替换第一个 'module.'
    else:
        new_key = key  # 如果不符合条件则不修改
    new_state_dict[new_key] = value

model.load_state_dict(new_state_dict, strict=False)
cpt = {
    "model_state_dict": model.state_dict()
}
torch.save(cpt, "./rnnvqe_last.pt.tar")