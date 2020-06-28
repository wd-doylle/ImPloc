from extractor import img_res18_fv
from extractor import ensg_res18_fv
from extractor import ensg_resnext_fv
from model import run
import sys

# for fv in ['img-resnet18-256']:
#     for num_heads in [4]:
#         for hid_dim in [64]:
#             if fv.startswith('img'):
#                 img_res18_fv.extract(fv_dim=int(fv.split('-')[-1]))
#             elif fv.startswith('ensg-resnet'):
#                 ensg_res18_fv.extract(fv_dim=int(fv.split('-')[-1]))
#             elif fv.startswith('ensg-resnext'):
#                 ensg_resnext_fv.extract(fv_dim=int(fv.split('-')[-1]))
#             run.transformer_bce(fv,fold=1,num_heads=num_heads,hid_dim=hid_dim)

fv = 'img-resnet18-256'
num_heads = 4
num_layers = 4
hid_dim = 64
img_res18_fv.extract(fv_dim=int(fv.split('-')[-1]),stage='train')
img_res18_fv.extract(fv_dim=int(fv.split('-')[-1]),stage='test')
model_path = run.transformer_bce(fv,fold=1,num_heads=num_heads,hid_dim=hid_dim,num_layers=num_layers)
# model_path = run.GAT_bce(fv,fold=1,num_heads=num_heads,hid_dim=hid_dim)
run.transformer_predict(fv,model_path,num_heads=num_heads,hid_dim=hid_dim)