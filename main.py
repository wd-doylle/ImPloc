from extractor import img_res18_fv
from model import run


for fv_dim in [256]:
    for num_heads in [4]:
        for hid_dim in [64]:
            for num_layers in [4]:
                img_res18_fv.extract(fv_dim=fv_dim)
                run.transformer_bce("res18-%d"%fv_dim,fold=1,num_heads=num_heads,hid_dim=hid_dim,num_layers=num_layers)