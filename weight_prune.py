# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 16:11:04 2019

@author: Admin
"""
import numpy as np
def weight_prune(model,pruning_perc):
    all_weights=[]
    for p in model.parameters():
        if len(p.data.size())==4:
            all_weights+=list(p.cpu().data.abs().numpy().flatten())
    threshold=np.percentile(np.array(all_weights),pruning_perc)
    
    masks=[]
    for p in model.parameters():
        if len(p.data.size())==4:
            pruned_index=p.data.abs()>threshold
            masks.append(pruned_index.float())
    return masks