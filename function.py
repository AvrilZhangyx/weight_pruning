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

def train(model,loss_fn,optm,param,loader_train):
    model.train()
    for epoch in range(param['num_epochs']):
        print('Epoch %d/%d'%(epoch+1,param['num_epochs']))
        for t,(x,y) in enumerate(loader_train):
            scores=model(x)
            loss=loss_fn(scores,y)
            if(t+1)%100==0:
                print('t=%d,loss=%.4f'%(t+1,loss.data[0]))
            optm.zero_grad()
            loss.backward()
            optm.step() 
def test(model,loader):
    model.eval()
    num_correct,num_samples=0,len(loader.dataset)
    for x,y in loader:
        scores=model(x)
        _,preds=scores.data.cpu().max(1)
        num_correct+=(preds==y).sum()
    acc=float(num_correct)/num_samples
    print('Test accuracy:{:.2f}%({}/{})'.format(100.*acc,num_correct,num_samples))
    return acc