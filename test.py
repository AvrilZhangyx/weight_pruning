# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:59:15 2019

@author: Admin
"""

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