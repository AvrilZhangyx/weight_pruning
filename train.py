# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:59:14 2019

@author: Admin
"""

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