'''
The pipeline programming model, i.e. the coarse-grained stage fusion programming model

Created by Dengke Han, on 2023/11/08
'''

import numpy as np
import queue as q

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_scatter 

import dgl
from dgl.nn.pytorch import GATConv

from dgl import DGLGraph
from dgl.data import citation_graph as citegrh

from hanConv_dgl import HAN, HAN_freebase

from utils import load_acm,load_imdb,load_dblp

import time
import concurrent.futures
import ipdb

import multiprocessing


if __name__=="__main__":
    
    import argparse as ap

    parser=ap.ArgumentParser(description='to execute the hgnn model with dataset specified')
    parser.add_argument("--dataset",type=str,default='imdb',help="the dataset that is to be processed")
    parser.add_argument('--device', type=str, default='cuda:0')
    args=parser.parse_args()

    dataset=args.dataset
    device=args.device
    
    #----------------------------------------load original datasets------------------------------
    # main
    # to load heterogeneous graph
    if dataset=='acm':
        g, features, labels, num_classes, train_sg_id, val_sg_id, test_sg_id, train_mask, val_mask, test_mask, meta_paths = load_acm()
    elif dataset=='imdb':
        g, features, labels, num_classes, train_sg_id, val_sg_id, test_sg_id, train_mask, val_mask, test_mask, meta_paths = load_imdb()
    else:
        g, features, labels, num_classes, train_sg_id, val_sg_id, test_sg_id, train_mask, val_mask, test_mask, meta_paths = load_dblp()
    #--------------------------------------------------------------------------------------------
    
    g=g.to(device)
    features=features.to(device)
    labels=labels.to(device)
    
    if dataset=='acm' or dataset=='dblp':
        labels_onehot=F.one_hot(labels).T
    else:
        labels_onehot=labels.T.clone().detach()
        
    #---------------------set model parameters to initialize a dgl implementation for HAN model-------------------------
    hidden_size=64
    num_heads=[1]
    slope=0.2
    act_func=F.elu
    
    net = HAN(
        meta_paths=meta_paths,
        in_size=features.shape[1],
        hidden_size=64,
        out_size=num_classes,
        num_heads=num_heads,
        dropout=0.0).to(device)
    
    #to build semantic graphs based on meta-paths
    num_semantic_graphs=len(meta_paths)
    meta_paths = list(tuple(meta_path) for meta_path in meta_paths)
    semantic_graphs={}
    for meta_path in meta_paths:
        semantic_graphs[meta_path]=dgl.metapath_reachable_graph(g, meta_path)   

    # the raw features,shape[0]
    feat_copy=features.T.clone().detach()
    num_target_nodes=feat_copy.shape[1]

    fp_weight_list=[]
    attn_src_list=[]
    attn_dst_list=[]
    
    sf_weight=net.layers[0].semantic_attention.project[0].weight.clone().detach().to(device)
    sf_bias=net.layers[0].semantic_attention.project[0].bias.view(128,1).clone().detach().to(device)
    sf_q=net.layers[0].semantic_attention.project[2].weight.clone().detach().to(device)
    
    # to get the initial trainable parameters for DGL-implemented HAN model
    for i in range(num_semantic_graphs):
        tmp_weight=net.layers[0].gat_layers[i].fc.weight.clone().detach()
        tmp_attn_src=net.layers[0].gat_layers[i].attn_l.clone().detach().view(hidden_size,1)
        tmp_attn_dst=net.layers[0].gat_layers[i].attn_r.clone().detach().view(hidden_size,1)
        tmp_weight.requires_grad=True
        tmp_attn_src.requires_grad=True
        tmp_attn_dst.requires_grad=True
        
        tmp_weight=tmp_weight.to(device)
        tmp_attn_src=tmp_attn_src.to(device)
        tmp_attn_dst=tmp_attn_dst.to(device)
        
        fp_weight_list.append(tmp_weight)
        attn_src_list.append(tmp_attn_src)
        attn_dst_list.append(tmp_attn_dst) 
    #-------------------------------------------------------------------------------------------------------------------

    #--------------------------the pipeline programming model with no optimization at all--------------------
    print("**************************************************************")
    print('Current processing dataset: ', dataset)
    print("**************************************************************")
    
    pipeline_pm_start_time=time.time()
    
    # to store the semantic_embeddings generated from GAT models
    semantic_embeddings_unweighted_list = []
    semantic_w_unormalized_list=[]
    # to perform GAT Conv on each semantic graph 
    for i,meta_path in enumerate(meta_paths):
        print('Current processing semantic graph: ', meta_path)
        cur_g=semantic_graphs[meta_path]
        
        agg_base=torch.zeros(hidden_size,num_target_nodes).to(device)
        
        '''forward'''
        # 1.feature projection: Wh=x*WT
        # src / dst nodes for each edge
        edges_src=cur_g.edges()[0]
        edges_dst=cur_g.edges()[1]
        
        # to record dst nodes for each semantic graph
        # semantic_graph_dst_nodes_list.append(edges_dst.unique())

        #-----------------------------FP Stage-----------------------------
        Wh=fp_weight_list[i] @ feat_copy
        #------------------------------------------------------------------

        #------------------------------NA Stage----------------------------
        # 2.compute e: e=[Whl || Whr]*attnT = Whl*attn_srcT + Whr*attn_dstT
        e_l=(attn_src_list[i].T @ Wh).view(cur_g.num_nodes(),1)[edges_src]
        e_r=(attn_dst_list[i].T @ Wh).view(cur_g.num_nodes(),1)[edges_dst]
        e_sum=e_l + e_r

        # 3.compute alpha: alpha=softmax(LeakyReLU(e)),slope=0.2
        # alpha
        alpha_unormalized=F.leaky_relu(e_sum, slope)

        #inherent implementation of softmax
        alpha=F.softmax(alpha_unormalized,dim=0)

        # 4. weighted aggregation: z=sum(alpha * Whl)
        z = torch_scatter.scatter(alpha.T * (Wh.T[edges_src]).T, edges_dst, dim=-1, out=agg_base, reduce='sum')
        # 5. activation

        #for hidden layer, act_func is elu
        out=act_func(z)
        #-------------------------------------------------------------------
        semantic_embeddings_unweighted_list.append(out)
    semantic_embeddings_unweighted=torch.stack(semantic_embeddings_unweighted_list,dim=0)

    #------------------------------SF Stage----------------------------
    for i in range(num_semantic_graphs):
        sf_w0=sf_weight @ semantic_embeddings_unweighted[i] + sf_bias
        sf_w1=torch.tanh(sf_w0)
        sf_w2=sf_q @ sf_w1
        sf_w=(1/num_target_nodes) * sf_w2.sum(dim=-1)
        semantic_w_unormalized_list.append(sf_w)

    semantic_w_unnormalized=torch.stack(semantic_w_unormalized_list,dim=0).view(num_semantic_graphs,1)
    semantic_w=F.softmax(semantic_w_unnormalized,dim=0)

    semantic_embeddings=[]
    # weighted
    for i in range(num_semantic_graphs):
        semantic_embeddings.append(semantic_embeddings_unweighted[i] * semantic_w[i])
    semantic_embeddings=torch.stack(semantic_embeddings,dim=0)

    # aggregation
    final_embeddings=semantic_embeddings.sum(dim=0)
    #-------------------------------------------------------------------
    
    pipeline_pm_end_time=time.time()
    
    print("**************************************************************")
    print(f"pipeline PM: The execution time of inference process is {pipeline_pm_end_time-pipeline_pm_start_time} s")
    print("**************************************************************")
    #----------------------------------------------------------------------------------------------------