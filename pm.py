'''
The stage fusion programming model implemented in software

The high-level abstraction of specific HGNN models

Created by Dengke Han, on 2023/10/30
'''

import numpy as np
import queue as q

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

from utils import load_acm,load_imdb,load_dblp

import time
import concurrent.futures
import ipdb

import multiprocessing


# to load the pre-trained parameters
def load_parameters():
    global dataset
    public_path="./parameters/"+dataset+"/"
    
    fpw_path=public_path+"fp_weight.txt"
    al_path=public_path+"a_l.txt"
    ar_path=public_path+"a_r.txt"
    lsfw_path=public_path+"lsf_w.txt"
    lsfb_path=public_path+"lsf_b.txt"
    lsfq_path=public_path+"lsf_q.txt"
    
    fp_w=np.loadtxt(fpw_path,dtype=float)
    a_l=np.loadtxt(al_path,dtype=float)
    a_r=np.loadtxt(ar_path,dtype=float)
    lsf_w=np.loadtxt(lsfw_path,dtype=float)
    lsf_b=np.loadtxt(lsfb_path,dtype=float)
    lsf_q=np.loadtxt(lsfq_path,dtype=float)
    
    return fp_w, a_l, a_r, lsf_w, lsf_b, lsf_q
    

# the abstract execution phases(FP, NA, LSF, GSF)
# feature projection (FP)
def feature_projection(sg_id):
    global fp_task_list, na_task_list, node_raw_features, projected_features, attn_coefs_l, attn_coefs_r, fp_w, a_l, a_r, fp_bitmap, na_bitmap
    while not fp_task_list.empty():
        node_id=fp_task_list.get()
        
        node_projected_feature=fp_w @ node_raw_features[node_id]
        projected_features[node_id]=node_projected_feature
        
        fp_bitmap[node_id]=True
        
        attn_coefs_l[sg_id][node_id]=a_l[sg_id] @ node_projected_feature
        na_bitmap[sg_id][node_id][0]=True
        attn_coefs_r[sg_id][node_id]=a_r[sg_id] @ node_projected_feature
        na_bitmap[sg_id][node_id][1]=True

# neighbor aggregation (NA)
def neighbor_aggregation(sg_id):
    global fp_task_list, na_task_list, lsf_task_list, projected_features, attn_coefs_l, attn_coefs_r, attn_coefs_exp_sum, sg_node_degrees
    
    while not na_task_list.empty():
        edge=na_task_list.get()
        src_id=edge[0]
        dst_id=edge[1]
        # if the feature has been projected, then the attention coefficients have been computed, too
        if fp_bitmap[src_id] and fp_bitmap[dst_id]:
            # ipdb.set_trace()
            exp_attn_coef=np.exp(attn_coefs_l[sg_id][src_id])
            na_features[sg_id][dst_id]+=(projected_features[src_id]*exp_attn_coef)
            attn_coefs_exp_sum[sg_id][dst_id]+=exp_attn_coef
            sg_node_degrees[sg_id][dst_id]-=1
            if sg_node_degrees[sg_id][dst_id]==0:
                na_features[sg_id][dst_id]/=attn_coefs_exp_sum[sg_id][dst_id]
                lsf_task_list.put(dst_id)
            
        else:
            # if not projected, send to feature projection
            fp_task_list.put(src_id)
            fp_task_list.put(dst_id)
            na_send2fp_list.put(edge)

# local semantic fusion (LSF)
def local_semantic_fusion(sg_id):
    global lsf_task_list, na_features, lsf_w, lsf_b, lsf_q, lsf_nodes_counter, sg_real_dst_num
    while not lsf_task_list.empty():
        node_id=lsf_task_list.get()
        node_semantic_w=np.tanh(lsf_w @ na_features[sg_id][node_id] + lsf_b)
        node_semantic_w=lsf_q @ node_semantic_w
        semantic_w[sg_id]+=node_semantic_w
        lsf_nodes_counter[sg_id]-=1
        
        if lsf_nodes_counter[sg_id] == 0:
            semantic_w[sg_id]/=sg_real_dst_num[sg_id]
            gsf_task_list.put(sg_id)
        

# global semantic fusion (GSF)
def global_semantic_fusion():
    global gsf_task_list, sg_real_dsts, semantic_w, semantic_features, semantic_w_exp_sum, gsf_semantic_counter, fn_ready
    while not gsf_task_list.empty():
        sg_id=gsf_task_list.get()
        exp_semantic_w=np.exp(semantic_w[sg_id])
        for dst_id in sg_real_dsts[sg_id]:
            semantic_features[dst_id]+=(na_features[sg_id][dst_id]*exp_semantic_w)
        semantic_w_exp_sum+=exp_semantic_w
        gsf_semantic_counter-=1
        
        # if all the semanic graphs have been processed already
        # ipdb.set_trace()
        if gsf_semantic_counter==0:
            fn_ready=True

# the final stage (FN)
def final_stage():
    global fn_ready, semantic_features, semantic_w_exp_sum
    if fn_ready:
        semantic_features=semantic_features / semantic_w_exp_sum
        
    

if __name__=="__main__":
    
    import argparse as ap

    parser=ap.ArgumentParser(description='to execute the hgnn model with dataset specified')
    parser.add_argument("-dataset",type=str,default='acm',help="the dataset that is to be processed")
    args=parser.parse_args()

    dataset=args.dataset
    
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
    
    #------------------------------to get the necessary info about dataset / set hyper-parameters-----------------------
    raw_feature_dim=features.shape[1]
    num_nodes=features.shape[0]
    num_semantic_graphs=len(meta_paths)
    
    projected_features_dim=64
    #-------------------------------------------------------------------------------------------------------------------
    
    #--------------------------to build semantic graphs based on meta-paths----------------------
    meta_paths = list(tuple(meta_path) for meta_path in meta_paths)
    semantic_graphs={}
    for meta_path in meta_paths:
        semantic_graphs[meta_path]=dgl.metapath_reachable_graph(g, meta_path)
    #--------------------------------------------------------------------------------------------

    #----------set the self-defined parameters in stage-fusion programming model----------

    # the global data
    sg_src_types=[]
    sg_src_nums=[]
    sg_dst_nums=[]
    sg_dst_types=[]
    sg_edge_nums=[]
    sg_node_degrees=[]
    
    sg_real_dsts=[]
    sg_real_dst_num=[]
    
    node_raw_features=[]
    
    for i in range(num_semantic_graphs):
        cur_semantic_graph=semantic_graphs[meta_paths[i]]
        
        # the src/dst type and number
        sg_src_types.append(int(cur_semantic_graph.srctypes[0]))
        sg_dst_types.append(int(cur_semantic_graph.dsttypes[0]))
        
        sg_src_nums.append(cur_semantic_graph.number_of_src_nodes())
        sg_dst_nums.append(cur_semantic_graph.number_of_dst_nodes())        
        
        # the edges num of each semantic graph
        sg_edge_nums.append(cur_semantic_graph.num_edges())

        # the in-degree of each dst node in each semantic graph
        sg_node_degrees.append(np.array(cur_semantic_graph.in_degrees()))
    
    # the raw features of each nodes
    for i in range(features.shape[0]):
        node_raw_features.append(np.array(features[i]))

    # the intermediate parameters
    projected_features=np.zeros(shape=(num_nodes,projected_features_dim),dtype=float)
    attn_coefs_l=np.zeros(shape=(num_semantic_graphs,num_nodes),dtype=float)
    attn_coefs_r=np.zeros(shape=(num_semantic_graphs,num_nodes),dtype=float)
    attn_coefs_exp_sum=np.zeros(shape=(num_semantic_graphs,num_nodes),dtype=float)
    na_features=np.zeros(shape=(num_semantic_graphs,num_nodes,projected_features_dim),dtype=float)
    semantic_w=np.zeros(shape=(num_semantic_graphs),dtype=float)
    semantic_w_exp_sum=0
    semantic_features=np.zeros(shape=(num_nodes,projected_features_dim),dtype=float)

    # the global parameters
    fp_w, a_l, a_r, lsf_w, lsf_b, lsf_q=load_parameters()

    # the global task lists
    fp_task_list=q.Queue()
    na_task_list=q.Queue()
    na_send2fp_list=q.Queue()
    lsf_task_list=q.Queue()
    gsf_task_list=q.Queue()

    # the global control bitmaps
    fp_bitmap=np.zeros(num_nodes,dtype=bool)
    na_bitmap=np.zeros(shape=(num_semantic_graphs,num_nodes,2),dtype=bool)
    lsf_nodes_counter=[]
    gsf_semantic_counter=num_semantic_graphs
    fn_ready=False
    
    for sg_id in range(num_semantic_graphs):
        cur_semantic_graph=semantic_graphs[meta_paths[sg_id]]
        # the real dst nodes
        sg_real_dsts.append(list(set(cur_semantic_graph.edges()[1].tolist())))
        sg_real_dst_num.append(len(sg_real_dsts[sg_id]))

    lsf_nodes_counter=sg_real_dst_num.copy()
    #--------------------------------------------------------------------------------------
    
    # num_semantic_graphs=2
    exe_time_list=[]
    
    #--------------------------the HGNN execution process------------------------------------
    for sg_id in range(num_semantic_graphs):
        # the semantic graph which is being processed
        cur_semantic_graph=semantic_graphs[meta_paths[sg_id]]
        cur_src_type=sg_src_types[sg_id]
        cur_dst_type=sg_dst_types[sg_id]

        # to initialize the neighbor aggregation task list with the edges of this semantic graph
        # it's assumed that the ([0],[1]) represents (src, dst)
        for edge_pair in zip(cur_semantic_graph.edges()[0], cur_semantic_graph.edges()[1]):
            na_task_list.put((int(edge_pair[0]), int(edge_pair[1])))
            
        # to record the start time of the HGNN model
        start_time=time.time()


        # the concurrent execution
        with concurrent.futures.ThreadPoolExecutor() as executor:
            while (not fp_task_list.empty()) or (not na_task_list.empty()) or (not lsf_task_list.empty()) or (not gsf_task_list.empty()):
                
                # # to perform feature projection
                # future_fp=executor.submit(feature_projection(sg_id))
                # # to perform neighbor aggregation
                # future_na=executor.submit(neighbor_aggregation(sg_id))
                # # to perform local semantic fusion
                # future_lsf=executor.submit(local_semantic_fusion(sg_id))
                # # to perform global semantic fusion
                # future_gsf=executor.submit(global_semantic_fusion())
                # # to perform the final stage
                # future_fn=executor.submit(final_stage())
                
                hgnn=[feature_projection(sg_id), neighbor_aggregation(sg_id), local_semantic_fusion(sg_id), global_semantic_fusion(), final_stage()]
                # futures=[executor.submit(stage) for stage in hgnn]
                # concurrent.futures.wait(futures)
                
                processes=[]
                for stage in hgnn:
                    processes.append(multiprocessing.Process(target=stage))
                
                for process in processes:
                    process.start()
                
                for process in processes:
                    process.join()
                
                # put the edges not been neighbor-aggregated this round into na_task_list again
                while not na_send2fp_list.empty():
                    na_task_list.put(na_send2fp_list.get())

        # the sequential execution
        # while (not fp_task_list.empty()) or (not na_task_list.empty()) or (not lsf_task_list.empty()) or (not gsf_task_list.empty()):
        #     # to perform feature projection
        #     feature_projection(sg_id)
        #     # to perform neighbor aggregation
        #     neighbor_aggregation(sg_id)
        #     # to perform local semantic fusion
        #     local_semantic_fusion(sg_id)
        #     # to perform global semantic fusion
        #     global_semantic_fusion()
        #     # to perform the final stage
        #     final_stage()
        #     # put the edges not been neighbor-aggregated this round into na_task_list again
        #     while not na_send2fp_list.empty():
        #         na_task_list.put(na_send2fp_list.get())  

        # to record the end time of the HGNN model
        end_time=time.time()
        exe_time_list.append(end_time-start_time)
    #--------------------------------------------------------------------------------------
    
    print(f"The execution time of inference process is")
    print(exe_time_list)
    print("s")
