'''
The stage fusion programming model implemented in software

The high-level abstraction of specific HGNN models, implement HAN in this File

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
    
    fp_w=np.loadtxt(fpw_path,dtype='double')
    a_l=np.loadtxt(al_path,dtype='double')
    a_r=np.loadtxt(ar_path,dtype='double')
    lsf_w=np.loadtxt(lsfw_path,dtype='double')
    lsf_b=np.loadtxt(lsfb_path,dtype='double')
    lsf_q=np.loadtxt(lsfq_path,dtype='double')
    
    return fp_w, a_l, a_r, lsf_w, lsf_b, lsf_q
    

# the abstract execution phases(FP, NA, LSF, GSF)
# feature projection (FP)
def feature_projection(sg_id):
    print("-------------processing FP Stage-------------")
    fp_start_time=time.time()
    global fp_task_list, na_task_list, lsf_task_list, gsf_task_list, node_raw_features, projected_features, attn_coefs_l, attn_coefs_r, fp_w, a_l, a_r, fp_bitmap, na_bitmap
    while (not fp_task_list.empty()) or (not na_task_list.empty()) or (not lsf_task_list.empty()) or (not gsf_task_list.empty()):
        if not fp_task_list.empty():
            node_id=fp_task_list.get()
            
            # ipdb.set_trace()
            node_projected_feature=fp_w @ node_raw_features[node_id]
            projected_features[node_id]=node_projected_feature
            
            fp_bitmap[node_id]=True
            
            attn_coefs_l[sg_id][node_id]=a_l[sg_id] @ node_projected_feature
            na_bitmap[sg_id][node_id][0]=True
            attn_coefs_r[sg_id][node_id]=a_r[sg_id] @ node_projected_feature
            na_bitmap[sg_id][node_id][1]=True
    fp_end_time=time.time()
    print(f"the execution time of FP stage is {fp_end_time-fp_start_time}")
    print("-------------finish FP Stage-------------")

# neighbor aggregation (NA)
def neighbor_aggregation(sg_id):
    global device, fp_task_list, na_task_list, lsf_task_list, gsf_task_list, projected_features, attn_coefs_l, attn_coefs_r, attn_coefs_exp_sum, sg_node_degrees, processed_edge_counter
    while (not fp_task_list.empty()) or (not na_task_list.empty()) or (not lsf_task_list.empty()) or (not gsf_task_list.empty()):
        if not na_task_list.empty():
            print("-------------processing NA Stage-------------")
            na_start_time=time.time()
            
            edge=na_task_list.get()
            src_id=edge[0]
            dst_id=edge[1]
            # if the feature has been projected, then the attention coefficients have been computed, too
            if fp_bitmap[src_id] and fp_bitmap[dst_id]:
                # ipdb.set_trace()
                if device=='cpu':
                    attn_coef=attn_coefs_l[sg_id][src_id]+attn_coefs_r[sg_id][dst_id]
                    exp_attn_coef=np.exp(attn_coef)
                else:
                    attn_coef=F.leaky_relu(attn_coefs_l[sg_id][src_id]+attn_coefs_r[sg_id][dst_id])
                    exp_attn_coef=torch.exp(attn_coef)
                na_features[sg_id][dst_id]+=(projected_features[src_id]*exp_attn_coef)
                
                processed_edge_counter+=1
                
                attn_coefs_exp_sum[sg_id][dst_id]+=exp_attn_coef
                sg_node_degrees[sg_id][dst_id]-=1
                if sg_node_degrees[sg_id][dst_id]==0:
                    na_features[sg_id][dst_id]/=attn_coefs_exp_sum[sg_id][dst_id]
                    lsf_task_list.put(dst_id)
                
            else:
                # if not projected, send to feature projection
                fp_task_list.put(src_id)
                fp_task_list.put(dst_id)
                # na_send2fp_list.put(edge)
                na_task_list.put(edge)
                
            na_end_time=time.time()
            print(f"the execution time of NA stage is {na_end_time-na_start_time}")
            print("-------------finish NA Stage-------------")

# local semantic fusion (LSF)
def local_semantic_fusion(sg_id):
    global fp_task_list, na_task_list, lsf_task_list, gsf_task_list, na_features, lsf_w, lsf_b, lsf_q, semantic_w, lsf_nodes_counter, sg_real_dst_num
    while (not fp_task_list.empty()) or (not na_task_list.empty()) or (not lsf_task_list.empty()) or (not gsf_task_list.empty()):
        if not lsf_task_list.empty():
            print("-------------processing LSF Stage-------------")
            lsf_start_time=time.time()
            
            node_id=lsf_task_list.get()
            if device=='cpu':
                node_semantic_w=np.tanh(lsf_w @ na_features[sg_id][node_id] + lsf_b)
            else:
                node_semantic_w=torch.tanh(lsf_w @ na_features[sg_id][node_id] + lsf_b)
            node_semantic_w=lsf_q @ node_semantic_w
            semantic_w[sg_id]+=node_semantic_w
            lsf_nodes_counter[sg_id]-=1
            
            if lsf_nodes_counter[sg_id] == 0:
                semantic_w[sg_id]/=sg_real_dst_num[sg_id]
                gsf_task_list.put(sg_id)
            
            lsf_end_time=time.time()
            print(f"the execution time of LSF stage is {lsf_end_time-lsf_start_time}")
            print("-------------finish LSF Stage-------------")

# global semantic fusion (GSF)
def global_semantic_fusion():
    global fp_task_list, na_task_list, lsf_task_list, gsf_task_list, sg_real_dsts, semantic_w, semantic_features, semantic_w_exp_sum, gsf_semantic_counter, fn_ready
    while (not fp_task_list.empty()) or (not na_task_list.empty()) or (not lsf_task_list.empty()) or (not gsf_task_list.empty()):
        if not gsf_task_list.empty():
            print("-------------processing GSF Stage-------------")
            gsf_start_time=time.time()
            
            sg_id=gsf_task_list.get()
            if device=='cpu':
                exp_semantic_w=np.exp(semantic_w[sg_id])
            else:
                exp_semantic_w=torch.exp(semantic_w[sg_id])
            for dst_id in sg_real_dsts[sg_id]:
                semantic_features[dst_id]+=(na_features[sg_id][dst_id]*exp_semantic_w)
            semantic_w_exp_sum+=exp_semantic_w
            gsf_semantic_counter-=1
            
            # if all the semanic graphs have been processed already
            # ipdb.set_trace()
            if gsf_semantic_counter==0:
                fn_ready=True
            
            gsf_end_time=time.time()
            print(f"the execution time of GSF stage is {gsf_end_time-gsf_start_time}")
            print("-------------finish GSF Stage-------------")
    
# the final stage (FN)
def final_stage():
    global fp_task_list, na_task_list, lsf_task_list, gsf_task_list, fn_ready, semantic_features, semantic_w_exp_sum
    while (not fp_task_list.empty()) or (not na_task_list.empty()) or (not lsf_task_list.empty()) or (not gsf_task_list.empty()):
        if fn_ready:
            print("-------------processing Final Stage-------------")
            final_start_time=time.time()
            semantic_features=semantic_features / semantic_w_exp_sum
            final_end_time=time.time()
            print(f"the execution time of Final stage is {final_end_time-final_start_time}")
            print("-------------finish Final Stage-------------")
    

if __name__=="__main__":
    
    import argparse as ap

    parser=ap.ArgumentParser(description='to execute the hgnn model with dataset specified')
    parser.add_argument("--dataset",type=str,default='imdb',help="the dataset that is to be processed")
    parser.add_argument('--device', type=str, default='cuda:0',help="the device that choose to use")
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
    projected_features=np.zeros(shape=(num_nodes,projected_features_dim),dtype='double')
    attn_coefs_l=np.zeros(shape=(num_semantic_graphs,num_nodes),dtype='double')
    attn_coefs_r=np.zeros(shape=(num_semantic_graphs,num_nodes),dtype='double')
    attn_coefs_exp_sum=np.zeros(shape=(num_semantic_graphs,num_nodes),dtype='double')
    na_features=np.zeros(shape=(num_semantic_graphs,num_nodes,projected_features_dim),dtype='double')
    semantic_w=np.zeros(shape=(num_semantic_graphs),dtype='double')
    semantic_w_exp_sum=0
    semantic_features=np.zeros(shape=(num_nodes,projected_features_dim),dtype='double')

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
    
    processed_edge_counter=0
    
    for sg_id in range(num_semantic_graphs):
        cur_semantic_graph=semantic_graphs[meta_paths[sg_id]]
        # the real dst nodes
        sg_real_dsts.append(list(set(cur_semantic_graph.edges()[1].tolist())))
        sg_real_dst_num.append(len(sg_real_dsts[sg_id]))

    lsf_nodes_counter=sg_real_dst_num.copy()
    
    #---------transform the np array into tensors and download to GPU for execution--------
    if device=='cuda:0':
        fp_w=torch.tensor(fp_w).double().to(device)
        node_raw_features=torch.tensor(np.array(node_raw_features)).double().to(device)
        projected_features=torch.tensor(projected_features).double().to(device)
        na_features=torch.tensor(na_features).double().to(device)
        semantic_w=torch.tensor(semantic_w).double().to(device)
        semantic_features=torch.tensor(semantic_features).double().to(device)
        
        a_l=torch.tensor(a_l).double().to(device)
        a_r=torch.tensor(a_r).double().to(device)
        attn_coefs_l=torch.tensor(attn_coefs_l).double().to(device)
        attn_coefs_r=torch.tensor(attn_coefs_r).double().to(device)
        attn_coefs_exp_sum=torch.tensor(attn_coefs_exp_sum).double().to(device)
        
        lsf_w=torch.tensor(lsf_w).double().to(device)
        lsf_b=torch.tensor(lsf_b).double().to(device)
        lsf_q=torch.tensor(lsf_q).double().to(device)
    #--------------------------------------------------------------------------------------
    
    #--------------------------------------------------------------------------------------
    print("**************************************************************")
    print('Current processing dataset: ', dataset)
    print("**************************************************************")
    # num_semantic_graphs=2
    exe_time_list=[]
    
    #--------------------------the HGNN execution process------------------------------------
    # for sg_id in range(num_semantic_graphs):
        # the semantic graph which is being processed
    sg_id=0
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
    # with concurrent.futures.ThreadPoolExecutor() as executor:
        # while (not fp_task_list.empty()) or (not na_task_list.empty()) or (not lsf_task_list.empty()) or (not gsf_task_list.empty()):
            
        # #--------------------the parallel execution based on multithread------------------
        # hgnn=[executor.submit(feature_projection(sg_id)), executor.submit(neighbor_aggregation(sg_id)), executor.submit(local_semantic_fusion(sg_id)), executor.submit(global_semantic_fusion()), executor.submit(final_stage())]
        # #----------------------------------------------------------------------------------
        # concurrent.features.wait(hgnn)
        
    # --------------------the parallel execution based on multiprocess------------------
    hgnn=[feature_projection(sg_id), neighbor_aggregation(sg_id), local_semantic_fusion(sg_id), global_semantic_fusion(), final_stage()]
    
    processes=[]
    for stage in hgnn:
        processes.append(multiprocessing.Process(target=stage))
    
    for process in processes:
        process.start()
    
    for process in processes:
        process.join()
        
        #put the edges not been neighbor-aggregated this round into na_task_list again
        # while not na_send2fp_list.empty():
        #     na_task_list.put(na_send2fp_list.get())
    # ----------------------------------------------------------------------------------

        # ------------------------------the sequential execution-------------------------------
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
        # -------------------------------------------------------------------------------------

        # to record the end time of the HGNN model
    end_time=time.time()
    exe_time_list.append(end_time-start_time)
    #--------------------------------------------------------------------------------------
    
    print("**************************************************************")
    print("The number of processed edges is {}".format(processed_edge_counter))
    print(f"stage-fusion PM: The execution time of inference process is {sum(exe_time_list)} s")
    print("**************************************************************")