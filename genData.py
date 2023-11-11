'''
Created by Dengke Han, on 2023/10/31
'''

# to generate random values of subgraphs and trained parameters acoording to node type
import numpy as np
import random
import argparse as ap

parser=ap.ArgumentParser(description='to generate random data for datasets as source data')
parser.add_argument("-dataset",type=str,default='random',help="the dataset that is to be processed")
parser.add_argument("-reorder",type=bool,default=False, help="whether to reorder the edges in csc format")
args=parser.parse_args()

dataset=args.dataset

data_path="./parameters/"+dataset+"/"

#to read the dataset configuration
def readConfig(fileName):
    data=[]
    with open(data_path+fileName) as f:
        for line in f:
            line=line.strip()
            if line.startswith("#") or line.startswith(' ') or line.startswith(';'):
                continue
            try:
                linedata=[int(i) for i in line.split()]
                data.append(linedata)
            except ValueError:
                pass
    return data

# to generate random edges according to the src type and dst type and the number of 
# edges of each subgraph
def gen_edges(subgid,src_type_N,dst_type_N,edgeN):
    src=[]
    dst=[]
    src_num=0
    dst_num=0
    with open(data_path+"sg_"+str(subgid)+"_edges.txt","w") as f:
        for i in range(edgeN):
            srcnode=random.sample(range(0,src_type_N),1)
            if not srcnode[0] in src:
                src.append(srcnode[0])
                src_num+=1
            dstnode=random.sample(range(0,dst_type_N),1)
            if not dstnode[0] in dst:
                dst.append(dstnode[0])
                dst_num+=1
            line=str(srcnode[0])+" "+str(dstnode[0])+"\n"
            f.write(line)
    return src_num,dst_num

# to generate the node features according to its type and number, and dimension
def gen_node_feats(nodeT,nodeN,featD):
    with open(data_path+"nodeT_"+str(nodeT)+"_raw_features.txt","w") as f:
        for i in range(featD):
            numbers=[round(random.uniform(0,1),2) for i in range(nodeN)]
            line=' '.join(str(num) for num in numbers)
            # print(line)
            f.write(line+'\n')

# to generate projection matrix for each type of node
def gen_projct_matrix(nodeT,rawDim,prjDim):
    with open(data_path+"nodeT_"+str(nodeT)+"_projection_matrix.txt","w") as f:
        for i in range(prjDim):
            numbers=[round(random.uniform(-0.01,0.03),2) for i in range(rawDim)]
            line=' '.join(str(num) for num in numbers)
            # print(line)
            f.write(line+'\n')

#to generate trained multi-head attention vectors
def gen_multihead_src_attn_matrix(sg_id,groupN,prjDim):
    with open(data_path+"sg_"+str(sg_id)+"_src_atten_matrix.txt","w") as f:
        for i in range(groupN):
            numbers=[round(random.uniform(0,0.01),3) for i in range(prjDim)]
            line=' '.join(str(num) for num in numbers)
            # print(line)
            f.write(line+'\n')

def gen_multihead_dst_attn_matrix(sg_id,groupN,prjDim):
    with open(data_path+"sg_"+str(sg_id)+"_dst_atten_matrix.txt","w") as f:
        for i in range(groupN):
            numbers=[round(random.uniform(0,0.01),3) for i in range(prjDim)]
            line=' '.join(str(num) for num in numbers)
            # print(line)
            f.write(line+'\n')

# W
def gen_semantic_weight_matrix(prjDim,concatDim):
    with open(data_path+"semantic_weight_matrix.txt","w") as f:
        for i in range(prjDim):
            numbers=[round(random.uniform(-0.01,0.03),3) for i in range(concatDim)]
            line=' '.join(str(num) for num in numbers)
            # print(line)
            f.write(line+'\n')

# q
def gen_semantic_attention_vec(prjDim):
    with open(data_path+"semantic_atten_vec.txt","w") as f:
        numbers=[round(random.uniform(0,0.01),3) for i in range(prjDim)]
        line=' '.join(str(num) for num in numbers)
        # print(line)
        f.write(line+'\n')


datacfg=readConfig(dataset+".ini")
# print(datacfg)

# to extract node type infos
node_types=datacfg[0][0]
per_type_node_nums=datacfg[1]
per_type_node_dims=datacfg[2]

# to generate the public datas, including node features of each type,
# attention mechanism related parameters
for i in range(node_types):
    # gen features
    gen_node_feats(i,per_type_node_nums[i],per_type_node_dims[i])
    # gen projection matrix
    gen_projct_matrix(i,per_type_node_dims[i],64)

gen_semantic_weight_matrix(64,64*8)
gen_semantic_attention_vec(64)

# to extract subgraph infos
sg_num=datacfg[3][0]
per_sg_src_node_type=datacfg[4]
per_sg_dst_node_type=datacfg[5]
per_sg_edges=datacfg[6]

per_sg_src_node_num=[]
per_sg_dst_node_num=[]
# to identify the type of src/dst nodes of each subgraph
for i in range(sg_num):
    src_type=per_sg_src_node_type[i]
    dst_type=per_sg_dst_node_type[i]
    # gen edges
    src_num,dst_num=gen_edges(i,per_type_node_nums[src_type],per_type_node_nums[dst_type],per_sg_edges[i])
    per_sg_src_node_num.append(src_num)
    per_sg_dst_node_num.append(dst_num)
    
    gen_multihead_src_attn_matrix(i,8,64)
    gen_multihead_dst_attn_matrix(i,8,64)
# to count src / dst num for each subgraph

print(per_sg_src_node_num)
print(per_sg_dst_node_num)

with open(data_path+dataset+'.ini','r') as f:
    cfg=f.readlines()

per_sg_src_node_num_str=list(map(str, per_sg_src_node_num))
per_sg_src_node_num_str = ' '.join(per_sg_src_node_num_str)

per_sg_dst_node_num_str=list(map(str, per_sg_dst_node_num))
per_sg_dst_node_num_str = ' '.join(per_sg_dst_node_num_str)

cfg.insert(5,per_sg_src_node_num_str+'\n')
cfg.insert(7,per_sg_dst_node_num_str+'\n')
cfg.insert(14,'# number of src node for each subgraph: {}\n'.format(per_sg_src_node_num_str))
cfg.insert(16,'# number of dst node for each subgraph: {}\n'.format(per_sg_dst_node_num_str))
with open(data_path+dataset+'.ini','w') as f:
    f.writelines(cfg)