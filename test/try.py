import numpy as np
import faiss
import torch
import torch.utils
import torch.utils.cpp_extension
import csv
import pandas as pd
import tensorflow as tf
import os
from model.utils import *

rnn = torch.nn.GRU(10, 20, 2)
input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
output, hn = rnn(input, h0)  ##2,3,20

rnn = torch.nn.GRUCell(10, 20)
input = torch.randn(6, 3, 10)
hx = torch.randn(3, 20)
output = []
for i in range(6):
        hx = rnn(input[i], hx)  ##3,20
        output.append(hx)

# item=torch.tensor([100,302 ,30, 412])
# a=torch.tensor([1,4 ,3, 4])
# b=torch.tensor([4, 4,4, 4])
# res=torch.eq(a,b)
# res1=res+0
# res2=item*res1
# print("fes")

# m = torch.nn.Linear(20, 30)
# input = torch.randn(128,3, 20)
# output = m(input)
# print(output.size())

# m = torch.nn.Softmax()
# input = torch.randn(2, 3)
# output = m(input)
# torch.Size([128, 30])


# _x = [[1,5,-0.4,-0.3],[1,5,-0.4,-0.3]]
# m=tf.maximum(0.0, _x)
# n=tf.minimum(0.0, _x)
# _alpha = tf.get_variable("prelu_", shape=4,
#                                  dtype=tf.float32, initializer=tf.constant_initializer(0.1))
# ll=m + _alpha * n
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     a=sess.run(ll)
#     print("good")


#a=torch.full([3],fill_value=0.1)

# _x= torch.tensor([[1,5,-0.4,-0.3],[1,5,-0.4,-0.3]])
# prelu(_x,device='cpu')
#
# gru_cell = tf.nn.rnn_cell.GRUCell(num_units=16)
# input = np.random.rand(4, 16)
# inputs = tf.constant(value=input, shape=(4, 16), dtype=tf.float32)
# h0 = gru_cell.zero_state(4, np.float32)
# output, h1 = gru_cell.__call__(inputs, h0)
# m=tf.equal(output,h1)
#
# rnn = torch.nn.GRUCell(10, 20)

# tensor1 = torch.tensor([[1,5,4,3],[1,5,3,2]])
# tensor2 = torch.tensor([[1,5],[1,5],[2,5],[2,5]])
# a=torch.matmul(tensor1, tensor2)  #20  65  16  55
#
#
# print(a)
#hx = rnn(input, hx)
# for i in range(6):
#     hx = rnn(input[i], hx)
#     output.append(hx)



#
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(m))


# data1 = {
#     "a": [1, 1, 3],
#     "b": [5,5,7],
#     "c": [7, 8, 9]
# }
# df = pd.DataFrame(data1)
# df1=df.groupby('a')['b']
#
# item_ca=df.groupby('a')['b'].apply(lambda x:str(x)).to_dict()  ##set
# item_ca_=df.groupby('a')['b'].apply(set).to_dict()
# print("well")pip install torch===1.7.1+cu110 torchvision===0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html -i https://pypi.douban.com/simple/



# print(torch.cuda.is_available())
# print(torch.version.cuda )
# print(torch.utils.cpp_extension.CUDA_HOME)
# a = torch.Tensor(5,3)
# a=a.cuda()
# print(a)


'''
test=[['0.21','0.41','0.51','0.11'],['0.21','0.41','0.51','0.11'],['0.21','0.41','0.51','0.11']]
test2=[['0.21','0.41','0.51','0.11'],['0.21','0.41','0.51','0.11'],['0.21','0.41','0.51','0.11']]
item_part_pad = test[:len(test)-1-1] + [[0] * 4] * (6 - len(test)+1+1)
hist_list = list(zip(test, test2))

conts='-15124565'
ts = int(conts)
if ts>1512571193 or ts<1490053988:
    print(ts)
'''

'''
# 文件头，一般就是数据名
fileHeader = ["name", "score"]

# 假设我们要写入的是以下两行数据
d1 = ["Wang", 100]
d2 = ["Li", 80]

# 写入数据
csvFile = open("../data/taobao_data/instance.csv", "w")
writer = csv.writer(csvFile)

# 写入的内容都是以列表的形式传入函数
#writer.writerow(fileHeader)
writer.writerow(d1)
writer.writerow(d2)

csvFile.close()

df = pd.read_csv("../data/taobao_data/instance.csv", header=None, names=["name", "score"])
sco=sorted(df['score'].unique().tolist())
print("good")
'''
'''
item_list=[(1,2,3),(1,2,4),(1,3,4)]
item_l=[]
for item, cate, timestamp in item_list:  ##但这也包含次数没有filter_size的item，后面写入的时候没录入
    if timestamp==4:
        item_l.append((item, cate, timestamp))
    #n = n + 1
print(item_l)
'''
# item_list=[[{'3':2,'4':2},{'3':2,'4':2},{'3':2,'4':2}],[{'3':2,'4':2},{'3':2,'4':2},{'3':2,'4':2}],[{'3':2,'4':2},{'3':2,'4':2},{'3':2,'4':2}]]
# item_=[item_list[i][-2:] for i in range(len(item_list))]
# print(item_)
# a=torch.rand(5,6)
# b=a[:,:3]
# print("good")
#
#
#
# print(torch.cuda.is_available())
# print(torch.version.cuda )
# print(torch.utils.cpp_extension.CUDA_HOME)
#
# d = 64                           # dimension
# nb = 100000                      # database size
# nq = 10000                       # nb of queries
# np.random.seed(1234)             # make reproducible
# xb = np.random.random((nb, 9)).astype('float32')
# xb[:, 0] += np.arange(nb) / 1000.
# xq = np.random.random((nq, d)).astype('float32')
# xq[:, 0] += np.arange(nq) / 1000.
# m=np.ones([64,9]).astype('float32')
#
# index = faiss.IndexFlatL2(9)   # build the index
# print(index.is_trained)
# index.add(m)                  # add vectors to the index
# print(index.ntotal)
#
# res = faiss.StandardGpuResources()
# #cpu_indax=faiss.ParameterSpace()
#
# flat_config = faiss.GpuIndexFlatConfig()
# flat_config.device = 0
# gpu_index = faiss.GpuIndexFlatIP(res, 9, flat_config)
# gpu_index.add(m)
'''
reduceSize= torch.nn.Linear(32, 16)
input = torch.randn(128, 32)
output = reduceSize(input)
print(output.size())
'''


'''
def expand(x, dim, N):
    return t.cat([x.unsqueeze(dim) for _ in range(N)],dim)
#zero_Emb=t.nn.Embedding(10, 17)
#M = expand(t.tanh(zero_Emb(t.normal(mean=0.0,std=t.tensor(1e-5)))),dim=0, N=5)
#x=t.normal(mean=0.0,std=t.tensor(1e-5))
xx=t.empty(10,17)
x=t.nn.init.normal_(xx,mean=0.0,std=t.tensor(1e-5))
res=expand(t.tanh(x),dim=0,N=5)
'''




# X = tf.random_normal(shape=[3, 5, 6], dtype=tf.float32)
# #X = tf.reshape(X, [-1, 5, 6])
# cell = tf.nn.rnn_cell.GRUCell(10)
# init_state = cell.zero_state(3, dtype=tf.float32)   #3,10
# # for i in range(5):
# #    output, state = cell(X[:,i,:],init_state)
# #    init_state=state
#
# with tf.Session() as sess:
#    sess.run(tf.initialize_all_variables())
#    print(sess.run(init_state))
#
#
# #    print(sess.run(output))
# #    print(sess.run(state))
# #
# # #X=t.nn.init.normal_(t.empty(3,5,6), mean=0.0, std=t.tensor(1e-5))
# # X = Variable(torch.randn(3, 5, 6))
# # cell = torch.nn.GRUCell(6, 10)
# # init_state=torch.FloatTensor(3,10).fill_(0)
# init_state=torch.nn.init.uniform(torch.empty(3,10))
# output = []
# for i in range(5):
#    state = cell(X[:,i,:],init_state)
#    init_state = state
#    output.append(state)

'''
def create_linear_initializer(input_size, dtype=tf.float32):
   stddev = 1.0 / np.sqrt(input_size)
   # 截断的正态分布中输出随机值，如果生成的值大于平均值2个标准偏差的值则丢弃重新选择. stddev标准偏差 dtype 数据类型
   return tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)
o2p_initializer = create_linear_initializer(5)  #线性随机初始化
X=tf.random_normal(shape=[3, 6], dtype=tf.float32)
parameters = tf.contrib.layers.fully_connected(
                X, 10, activation_fn=None,
                weights_initializer=o2p_initializer)


def create_linear_initializer(input_size):
   stddev = 1.0 / np.sqrt(input_size)
   # 截断的正态分布中输出随机值，如果生成的值大于平均值2个标准偏差的值则丢弃重新选择. stddev标准偏差 dtype 数据类型
   # return tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)
   return stddev
o2p_initializer = create_linear_initializer(5)
def init_weight(par, output):
   l = t.nn.Linear(par.size()[-1], output)
   t.nn.init.normal(l.weight, std=o2p_initializer)
   return l(par)
X = Variable(t.randn(3, 6))
parameters =init_weight(X,10)   #(3,10)
print("good")
'''