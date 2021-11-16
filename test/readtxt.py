#coding:utf8
import torch
import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
def readpkl(feature_file):
    with open(feature_file, 'rb') as handle:
        feature_num= pickle.load(handle, encoding='iso-8859-1')
    print(feature_num)

readpkl("../data/taobao_data/10000000/taobao_feature.pkl")

# def readtxt(f1):
#     n=0
#     with open(f1,"r") as f1:
#         for i in range(5):
#             line=f1.readline().strip()
#             print(line)
#
# f1 = "../data/taobao_data/taobao_train.txt"  #设置文件对象
#
# #readtxt(f1)
#
# # coding=utf-8
# def readtxtf(f1, f2, num):
#     n = num
#     '''
#     myfile = open(f1)
#     lines = len(myfile.readlines())
#     print(lines)
#     '''
#     with open(f1, "r") as f1:
#         with open(f2, "w") as f2:
#             for i in range(n):
#                 line = f1.readline().strip()
#                 f2.writelines(line + "\n")
#
#
# f1 = "../data/taobao_data/taobao_valid.txt"  # 设置文件对象  691456条数据
# f2 = "../data/taobao_data/taobao_valid_256.txt"
# readtxtf(f1, f2, 256)
# def cal(file):
#     total = len(open(file).readlines())
#     print(total)
#
# #cal("../data/taobao_data/UserBehavior.csv")
#
# def to_csv(file_name):
#     df = pd.read_csv(file_name, header=None, names=['uid', 'iid', 'cid', 'btag', 'time'],nrows=100)
#     #df = df.sample(frac=FRAC, replace=False, random_state=0)
#     time_df = df.sort_values(['time'])
#     #时间位数不是13的会报错
#     #saveVariableOnDisk(time_list, 'taobao_data/time')
#     time = time_df[2:df.shape[0] - 50]
#     #item_key = sorted(df['iid'].unique().tolist())
#     #item_key2 = sorted(time['iid'].unique().tolist())
#     time.to_csv('../data/taobao_data/test.csv',header=None)
#     new=pd.read_csv('../data/taobao_data/test.csv', header=None, names=['uid', 'iid', 'cid', 'btag', 'time'])
#     tem_key2 = sorted(new['iid'].unique().tolist())
#     print("good")
#to_csv('../data/taobao_data/UserBehavior.csv')
'''
def transfer(datafile, todatafile):
    with open(datafile, 'rb') as f:
        inf = pickle.load(f,encoding='iso-8859-1')
    
    inf = str(inf)
    tof = open(todatafile, 'w')
    tof.write(inf)
    

def w_interst(datafile, todatafile):
    with open(datafile, 'rb') as f:
        inf = pickle.load(f,encoding='iso-8859-1')
    batch=len(inf)/200  #每个用户的item_list长度为200
    interest=[]
    for i in inf:
        for num in range(batch):
            batch_interest=[]   #256*4----200*256
            for j in i:
                for m in j:
                    m = m.tolist()
                    m = m.index(max(m))
                    batch_interest.append(m)
            #batch_intest 转置 ----256*200
            interest.append(batch_interest)





#transfer('../data/pickles/w_taobao.pickle', '../data/pickles/ll.txt')


# 测试transfer('../data/pickles/r_citeseer.pickle','../data/pickles/ll.txt')



def expand(x, dim, N):
    return tf.concat([tf.expand_dims(x, dim) for _ in range(N)], axis=dim)

def learned_init(units):
    a=tf.contrib.layers.fully_connected(tf.ones([1, 1]), units,
                                      activation_fn=None, biases_initializer=None)  #(1,unit)
    print(a)
    return tf.squeeze(a)  #(units,)

def create_linear_initializer(input_size, dtype=tf.float32):
    stddev = 1.0 / np.sqrt(input_size)
    #截断的正态分布中输出随机值，如果生成的值大于平均值2个标准偏差的值则丢弃重新选择. stddev标准偏差 dtype 数据类型
    return tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)

x=tf.ones([1,1])
y=tf.ones([2,3])
#a=learned_init(2)

def create_linear_initializer(input_size, dtype=tf.float32):
    stddev = 1.0 / np.sqrt(input_size)
    #截断的正态分布中输出随机值，如果生成的值大于平均值2个标准偏差的值则丢弃重新选择. stddev标准偏差 dtype 数据类型
    return tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())


#pytorch
def lea(units):
    layer=torch.nn.Linear(1, units, bias=False)     #(1,units
    return torch.squeeze(layer(torch.ones([1,1])))


def expand_(x, dim, N):
    return torch.cat([x.unsqueeze(dim) for _ in range(N)],dim)

def creat(input_size):
    stddev = 1.0 / np.sqrt(input_size)
    return torch.normal(stddev=stddev)

b = torch.ones([2,3])
a=torch.ones([2,3])
#mm=lea(2)

print("test")

'''

