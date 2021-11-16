# TDPP_new
环境配置：
python 3
pytorch
faiss

数据集：
Amazon:- http://jmcauley.ucsd.edu/data/amazon/index.html  filter_size=20, filter_                                                                                                     len=20
Taobao:- https://tianchi.aliyun.com/dataset/dataDetail?dataId=649&userId=1   buy数据，filter_size=15, filter_len=8 ==item  19298 user 3447
item 7140
filter_size=15, filter_len=10

- Two preprocessed datasets can be downloaded through:
  - Tsinghua Cloud: https://cloud.tsinghua.edu.cn/f/e5c4211255bc40cba828/?dl=1
  - Dropbox: https://www.dropbox.com/s/m41kahhhx0a5z0u/data.tar.gz?dl=1
  
  

数据处理
1）data_iterator:
训练集 ：随机选batchsize个，随机切分用户序列，最后一个是target
测试集，验证集：按顺序选batchsize个，target item是后topk个
1）兴趣标签的学习：gru--kmeans:gru_a :用gru训练后用kmeans聚类后存储到对应文件夹下的 item_interest
调用训练 train_interst



做的实验：
一个dnn: DNN
一个gru：gru
loss的尝试在 gru_distance
多个gru的尝试在：multiGRU

组件：
GRUCell:  IGRU部分的实现
cont_time_cell   nhp: lstmtpp部分实现


model_interst: 用IGRU学习用户的多兴趣表达


训练：
train_nhp : 训练lstmtpp，得到概率值 lamda.pkl文件夏下
train:  1)先训练多兴趣，保存最好的模型，然后训练tpp（train_tpp）得到每个用户对应的兴趣概率，最后通过dpp输出
