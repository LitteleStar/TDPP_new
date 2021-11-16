# coding=UTF-8
import csv
import pandas as pd
import pickle

RAW_DATA_FILE = '../data/taobao_data/UserBehavior.csv'
def to_df(file_name):
    df = pd.read_csv(RAW_DATA_FILE, header=None, names=['uid', 'iid', 'cid', 'btag', 'time'])
    return df



Num=100000   #生成user和ad是1000，另两个是100000

def read_write(file_path1,file_path2):
    # 从path1读取前num行到path2
    i = 1
    with open(file_path1) as f:
        csv_read = csv.reader(f)
        with open(file_path2,'w',newline='') as f_w:
            f_csv = csv.writer(f_w)
            for content in csv_read:
                # print(content)
                f_csv.writerow(content)
                i=i+1
                if i>Num:
                    break

def test_match(file1,file2):
    user = pd.read_csv(file1)
    test = pd.read_csv(file2)
    i=0
    print(len(user.userid.unique()))
    for test_user in test['user']:
        # print(user1)
        if test_user in user.userid.unique():
            i=i+1
            print(test_user)
    print("repeat user ",i)  #i=2

def to_csv(file1,file2):
    # f = open(file1,'rb')
    # data = pickle.load(f)
    dataDict=pd.read_pickle(file1)
    datatype=type(dataDict)
    print(str(datatype))
    if str(datatype)=='<class \'dict\'>':
        k=list(dataDict.keys())
        v=list(dataDict.values())
        # 使用dataframe(dict) 构建dataframe时，每个key会变成一个column，
        # list-like values会变为行，每个values中的list长度不一致就会出现错误
        # 所以讲行列互换就行了
        data=pd.DataFrame(list(zip(k,v)),columns=['k','v'])
        data.to_csv(file2)
    elif str(datatype)=='<class \'list\'>':
        with open(file2, 'w', newline='') as csvfile:
            writer  = csv.writer(csvfile)
            for row in dataDict:
                writer.writerow(row)
    else:
        print(datatype)

ad_feature='/home/ubuntu/jx/Data/TaobaoData/ad_feature.csv'
new_ad_feature='/home/ubuntu/jx/Data/TaobaoData/ad_feature_'+str(Num)+'.csv'
behavior='/home/ubuntu/jx/Data/TaobaoData/behavior_log.csv'
new_behavior='/home/ubuntu/jx/Data/TaobaoData/behavior_log_'+str(Num)+'.csv'
raw_sample='/home/ubuntu/jx/Data/TaobaoData/raw_sample.csv'
new_raw_sample='/home/ubuntu/jx/Data/TaobaoData/raw_sample_'+str(Num)+'.csv'
user='/home/ubuntu/jx/Data/TaobaoData/user_profile.csv'
new_user='/home/ubuntu/jx/Data/TaobaoData/user_profile_'+str(Num)+'.csv'

