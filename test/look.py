def readtxt(f1):
    with open(f1,"r") as f1:
        m=f1.read()
        length=len(m.splitlines())
        print(length)  ##68354735 train  8514014 valid
readtxt('../data/taobao_data/taobao_valid.txt')


def readtxtf(f1, f2, num):
    n = num
    '''
    myfile = open(f1)
    lines = len(myfile.readlines())
    print(lines)
    '''
    with open(f1, "r") as f1:
        with open(f2, "w") as f2:
            for i in range(n):
                line = f1.readline().strip()
                f2.writelines(line + "\n")


f1 = "../data/taobao_data/taobao_train.txt"  # 设置文件对象  691456条数据
f2 = "../data/taobao_data/taobao_train_1024.txt"
#readtxtf(f1, f2, 1024)