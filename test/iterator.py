# coding=utf-8
##测试函数






#带yield的函数是一个生成器，而不是一个函数了，这个生成器有一个函数就是next函数，next就相当于“下一步”生成哪个数，这一次的next开始的地方是接着上一次的next停止的地方执行的，
# 所以调用next的时候，生成器并不会从foo函数的开始执行，只是接着上一步停止的地方开始，然后遇到yield后，return出要生成的数，此步就结束。
# import numpy as np
# def add(s, x):
#     return s + x
#
#
# def gen():
#     for i in range(4):
#         yield i
#
#
# base = gen()   #调用gen()并没有真实执行函数，而是只是返回了一个生成器对象
# #执行第一次a.next()时，才真正执行函数，执行到yield一个返回值，然后就会挂起，保持当前的名字空间等状态。然后等待下一次的调用,从yield的下一行继续执行。
# for n in [1, 10]:  #n是1 10
#     base = (add(i, n) for i in base)
#
# print(list(base))  #[20, 21, 22, 23]
#
# '''
# W_train = 0.8
# B_train = np.array([[1,2,3],[1,2,3]])
# B_train = W_train * B_train
# print("fewf")
# '''
# B_train = np.random.randn(2, 3) # 返回列表item的embedding 矩阵
# B_train /= np.linalg.norm(B_train, axis=1, keepdims=True)
# print("fewf")