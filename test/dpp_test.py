from no_need.dpp import *

import time

item_size = 5000
feature_dimension = 5000
max_length = 1000

scores = np.exp(0.01 * np.random.randn(item_size) + 0.2)   #返回一组采样样本 (item_size,)
feature_vectors = np.random.randn(item_size, feature_dimension)

feature_vectors /= np.linalg.norm(feature_vectors, axis=1, keepdims=True)
similarities = np.dot(feature_vectors, feature_vectors.T)  #矩阵乘法
kernel_matrix = scores.reshape((item_size, 1)) * similarities * scores.reshape((1, item_size))   #点乘，得到一个（item_size,item_size）

print('kernel matrix generated!')

t = time.time()
#max_length：表示最后选出的集合大小
result = dpp(kernel_matrix, max_length)
print('algorithm running time: ' + '\t' + "{0:.4e}".format(time.time() - t))

window_size = 10
t = time.time()
result_sw = dpp_sw(kernel_matrix, window_size, max_length)
print('sw algorithm running time: ' + '\t' + "{0:.4e}".format(time.time() - t))