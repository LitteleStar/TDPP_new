import faiss
import numpy as np
import torch
import torch.utils
import torch.utils.cpp_extension

print(torch.cuda.is_available())
print(torch.version.cuda )
print(torch.utils.cpp_extension.CUDA_HOME)

d = 64                           # dimension
nb = 100000                      # database size
nq = 10000                       # nb of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.

index = faiss.IndexFlatL2(d)   # build the index
print(index.is_trained)
index.add(xb)                  # add vectors to the index
print(index.ntotal)

res = faiss.StandardGpuResources()
cpu_indax=faiss.ParameterSpace()

# flat_config = faiss.GpuIndexFlatConfig()
# flat_config.device = 0
# gpu_index = faiss.GpuIndexFlatIP(res, 9, flat_config)
# print("foo")