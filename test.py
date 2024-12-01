import torch
import graphpy
import numpy as np

x = np.array([0,2,4,6,8], dtype=np.int32)
y = np.array([1,3,0,1,1,3,0,1], dtype=np.int32)
z = np.array([2,2,2,2], dtype=np.int32)
a = graphpy.init_graph(x, y, z)
a.print_graph()
print(a.get_vcount())
print(a.get_edge_count())
print(a.get_nebrs()[0])