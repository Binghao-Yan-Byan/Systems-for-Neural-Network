import torch
import graphpy
import numpy as np

x = np.array([1,2,3, 4], dtype=np.int32)
y = np.array([4,5,6, 7], dtype=np.int32)
z = np.array([7,8,9, 10], dtype=np.int32)
print(x)
a = graphpy.init_graph(x, y, z)
a.print_graph()
print(a.get_vcount())
print(a.get_edge_count())
