import numpy as np
import faiss 
from rich import print as rprint

data = np.random.random((5,4)).astype('float32')
index = faiss.IndexFlatL2(4)
index.add(data)

query = np.random.random((1,4)).astype('float32')
distances, indices = index.search(query, k=3)

rprint("Query Vector:", query)
rprint("Top 3 Nearest Indices:", indices)
rprint("Distances:", distances)