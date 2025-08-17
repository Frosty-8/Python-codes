import torch 
import torch.nn.functional as F
from rich import print as rprint

x = torch.rand(1,3,4)
Q,K,V = x,x,x

scores = torch.matmul(Q,K.transpose(-2,-1))  / (4 ** 0.5)
weights = F.softmax(scores, dim=-1)
output = torch.matmul(weights, V)

rprint("Attention Weights:", weights)
rprint("\nOutput:", output)