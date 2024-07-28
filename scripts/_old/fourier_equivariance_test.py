import torch
from math import cos, sin, pi


X = torch.randn(1000,2,1)

phi, r = torch.atan2(X[:,0],X[:,1])[:,None], X.norm(dim=1)[:,None]
W = torch.randn(16,1)


emb_so2 = torch.cat(((phi).cos(), (phi).sin()),dim=1)[...,0].T
emb = torch.cat(((X).cos(),
                 (X).sin()),dim=1)[...,0].T

theta = 1.509126391
R = torch.tensor([[cos(theta), -sin(theta)],[sin(theta), cos(theta)]])
t = torch.randn(1,2,1)


rot_X = R@X #R@X
phi, r = torch.atan2(rot_X[:,0],rot_X[:,1])[:,None], rot_X.norm(dim=1)[:,None]
rot_emb_so2 = torch.cat(((phi).cos(), (phi).sin()),dim=1)[...,0].T
rot_emb = torch.cat(((rot_X).cos(),
                 (rot_X).sin()),dim=1)[...,0].T

print(rot_emb_so2.shape,emb_so2.shape)
C = rot_emb @ emb.T

U, S, Vt = torch.linalg.svd(C)
R = U @ Vt
print(R)
R_emb = R @ emb

C_so2 = rot_emb_so2 @ emb_so2.T
U, S, Vt = torch.linalg.svd(C_so2)
R = U @ Vt
print(R)
R_emb_so2 = R @ emb_so2



print(f"Error for {theta=} is: {torch.linalg.norm(rot_emb-R_emb).item()}")
print(f"Error for {theta=} is: {torch.linalg.norm(rot_emb_so2-R_emb_so2).item()}")
