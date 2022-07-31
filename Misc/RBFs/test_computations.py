import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time

nodes = np.vstack(([1,1,1],[0,0,0],[-1,-1,-1]))
num_RBFs = len(nodes)

num_RBFs = 2000
ang = np.linspace(0,2*np.pi,num_RBFs)
nodes = np.column_stack((3*np.cos(ang),np.sin(ang),ang))

res = 50
num_pts = res**3
rng = 2
pt = np.empty((num_pts,3))
pt[:,0] = np.einsum('i,j,k->ijk', np.linspace(-rng,rng,res), np.ones(res), np.ones(res)).flatten()
pt[:,1] = np.einsum('i,j,k->ijk', np.ones(res), np.linspace(-rng,rng,res), np.ones(res)).flatten()
pt[:,2] = np.einsum('i,j,k->ijk', np.ones(res), np.ones(res), np.linspace(-rng,rng,res)).flatten()

# for _ in range (5):
#     pt_expand = pt.reshape(num_pts,1,3)
#     node_expand = nodes.reshape(1,num_RBFs,3)
#     norm = np.linalg.norm(pt_expand-node_expand, axis=2)
#     phi = np.sum(np.exp(- (norm**2) / (r**2)),axis=1)

r = 1.5

# pt_expand = pt.reshape(num_pts,1,3)
# node_expand = nodes.reshape(1,num_RBFs,3)
# diff = pt_expand - node_expand
# norm = np.linalg.norm(diff, axis=2)
# phi = np.sum(np.exp(- (norm**2) / (r**2)),axis=1)

# diff = pt.reshape(num_pts,1,3) - nodes.reshape(1,num_RBFs,3)
# norm2 = np.einsum('ijk,ijk->ij', diff,diff)
# phi = np.sum(np.exp(- (norm2) / (r**2)),axis=1)

# pt_expand = pt.reshape(num_pts,1,3)
# node_expand = nodes.reshape(1,num_RBFs,3)
# diff = pt_expand-node_expand
# norm2 = np.einsum('ijk,ijk->ij', diff,diff)
# RBF_eval = np.exp(- (norm2) / (r**2) )
# phi = np.sum(RBF_eval,axis=1)

phi = np.empty(num_pts)
for k,i_pt in enumerate(pt):
    norm = np.linalg.norm(i_pt-nodes, axis=1)
    phi[k] = np.sum(np.exp(- (norm**2) / (r**2)))

# z = np.empty(num_pts)
# for k,i_pt in enumerate(pt):
#     norm2 = np.einsum('ij,ij->j',i_pt-nodes, i_pt-nodes)**0.5
#     z[k] = np.sum(np.exp( -(norm2)/(r**2) ))

# phi = np.empty(num_pts)
# for k,i_pt in enumerate(pt):
#     diff = np.transpose(i_pt-nodes)
#     norm2 = np.einsum('ij,ij->j',diff, diff)
#     phi[k] = np.sum(np.exp( -(norm2)/(r**2) ))



print(phi-z)
print('Num_pts: ', num_pts)
print('Num_RBFs: ', num_RBFs)
exit()

## Doesnt work yet ##
# var = 2/(-r**2) * np.exp( -(norm**2)/(r**2) )
# diff = pt_expand-node_expand
# d = np.sum(diff*var,axis=1)
# print(d)
######

deriv_xyz = np.zeros((num_pts,3))
deriv_RBF = np.empty((num_pts,3*num_RBFs))
for k,i_pt in enumerate(pt):
    # The euclidean norm between point and each RBF
    norm = np.linalg.norm(i_pt-nodes, axis=1)

    # Derivative w.r.t. x-y-z evaluations
    var = 2/(-r**2) * np.exp( -(norm**2)/(r**2) )
    diff = np.transpose(i_pt-nodes) # each row relates to (x-y-z)
    deriv_xyz[k,:] = np.sum(diff*var,axis=1)
    # Derivatives w.r.t. moving RBFs
    diff = np.transpose(i_pt-nodes)
    norm = np.linalg.norm(diff,axis=0)
    var = 2/(r**2) * np.exp( -(norm**2)/(r**2) )
    deriv_RBF[k,:] = (diff*var).transpose().flatten()

exit()
sns.set()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
img = ax.scatter(pt[:,0],pt[:,1],pt[:,2],c=phi,cmap=plt.hot())
fig.colorbar(img)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()