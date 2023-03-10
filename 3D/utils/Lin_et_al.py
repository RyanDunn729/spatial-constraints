import numpy as np

def Lin_et_al_Method(sample_pts,mesh_pts,normals):

    num_pt = len(sample_pts)
    p = mesh_pts
    pt = sample_pts

    vec = np.zeros((num_pt,1,p.shape[0],3))
    norm_vec = np.zeros((num_pt,1,p.shape[0],3))
    
    # compute the distance btw ctr points and mesh points
    for i in range(num_pt):
        vec[i,:,:] = pt[i,:] -  p
    dis = np.linalg.norm(vec,axis=3)
    epsilon = 1e-8
    
    # normalize the vector
    norm_vec[:,:,:,0] = vec[:,:,:,0] / (dis+epsilon)
    norm_vec[:,:,:,1] = vec[:,:,:,1] / (dis+epsilon)
    norm_vec[:,:,:,2] = vec[:,:,:,2] / (dis+epsilon)

    # using the formulation of the anatomical constriaints
    distance = np.sum((vec)**2,axis=3)**0.125

    num_nodes = num_pt
    k = 1

    # vectorize the normals from mesh points
    normals_ = np.zeros((num_nodes,k,normals.shape[0],3))

    # Need to make sure the normals you get is facing inwards or outwards
    # Mine was facing outward originally, so I add the negative term here
    normals_[np.arange(num_nodes),:,:,:] = -normals
    
    # Compute the inner product
    # norm_vec is the vector of ctr backbone points
    # normals_ here is the normals you pre_computed from your mesh points 

    inner_product = np.einsum("ijkl,ijkl->ijk", norm_vec,normals_)
    
    # The signed distance function, which is based on the 
    # signed distance function array
    f = (-1*np.sign(inner_product)) * (distance) # rank 2

    signedfun = np.sum(np.sum(f,axis=2),axis=0)
    return signedfun.reshape(-1,1)

if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt

    Ns = 1000
    res = 20

    data = np.logspace(np.log10(4),np.log10(130000),res,dtype=int)
    time_data = np.zeros(res)
    for j,Ng in enumerate(data):
        mesh_pts = np.random.rand(Ng,3)
        normals = np.random.rand(Ng,3)
        for i in range(Ng):
            normals[i] = normals[i] / np.linalg.norm(normals[i])
        sample_pts = np.random.rand(Ns,3)

        t1 = time.perf_counter()
        f = Lin_et_al_Method(sample_pts,mesh_pts,normals)
        t2 = time.perf_counter()
        time_data[j] = t2-t1
        print(Ns,Ng,time_data[j])

    plt.loglog(data,time_data)
    plt.show()
