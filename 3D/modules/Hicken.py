import numpy as np
from scipy.spatial import KDTree
import time

def Exact_eval(pts,exact):
    dataset = KDTree(exact)
    distances,_ = dataset.query(pts,k=1)
    return distances

def Hicken_deriv_eval(pts,dataset,norm_vec,k,rho):
    distances,indices = dataset.query(pts,k=k)
    di = dataset.data[indices] - pts.reshape(num_pts,1,3)
    check = 2*np.heaviside(np.einsum('ijk,ijk->ij',di,norm_vec[indices]),1) - 1
    sign = 2*np.heaviside(np.sum(check,axis=1),1) - 1

    d_norm = (distances.T - distances[:,0]).T
    exp = np.exp(-rho*d_norm)

    dhi = np.empty((len(pts),3))
    dlow = np.empty((len(pts),3))
    for i,(i_pt,ind) in enumerate(zip(pts,indices)):
        k_pts = dataset.data[ind]

        dx = np.transpose((i_pt-k_pts).T/(distances[i] + 1e-20)) # Avoid dividing by zero when distance = 0

        hi_terms = np.empty((k,3))
        dexp = np.empty((k,3))
        dexp = dx-np.repeat(dx[0],k).reshape(3,k).T
        hi_terms = dx - rho*np.einsum('i,ij->ij',distances[i],dexp)
        dhi[i] = np.einsum('ij,i->j',hi_terms,exp[i])
        dlow[i] = np.einsum('ij,i->j',-rho*dexp,exp[i])

    low = np.sum(exp,axis=1)
    hi = np.einsum('ij,ij->i',distances,exp)
    
    # Quotient Rule (OLD)
    deriv = np.einsum('i,ij->ij', 1/low**2, (np.einsum('i,ij->ij',sign*low,dhi) - np.einsum('i,ij->ij',sign*hi,dlow)))
    return deriv

def KS_eval(pts,KDTree,norm_vec,k,rho):
    distances,indices = KDTree.query(pts,k=k)
    d_norm = np.transpose(distances.T - distances[:,0]) + 1e-20
    exp = np.exp(-rho*d_norm)
    Dx = KDTree.data[indices] - np.reshape(pts,(pts.shape[0],1,pts.shape[1]))
    phi = np.einsum('ijk,ij->i',Dx*norm_vec[indices],exp)/np.sum(exp,axis=1)
    return phi

def KS_eval_timing(pts,KDTree,norm_vec,k,rho):
    t1 = time.perf_counter()
    distances,indices = KDTree.query(pts,k=k)
    t2 = time.perf_counter()
    d_norm = np.transpose(distances.T - distances[:,0]) + 1e-20
    exp = np.exp(-rho*d_norm)
    Dx = KDTree.data[indices] - np.reshape(pts,(pts.shape[0],1,pts.shape[1]))
    # phi = np.einsum('ijk,ijk,ij,i->i',Dx,norm_vec[indices],exp,1/np.sum(exp,axis=1))
    phi = np.einsum('ijk,ij->i',Dx*norm_vec[indices],exp)/np.sum(exp,axis=1)
    t3 = time.perf_counter()
    return phi, t2-t1, t3-t2

def time_vs_Ngamma():
    import time
    import matplotlib.pyplot as plt
    
    exact = np.random.rand(10000,3)
    dataset = KDTree(exact)
    norm_vec = np.random.rand(10000,3)

    k = 50
    rho = 20

    res = 60
    time_set1 = np.zeros((20,res))
    time_set2 = np.zeros((20,res))
    deriv_time_set = np.zeros(res)
    time_data = np.logspace(2,6,res,dtype=int)
    i_pts = np.random.rand(1000,3)
    log_N = np.zeros(res)
    for j in range(20):
        np.random.seed(0)
        for i,N_gamma in enumerate(time_data):
            print(N_gamma)
            exact = np.random.rand(N_gamma,3)
            dataset = KDTree(exact)
            norm_vec = np.random.rand(N_gamma,3)
            # t1 = time.perf_counter()
            phi, KD_time, other_time = KS_eval_timing(i_pts,dataset,norm_vec,k,rho)
            # t2 = time.perf_counter()
            time_set1[j,i] = KD_time
            time_set2[j,i] = other_time
            # t1 = time.perf_counter()
            # deriv = Hicken_deriv_eval(i_pts,dataset,norm_vec,k,rho)
            # t2 = time.perf_counter()
            # deriv_time_set[i] = t2-t1
            # print(num_pts,time_set[i],deriv_time_set[i])
            log_N[i] = np.log10(N_gamma)
        
    plt.loglog(time_data,np.mean(time_set1,axis=0),'.--',markersize=15,label='KD_time')
    plt.loglog(time_data,np.mean(time_set2,axis=0),'.--',markersize=15,label='other_time')
    plt.loglog(time_data,log_N,'k',label='Ideal')
    # plt.loglog(time_data,deriv_time_set,label='Derivative Evaluation')
    plt.legend()
    plt.xlabel('Number of evaluations')
    plt.ylabel('CPU Time')
    plt.show()

if __name__ == '__main__':
    time_vs_Ngamma()