import numpy as np
from scipy.spatial import KDTree

def Hicken_eval(pts,dataset,norm_vec,k,rho):
    distances,indices = dataset.query(pts,k=k)
    if k==1:
        phi = (dataset.data[indices] - pts)*norm_vec[indices]
        return phi
    d_norm = np.transpose(distances.T - distances[:,0]) + 1e-20
    exp = np.exp(-rho*d_norm)
    Dx = dataset.data[indices,:] - np.reshape(pts,(pts.shape[0],1,pts.shape[1]))
    phi = np.einsum('ijk,ij->i',Dx*norm_vec[indices],exp)/np.sum(exp,axis=1)
    return phi

def Exact_eval(pts,exact):
    dataset = KDTree(exact)
    distances,_ = dataset.query(pts,k=1)
    return distances

def Hicken_deriv_eval(pts,dataset,norm_vec,k,rho):
    distances,indices = dataset.query(pts,k=k)
    di = dataset.data[indices] - pts.reshape(len(pts),1,2)
    check = 2*np.heaviside(np.einsum('ijk,ijk->ij',di,norm_vec[indices]),1) - 1
    sign = 2*np.heaviside(np.sum(check,axis=1),1) - 1

    d_norm = (distances.T - distances[:,0]).T
    exp = np.exp(-rho*d_norm)

    dhi = np.empty((len(pts),2))
    dlow = np.empty((len(pts),2))
    for i,(i_pt,ind) in enumerate(zip(pts,indices)):
        k_pts = dataset.data[ind]

        dx = np.transpose((i_pt-k_pts).T/(distances[i] + 1e-20)) # Avoid dividing by zero when distance = 0

        hi_terms = np.empty((k,2))
        dexp = np.empty((k,2))
        dexp = dx-np.repeat(dx[0],k).reshape(2,k).T
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

def main():
    import time
    import matplotlib.pyplot as plt
    
    exact = np.random.rand(10000,3)
    dataset = KDTree(exact)
    norm_vec = np.random.rand(10000,3)

    k = 4
    rho = 10

    res = 200
    time_set = np.zeros(res)
    deriv_time_set = np.zeros(res)
    time_data = np.logspace(1,3,res,dtype=int)
    for i,num_pts in enumerate(time_data):
        i_pts = np.random.rand(num_pts,3)
        t1 = time.perf_counter()
        phi = KS_eval(i_pts,dataset,norm_vec,k,rho)
        t2 = time.perf_counter()
        time_set[i] = t2-t1
        t1 = time.perf_counter()
        deriv = Hicken_deriv_eval(i_pts,dataset,norm_vec,k,rho)
        t2 = time.perf_counter()
        deriv_time_set[i] = t2-t1
        print(num_pts,time_set[i],deriv_time_set[i])
    plt.loglog(time_data,time_set,label='Evaluation')
    plt.loglog(time_data,deriv_time_set,label='Derivative Evaluation')
    plt.legend()
    plt.xlabel('Number of evaluations')
    plt.ylabel('CPU Time')
    plt.show()

def main2():
    import time
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set()

    k = 5
    d = 2
    rho = 10
    res = 10

    ng_set = np.logspace(1,7,res, dtype=int)
    time_set = np.zeros(res)
    np.random.seed(1)
    for i,ng in enumerate(ng_set):
        exact = np.random.rand(ng,d)
        norm_vec = np.random.rand(ng,d)
        dataset = KDTree(exact, leafsize=10, compact_nodes=False, balanced_tree=False)
        i_pts = 100*np.random.rand(1000,d)
        t1 = time.perf_counter()
        phi = KS_eval(i_pts,dataset,norm_vec,k,rho)
        t2 = time.perf_counter()
        time_set[i] = (t2-t1)/len(phi)
        print(ng)
    plt.loglog(ng_set,time_set,label='Eval Time')
    plt.loglog(ng_set,np.power(np.array(d*ng_set),1/d)*time_set[0]/(d*ng_set[0]**(1/d)),'k-',linewidth=6,alpha=0.13)
    plt.legend()
    plt.xlabel('Num Nodes in k-d tree')
    plt.ylabel('CPU Time')
    plt.show()

if __name__ == '__main__':
    # main()
    main2()