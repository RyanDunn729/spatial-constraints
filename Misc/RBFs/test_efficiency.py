import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import perf_counter as time
import pickle
from Hicken_comp import Hicken_comp
from scipy.spatial import KDTree
import handy_funcs as mine
from openmdao.api import Problem, Group
from lsdo_geo.bsplines.bspline_volume import BSplineVolume

Hicken      = True
RBF_check   = False
Bspl_check  = True

RBFs = [100, 750, 2500]

num_cp = [10, 100]
cps = [10**3, 100**3]

pts = [10, 40, 60]
plt_pts = [10**3, 40**3, 60**3]

files = ['stl-files/heart_case03.stl', 'stl-files/Dragon_pro.stl'] # 1888, 255830, 359224 mesh_pts
num_mesh_pts = [1888, 359224]

sns.set()
plt.figure(1)
sns.set()
plt.figure(2)

if Bspl_check:
    order = 4
    def std_uniform_knot_vec(num_cps,order):
        knot_vector = np.zeros(num_cps + order)
        den = num_cps - order + 1
        for i in range(num_cps + order):
            knot_vector[i] = (i - order + 1) / den
        return knot_vector
    btimes = np.empty((len(cps),len(plt_pts)))
    dbtimes = np.empty((len(cps),len(plt_pts)))
    for i,num_cps in enumerate(num_cp):
        for j,res in enumerate(plt_pts):
            kvec = std_uniform_knot_vec(num_cps,order)
            temp_cps = np.empty((num_cps,num_cps,num_cps,4))
            temp_cps[:, :, :, 0] = np.einsum('i,j,k->ijk', np.linspace(-1,1, num_cps), np.ones(num_cps),np.ones(num_cps))
            temp_cps[:, :, :, 1] = np.einsum('i,j,k->ijk', np.ones(num_cps), np.linspace(-1,1, num_cps),np.ones(num_cps))
            temp_cps[:, :, :, 2] = np.einsum('i,j,k->ijk', np.ones(num_cps), np.ones(num_cps),np.linspace(-1,1, num_cps))
            temp_cps[:,:,:,3] = np.random.rand(num_cps,num_cps,num_cps)
            temp_cps = temp_cps.reshape((num_cps**3,4))
            Volume = BSplineVolume('name',order,order,order,kvec,kvec,kvec,[num_cps,num_cps,num_cps],temp_cps)

            t1 = time()

            basis = Volume.get_basis_matrix(0.5*np.ones(res), 0.5*np.ones(res), np.linspace(0,1,res), 0, 0, 0)
            temp = basis.dot(temp_cps)

            t2 = time()
            btimes[i,j] = t2-t1

            t1 = time()
            
            basis = Volume.get_basis_matrix(0.5*np.ones(res), 0.5*np.ones(res), np.linspace(0,1,res), 1, 0, 0)
            temp = basis.dot(temp_cps)
            basis = Volume.get_basis_matrix(0.5*np.ones(res), 0.5*np.ones(res), np.linspace(0,1,res), 0, 1, 0)
            temp = basis.dot(temp_cps)
            basis = Volume.get_basis_matrix(0.5*np.ones(res), 0.5*np.ones(res), np.linspace(0,1,res), 0, 0, 1)
            temp = basis.dot(temp_cps)

            t2 = time()
            dbtimes[i,j] = t2-t1

    plt.figure(1)
    for i in range(len(cps)):
        plt.loglog(plt_pts,btimes[i,:],'-',label='Bspline num_cps={}'.format(cps[i]))
    plt.figure(2)
    for i in range(len(cps)):
        plt.loglog(plt_pts,dbtimes[i,:],'-',label='deriv_Bspline num_cps={}'.format(cps[i]))

if Hicken:
    k = 15
    rho = 100
    border = 0.10
    Htimes = np.empty((len(files),len(plt_pts)))
    dHtimes = np.empty((len(files),len(plt_pts)))
    for j,filename in enumerate(files):
        mesh_import, _, midpts, norm_vec = mine.import_mesh(filename)
        dataset = KDTree(midpts)
        minx,miny,minz = mesh_import.min_
        maxx,maxy,maxz = mesh_import.max_
        minx = minx-((maxx-minx)*border)
        maxx = maxx+((maxx-minx)*border)
        miny = miny-((maxy-miny)*border)
        maxy = maxy+((maxy-miny)*border)
        minz = minz-((maxz-minz)*border)
        maxz = maxz+((maxz-minz)*border)
        for i,res in enumerate(pts):
            samp = [res,res,res]
            num_pts = plt_pts[i]
            pt_grid = np.empty((res**3, 3))
            pt_grid[:, 0] = np.einsum('i,j,k->ijk', np.linspace(minx,maxx,res), np.ones(res), np.ones(res)).flatten()
            pt_grid[:, 1] = np.einsum('i,j,k->ijk', np.ones(res), np.linspace(miny,maxy,res), np.ones(res)).flatten()
            pt_grid[:, 2] = np.einsum('i,j,k->ijk', np.ones(res), np.ones(res), np.linspace(minz,maxz,res)).flatten()

            t1 = time()

            distances,indices = dataset.query(pt_grid,k=k)
            di = dataset.data[indices] - pt_grid.reshape(num_pts,1,3)
            check = 2*np.heaviside(np.einsum('ijk,ijk->ij',di,norm_vec[indices]),1) - 1 # -1 or +1
            sign = 2*np.heaviside(np.sum(check,axis=1),1) - 1
            d_norm = np.transpose(distances.T - distances[:,0]) + 1e-20
            exp = np.exp(-rho*d_norm)
            phi = sign*np.einsum('ij,ij->i',distances,exp)/np.sum(exp,axis=1)

            t2 = time()
            Htimes[j,i] = t2-t1

            t1 = time()

            distances,indices = dataset.query(pt_grid,k=k)
            di = dataset.data[indices] - pt_grid.reshape(num_pts,1,3)
            check = 2*np.heaviside(np.einsum('ijk,ijk->ij',di,norm_vec[indices]),1) - 1
            sign = 2*np.heaviside(np.sum(check,axis=1),1) - 1
            d_norm = np.transpose(distances.T - distances[:,0]) + 1e-20
            exp = np.exp(-rho*d_norm)
            dhi = np.empty((len(pt_grid),3))
            dlow = np.empty((len(pt_grid),3))
            for q,i_pt in enumerate(pt_grid):
                k_pts = dataset.data[indices[q]]
                ddist = np.transpose((i_pt-k_pts).T/(distances[q] + 1e-20)) # Avoid dividing by zero when distance = 0
                dexp = ddist-np.repeat(ddist[0],k).reshape(3,k).T
                hi_terms = ddist - rho*np.einsum('i,ij->ij',distances[q],dexp)
                dhi[q] = np.einsum('ij,i->j',hi_terms,exp[q])
                dlow[q] = np.einsum('ij,i->j',-rho*dexp,exp[q])
            low = np.sum(exp,axis=1)
            hi = np.einsum('ij,ij->i',distances,exp)
            deriv = np.einsum('i,ij->ij', 1/low**2, (np.einsum('i,ij->ij',sign*low,dhi) - np.einsum('i,ij->ij',sign*hi,dlow)))
            
            t2 = time()
            dHtimes[j,i] = t2-t1

            del pt_grid
        del mesh_import, midpts, norm_vec
    plt.figure(1)
    for i in range(len(files)):
        plt.loglog(plt_pts,Htimes[i,:],'-',label='Hicken ({}pt KDTree)'.format(num_mesh_pts[i]))
    plt.figure(2)
    for i in range(len(files)):
        plt.loglog(plt_pts,dHtimes[i,:],'-',label='deriv_Hicken ({}pt KDTree)'.format(num_mesh_pts[i]))

if RBF_check:
    r = .25
    eval_time = np.empty((len(RBFs),len(pts)))
    deriv_xyz_time = np.empty((len(RBFs),len(pts)))
    deriv_RBF_time = np.empty((len(RBFs),len(pts)))
    for i,num_RBFs in enumerate(RBFs):
        ang = np.linspace(0,2*np.pi,num_RBFs)
        nodes = np.column_stack((3*np.cos(ang),np.sin(ang),ang))
        for j,res in enumerate(pts):
            num_pts = res**3
            rng = 5
            pt_grid = np.empty((num_pts,3))
            pt_grid[:,0] = np.einsum('i,j,k->ijk', np.linspace(-rng,rng,res), np.ones(res), np.ones(res)).flatten()
            pt_grid[:,1] = np.einsum('i,j,k->ijk', np.ones(res), np.linspace(-rng,rng,res), np.ones(res)).flatten()
            pt_grid[:,2] = np.einsum('i,j,k->ijk', np.ones(res), np.ones(res), np.linspace(-rng,rng,res)).flatten()

            t1 = time()
            z = np.empty(num_pts)
            for k,i_pt in enumerate(pt_grid):
                norm2 = np.einsum('ij,ij->j',i_pt-nodes, i_pt-nodes)
                z[k] = np.sum(np.exp( -(norm2)/(r**2) ))
            t2 = time()
            eval_time[i,j] = t2-t1

            # t1 = time()
            # z = np.empty((num_pts,3))
            # for k,i_pt in enumerate(pt):
            #     diff = np.transpose(i_pt-nodes)
            #     norm = np.linalg.norm(diff, axis=0)
            #     var = 2/(-r**2) * np.exp( -(norm**2)/(r**2) )
            #     z[k,:] = np.sum(diff*var,axis=1)
            # t2 = time()
            # deriv_xyz_time[i,j] = t2-t1

            # t1 = time()
            # z = np.empty((num_pts,3*num_RBFs))
            # for k,i_pt in enumerate(pt):
            #     diff = np.transpose(i_pt-nodes)
            #     norm = np.linalg.norm(diff,axis=0)
            #     var = 2/(r**2) * np.exp( -(norm**2)/(r**2) )
            #     z[k,:] = (diff*var).transpose().flatten()
            # t2 = time()
            # deriv_RBF_time[i,j] = t2-t1
    for i in range(len(RBFs)):
        plt.loglog(plt_pts,eval_time[i,:],'-',label='# RBFs={}'.format(RBFs[i]))
    # for i in range(len(RBFs)):
    #     plt.loglog(plt_pts,deriv_xyz_time[i,:],'-',label='dxyz # RBFs={}'.format(RBFs[i]))
    # for i in range(len(RBFs)):
    #     plt.loglog(plt_pts,deriv_RBF_time[i,:],'-',label='dRBF # RBFs={}'.format(RBFs[i]))


plt.figure(1)
plt.title('Evaluation Time Comparison')
plt.xlabel('# of Evaluations')
plt.ylabel('Time (sec)')
plt.legend()
plt.figure(2)
plt.title('Derivative Computation Time Comparison')
plt.xlabel('# of Evaluations')
plt.ylabel('Time (sec)')
plt.legend()
plt.show()