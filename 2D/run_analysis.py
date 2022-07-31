from modules.multi_circle import multi_circle
from modules.multi_obj import multi_obj
from modules.rectangle import rectangle
from modules.ellipse import ellipse
from modules.Analyze import model
from modules.Hicken_Method import KS_eval
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import seaborn as sns
import numpy as np
import pickle

sns.set()
def evaluate(pts,KDTree,norm_vec,curv,k,rho):
    distances,indices = KDTree.query(pts,k=k)
    d_norm = np.transpose(distances.T - distances[:,0]) + 1e-20
    exp = np.exp(-rho*d_norm)
    Dx = KDTree.data[indices] - np.reshape(pts,(pts.shape[0],1,pts.shape[1]))
    phi = np.einsum('ijk,ijk,ij,i->i',Dx,norm_vec[indices],exp,1/np.sum(exp,axis=1))
    return phi

# mode = 'gen_data'
# mode = 'plot_data'
mode = 'Bspline_analysis_vary_L1'
mode = 'Bspline_analysis_vary_L2'
# mode = 'Bspline_analysis_vary_L3'
mode = 'Visualize_lambdas'
# mode = 'Visualize_lambda-contours'
# mode = 'Visualize_lambda-slices'

# shape = 'ellipse'
# shape = 'rectangle'
# shape = 'multi-circles'
shape = 'multi-obj'

max_cps = 74
# max_cps = 140
order = 4
a = 5
b = 7
R = 2
dim = 2
border = 0.20
soft_const = True
tol = 1e-4

num_surf_pts = 1000
centers = [[-13.,-0.5],[-7.,2.],[2.,0.],[10.,-4.]]
radii = [2.,2.,4.,3.]

if shape == 'multi-obj':
    centers = [[-13.,-0.5],[5.,-3.],[2.,6.],[-5.,-2]]
    radii = [2.5,4.]
    ab = [[12,4],[3,6]]

iter = 1
L = [1., 1., 1.]
# iter = 2
# L = [1e-3, 1e-3, 8e-1]
# iter = 3
# L = [1e-3, 1e1, 1e-1]
data = np.logspace(-8,4,13)

if mode == 'gen_data':
    # Parameters
    a = 5
    b = 7
    if shape == 'ellipse':
        e = ellipse(a,b)
        tol = 1e-4
    elif shape == 'rectangle':
        e = rectangle(a,b)
        tol = 1e-4
    # BSpline Volume Parameters #
    dim = 2
    R = 2
    border = 0.25

    soft_const = True
    L1 = 1.
    L2 = 1.
    L3 = 1.
    m = model(e,max_cps,R,border,dim,soft_const,tol)
    data = np.logspace(1,3,12,dtype=int)
    for i,ng in enumerate(data):
        filename = 'SAVED_DATA/'+shape+str(ng)+'o'+str(order)+str(max_cps)
        Func = m.inner_solve(ng,L1,L2,L3,order)
        pickle.dump(Func, open( filename+".pkl","wb"))

if mode == 'plot_data':
    orders = [4,5,6]
    pt_data = np.logspace(1,3,12,dtype=int)
    res = 65
    bbox_max = 5

    # h_surf_RMS = np.empty((len(rng_data),len(data)))
    # h_local_RMS = np.empty((len(rng_data),len(data),len(ep_data)))
    b_local_RMS = np.empty((len(pt_data),len(orders),res))
    b_surf_RMS = np.empty((len(pt_data),len(orders)))
    for i,ng in enumerate(pt_data):
        for j,order in enumerate(orders):
            filename = 'SAVED_DATA/'+shape+str(ng)+'o'+str(order)+str(max_cps)
            Func = pickle.load(open(filename+".pkl","rb"))
            data, b_local_RMS[i,j] = Func.check_local_RMS_error(bbox_max,res)
            phi = Func.eval_surface()
            b_surf_RMS[i,j] = np.sqrt(np.sum(phi**2)/len(Func.exact[0]))
            # Func.visualize_current()
            # plt.show()
        # for n,ep in enumerate(ep_data):
        #     b_local_RMS[i,n] = Func.check_local_RMS_error(n_exact,ep=ep)
        # for j,rho_ng in enumerate(rng_data):
        #     rho = rho_ng*ng
        #     pts = e.points(ng)
        #     norm_vec = e.unit_pt_normals(ng)
        #     curv = e.get_princ_curvatures(ng)
        #     dataset = KDTree(pts)
        #     _,phi = evaluate(sample_pts,dataset,norm_vec,curv,k,rho)
        #     h_surf_RMS[j,i] = np.sqrt(np.sum(  phi**2  )/n_exact)

        #     for n,ep in enumerate(ep_data):
        #         ext_pts = sample_pts + ep*e.unit_pt_normals(n_exact)
        #         int_pts = sample_pts - ep*e.unit_pt_normals(n_exact)
        #         pts = np.vstack((ext_pts,int_pts))
        #         _,phi = evaluate(pts,dataset,norm_vec,curv,k,rho)
        #         d_exact,_ = dataset_exact.query(pts,k=1)
        #         h_local_RMS[j,i,n] = np.sqrt(np.sum(  (abs(phi)-d_exact)**2  )/ (2*n_exact))

    # line = np.zeros((2,2))
    # dx = pt_data[-1]/pt_data[0]

    colors = ['b','tab:orange','g']

    plt.figure(figsize=(16,9), dpi=120)
    sns.set()
    for j,order in enumerate(orders):
        plt.semilogx(pt_data,b_surf_RMS[:,j],'.-',label='Order: '+str(order))
    # line[0] = np.array([pt_data[0], b_surf_RMS[-1]])
    # line[1] = np.array([pt_data[-1], b_surf_RMS[-1]/dx])
    # plt.loglog(line[:,0],line[:,1],'k:',label='Linear')
    # for i,rho_ng in enumerate(rng_data):
        # plt.loglog(data,h_surf_RMS[i,:],'.-',label="Hicken's method")
    plt.xlabel('$n_{\Gamma}$')
    plt.ylabel('RMS Error')
    plt.title("Surface RMS error for "+shape+" model")
    plt.legend(loc='upper right')

    plt.figure()
    sns.set()
    for i,ng in enumerate(pt_data):
        for j,order in enumerate(orders):
            if j==0:
                # plt.loglog(data,h_local_RMS[i,:,j],'.-',color=colors[1],label="Hicken's ep = {}".format(ep))
                plt.plot(data,b_local_RMS[i,j,:],'-',color=colors[j],label='Order: {}'.format(order))
            else:
                # plt.loglog(data,h_local_RMS[i,:,j],'.--',color=colors[1],label="Hicken's ep = {}".format(ep))
                plt.plot(data,b_local_RMS[i,j,:],'-',color=colors[j],label='Order: {}'.format(order)) 
        plt.xlabel('$\epsilon$')
        plt.ylabel('Normalized RMS')
        plt.title("Local Error for {} surface points".format(ng))
        plt.legend(loc='upper center')

if mode == 'Bspline_analysis_vary_L1':
    L1_data = L[0]*data
    L2 = L[1]
    L3 = L[2]
    ng = 100
    if shape == 'ellipse':
        e = ellipse(a,b)
    elif shape == 'rectangle':
        e = rectangle(a,b)
    elif shape == 'multi-circles':
        e = multi_circle(centers,radii)
    elif shape == 'multi-obj':
            e = multi_obj(centers,radii,ab)
    m = model(e,max_cps,R,border,dim,soft_const,tol)
    for i,L1 in enumerate(L1_data):
        # if i > 1:
        #     temp = pickle.load( open('SAVED_DATA/Opt_'+str(shape)+'_L1_'+str(iter)+'_0.pkl',"rb"))
        #     Func = m.inner_solve(ng,L1,L2,L3,order,init_manual=temp.cps[:,2])
        #     del temp
        # else:
        Func = m.inner_solve(ng,L1,L2,L3,order)
        pickle.dump(Func, open( 'SAVED_DATA/Opt_'+str(shape)+'_L1_'+str(iter)+'_'+str(i)+'.pkl',"wb"))
        del Func
        print('Finished L1 =',str(L1),'Optimization')

if mode == 'Bspline_analysis_vary_L2':
    L1 = L[0]
    L2_data = L[1]*data
    L3 = L[2]
    ng = 100
    if shape == 'ellipse':
        e = ellipse(a,b)
    elif shape == 'rectangle':
        e = rectangle(a,b)
    elif shape == 'multi-circles':
        e = multi_circle(centers,radii)
    elif shape == 'multi-obj':
            e = multi_obj(centers,radii,ab)
    m = model(e,max_cps,R,border,dim,soft_const,tol)
    for i,L2 in enumerate(L2_data):
        # if i > 1:
        #     temp = pickle.load( open('SAVED_DATA/Opt_'+str(shape)+'_L2_'+str(iter)+'_0.pkl',"rb"))
        #     Func = m.inner_solve(ng,L1,L2,L3,order,init_manual=temp.cps[:,2])
        #     del temp
        # else:
        Func = m.inner_solve(ng,L1,L2,L3,order)
        pickle.dump(Func, open( 'SAVED_DATA/Opt_'+str(shape)+'_L2_'+str(iter)+'_'+str(i)+'.pkl',"wb"))
        del Func
        print('Finished L2 =',str(L2),'Optimization')

if mode == 'Bspline_analysis_vary_L3':
    L1 = L[0]
    L2 = L[1]
    L3_data = L[2]*data
    ng = 100
    if shape == 'ellipse':
        e = ellipse(a,b)
    elif shape == 'rectangle':
        e = rectangle(a,b)
    elif shape == 'multi-circles':
        e = multi_circle(centers,radii)
    elif shape == 'multi-obj':
            e = multi_obj(centers,radii,ab)
    m = model(e,max_cps,R,border,dim,soft_const,tol)
    for i,L3 in enumerate(L3_data):
        # if i > 1:
        #     temp = pickle.load( open('SAVED_DATA/Opt_'+str(shape)+'_L3_'+str(iter)+'_0.pkl',"rb"))
        #     Func = m.inner_solve(ng,L1,L2,L3,order,init_manual=temp.cps[:,2])
        #     del temp
        # else:
        Func = m.inner_solve(ng,L1,L2,L3,order)
        pickle.dump(Func, open( 'SAVED_DATA/Opt_'+str(shape)+'_L3_'+str(iter)+'_'+str(i)+'.pkl',"wb"))
        del Func
        print('Finished L3 =',str(L3),'Optimization')

if mode == 'Visualize_lambdas':
    L1_data = L[0]*data
    L2_data = L[1]*data
    L3_data = L[2]*data
    i1 = np.argwhere(data==1e0)[0][0]
    i2 = np.argwhere(data==1e0)[0][0]
    i3 = np.argwhere(data==1e0)[0][0]

    RMS_surf_L1 = np.zeros(len(L1_data))
    MAX_surf_L1 = np.zeros(len(L1_data))
    RMS_local_L1 = np.zeros(len(L1_data))
    Energy1_L1 = np.zeros(len(L1_data))
    Energy2_L1 = np.zeros(len(L1_data))
    Energy3_L1 = np.zeros(len(L1_data))
    for i,L1 in enumerate(L1_data):
        Func = pickle.load( open( 'SAVED_DATA/Opt_'+str(shape)+'_L1_'+str(iter)+'_'+str(i)+'.pkl', "rb" ) )
        Energy1_L1[i] = Func.E_scaled[0] # Measurement of the curvature energy
        Energy2_L1[i] = Func.E_scaled[1] # Surf energy
        Energy3_L1[i] = Func.E_scaled[2] # local energy
        ep_range,data = Func.check_local_RMS_error(1,2) # 1% both ways, average the error
        RMS_local_L1[i] = np.mean(data)
        phi = Func.eval_surface()
        MAX_surf_L1[i] = np.max(abs(phi))/Func.Bbox_diag
        RMS_surf_L1[i] = np.sqrt(np.sum(phi**2)/len(phi))/Func.Bbox_diag
        print('Finished L1='+str(L1)+' dataset')
    RMS_surf_L2 = np.zeros(len(L2_data))
    MAX_surf_L2 = np.zeros(len(L2_data))
    RMS_local_L2 = np.zeros(len(L2_data))
    Energy1_L2 = np.zeros(len(L2_data))
    Energy2_L2 = np.zeros(len(L2_data))
    Energy3_L2 = np.zeros(len(L2_data))
    for i,L2 in enumerate(L2_data):
        Func = pickle.load( open( 'SAVED_DATA/Opt_'+str(shape)+'_L2_'+str(iter)+'_'+str(i)+'.pkl', "rb" ) )
        Energy1_L2[i] = Func.E_scaled[0] # Measurement of the curvature energy
        Energy2_L2[i] = Func.E_scaled[1] # Surf energy
        Energy3_L2[i] = Func.E_scaled[2] # local energy
        ep_range,data = Func.check_local_RMS_error(1,2) # 1% both ways, average the error
        RMS_local_L2[i] = np.mean(data)
        phi = Func.eval_surface()
        MAX_surf_L2[i] = np.max(abs(phi))/Func.Bbox_diag
        RMS_surf_L2[i] = np.sqrt(np.sum(phi**2)/len(phi))/Func.Bbox_diag
        print('Finished L2='+str(L2)+' dataset')

    RMS_surf_L3 = np.zeros(len(L3_data))
    MAX_surf_L3 = np.zeros(len(L3_data))
    RMS_local_L3 = np.zeros(len(L3_data))
    Energy1_L3 = np.zeros(len(L3_data))
    Energy2_L3 = np.zeros(len(L3_data))
    Energy3_L3 = np.zeros(len(L3_data))
    for i,L3 in enumerate(L3_data):
        Func = pickle.load( open( 'SAVED_DATA/Opt_'+str(shape)+'_L3_'+str(iter)+'_'+str(i)+'.pkl', "rb" ) )
        Energy1_L3[i] = Func.E_scaled[0] # Measurement of the curvature energy
        Energy2_L3[i] = Func.E_scaled[1] # Surf energy
        Energy3_L3[i] = Func.E_scaled[2] # local energy
        ep_range,data = Func.check_local_RMS_error(1,2) # 1% both ways, average the error
        RMS_local_L3[i] = np.mean(data)
        phi = Func.eval_surface()
        MAX_surf_L3[i] = np.max(abs(phi))/Func.Bbox_diag
        RMS_surf_L3[i] = np.sqrt(np.sum(phi**2)/len(phi))/Func.Bbox_diag
        print('Finished L3='+str(L3)+' dataset')
    
    plt.figure(figsize=(14,10),dpi=120)
    ax1 = plt.subplot(2,3,1)
    ax1.loglog(L1_data,Energy1_L1/Energy1_L1[i1],'.--',color='tab:orange',label='$\Delta E_1$')
    ax1.loglog(L1_data,Energy2_L1/Energy2_L1[i1],'.--',color='tab:red',label='$\Delta E_2$')
    ax1.loglog(L1_data,Energy3_L1/Energy3_L1[i1],'.--',color='tab:blue',label='$\Delta E_3$')
    ax1.set_xlabel('$\lambda_1$')
    ax1.set_ylabel('Change in Magnitude about $\lambda_1={}$'.format(L[0]))
    ax1.set_title('Energy Varying $\lambda_1$')
    ax1.legend()

    ax4 = plt.subplot(2,3,4)
    ax4.loglog(L1_data,RMS_surf_L1,'.-',color='tab:blue',label='RMS Surface')
    ax4.loglog(L1_data,MAX_surf_L1,'.-',color='tab:cyan',label='Maximum Surface')
    ax4.loglog(L1_data,RMS_local_L1,'.-',color='tab:red',label='Local ($\epsilon = 1\%$)')
    ax4.set_xlabel('$\lambda_1$')
    ax4.set_ylabel('Normalized Error')
    ax4.set_title('Error Varying $\lambda_1$')
    ax4.legend() #loc='upper right'

    ax2 = plt.subplot(2,3,2,sharey=ax1)
    ax2.loglog(L2_data,Energy1_L2/Energy1_L2[i2],'.--',color='tab:orange',label='$\Delta E_1$')
    ax2.loglog(L2_data,Energy2_L2/Energy2_L2[i2],'.--',color='tab:red',label='$\Delta E_2$')
    ax2.loglog(L2_data,Energy3_L2/Energy3_L2[i2],'.--',color='tab:blue',label='$\Delta E_3$')
    ax2.set_xlabel('$\lambda_2$')
    ax2.set_ylabel('Change in Magnitude about $\lambda_2={}$'.format(L[1]))
    ax2.set_title('Energy Varying $\lambda_2$')
    ax2.legend()

    ax5 = plt.subplot(2,3,5,sharey=ax4)
    ax5.loglog(L2_data,RMS_surf_L2,'.-',color='tab:blue',label='RMS Surface')
    ax5.loglog(L2_data,MAX_surf_L2,'.-',color='tab:cyan',label='Maximum Surface')
    ax5.loglog(L2_data,RMS_local_L2,'.-',color='tab:red',label='Local ($\epsilon = 1\%$)')
    ax5.set_xlabel('$\lambda_2$')
    ax5.set_ylabel('Normalized Error')
    ax5.set_title('Varying $\lambda_2$')
    ax5.legend()

    ax3 = plt.subplot(2,3,3,sharey=ax1)
    ax3.loglog(L3_data,Energy1_L3/Energy1_L3[i3],'.--',color='tab:orange',label='$\Delta E_1$')
    ax3.loglog(L3_data,Energy2_L3/Energy2_L3[i3],'.--',color='tab:red',label='$\Delta E_2$')
    ax3.loglog(L3_data,Energy3_L3/Energy3_L3[i3],'.--',color='tab:blue',label='$\Delta E_3$')
    ax3.set_xlabel('$\lambda_3$')
    ax3.set_ylabel('Change in Magnitude about $\lambda_3={}$'.format(L[2]))
    ax3.set_title('Energy Varying $\lambda_3$')
    ax3.legend()

    ax6 = plt.subplot(2,3,6,sharey=ax4)
    ax6.loglog(L3_data,RMS_surf_L3,'.-',color='tab:blue',label='RMS Surface')
    ax6.loglog(L3_data,MAX_surf_L3,'.-',color='tab:cyan',label='Maximum Surface')
    ax6.loglog(L3_data,RMS_local_L3,'.-',color='tab:red',label='Local ($\epsilon = 1\%$)')
    ax6.set_xlabel('$\lambda_3$')
    ax6.set_ylabel('Normalized Error')
    ax6.set_title('Varying $\lambda_3$')
    ax6.legend()
    plt.tight_layout()

if mode == 'Visualize_lambda-contours':
    L1_data = L[0]*data
    L2_data = L[1]*data
    L3_data = L[2]*data
    i1 = np.argwhere(data==1e0)[0][0]
    i2 = np.argwhere(data==1e0)[0][0]
    i3 = np.argwhere(data==1e0)[0][0]
    L1_data = L1_data[i1-1:i1+1]
    L2_data = L2_data[i2-1:i2+1]
    L3_data = L3_data[i3-1:i3+1]
    plt.figure(figsize=(10,5.5))
    ax1 = plt.subplot(1,3,1)
    styles = ['-','-',':']
    colors = ['green','red']
    for i,L1 in enumerate(L1_data):
        Func = pickle.load( open( 'SAVED_DATA/Opt_'+str(shape)+'_L1_'+str(iter)+'_'+str(i)+'.pkl', "rb" ) )
        res = 500
        x = Func.dimensions[0]
        y = Func.dimensions[1]
        uu,vv = np.meshgrid(np.linspace(0,1,res),
                            np.linspace(0,1,res))
        b = Func.Surface.get_basis_matrix(uu.flatten(),vv.flatten(),0,0)
        xx = b.dot(Func.cps[:,0]).reshape(res,res)
        yy = b.dot(Func.cps[:,1]).reshape(res,res)
        phi = b.dot(Func.cps[:,2]).reshape(res,res)
        ax1.contour(xx,yy,phi,linestyles='-',levels=[-2,-1,0,1,2],colors=colors[i],linewidths=1.5)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_xticks([x[0],np.sum(x)/2,x[1]])
        ax1.set_yticks([y[0],np.sum(y)/2,y[1]])
        ax1.set_xlim(Func.dimensions[0,0], Func.dimensions[0,1])
        ax1.set_ylim(Func.dimensions[1,0], Func.dimensions[1,1])
        ax1.set_title('Varying $\lambda_1$')
        ax1.axis('equal')
        print('Finished L1='+str(L1)+' dataset')
    ax1.plot(Func.exact[0][:,0],Func.exact[0][:,1],'k-',label='Exact')

    ax2 = plt.subplot(1,3,2)
    for i,L2 in enumerate(L2_data):
        Func = pickle.load( open( 'SAVED_DATA/Opt_'+str(shape)+'_L2_'+str(iter)+'_'+str(i)+'.pkl', "rb" ) )
        res = 500
        x = Func.dimensions[0]
        y = Func.dimensions[1]
        uu,vv = np.meshgrid(np.linspace(0,1,res),
                            np.linspace(0,1,res))
        b = Func.Surface.get_basis_matrix(uu.flatten(),vv.flatten(),0,0)
        xx = b.dot(Func.cps[:,0]).reshape(res,res)
        yy = b.dot(Func.cps[:,1]).reshape(res,res)
        phi = b.dot(Func.cps[:,2]).reshape(res,res)
        ax2.contour(xx,yy,phi,linestyles='--',levels=[-2,-1,0,1,2],colors=colors[i],linewidths=1.5)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_xticks([x[0],np.sum(x)/2,x[1]])
        ax2.set_yticks([y[0],np.sum(y)/2,y[1]])
        ax2.set_xlim(Func.dimensions[0,0], Func.dimensions[0,1])
        ax2.set_ylim(Func.dimensions[1,0], Func.dimensions[1,1])
        ax2.set_title('Varying $\lambda_2$')
        ax2.axis('equal')
        print('Finished L2='+str(L2)+' dataset')
    ax2.plot(Func.exact[0][:,0],Func.exact[0][:,1],'k-',label='Exact')

    ax3 = plt.subplot(1,3,3)
    for i,L3 in enumerate(L3_data):
        Func = pickle.load( open( 'SAVED_DATA/Opt_'+str(shape)+'_L3_'+str(iter)+'_'+str(i)+'.pkl', "rb" ) )
        res = 500
        x = Func.dimensions[0]
        y = Func.dimensions[1]
        uu,vv = np.meshgrid(np.linspace(0,1,res),
                            np.linspace(0,1,res))
        b = Func.Surface.get_basis_matrix(uu.flatten(),vv.flatten(),0,0)
        xx = b.dot(Func.cps[:,0]).reshape(res,res)
        yy = b.dot(Func.cps[:,1]).reshape(res,res)
        phi = b.dot(Func.cps[:,2]).reshape(res,res)
        ax3.contour(xx,yy,phi,linestyles='--',levels=[-2,-1,0,1,2],colors=colors[i],linewidths=1.5)
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_xticks([x[0],np.sum(x)/2,x[1]])
        ax3.set_yticks([y[0],np.sum(y)/2,y[1]])
        ax3.set_xlim(Func.dimensions[0,0], Func.dimensions[0,1])
        ax3.set_ylim(Func.dimensions[1,0], Func.dimensions[1,1])
        ax3.set_title('Varying $\lambda_3$')
        ax3.axis('equal')
        print('Finished L3='+str(L3)+' dataset')
    ax3.plot(Func.exact[0][:,0],Func.exact[0][:,1],'k-',label='Exact')

if mode == 'Visualize_lambda-slices':
    L1_data = L[0]*data
    L2_data = L[1]*data
    L3_data = L[2]*data
    styles = ['--','--','--']
    
    plt.figure()
    ax = plt.subplot(2,2,1)
    Func = pickle.load( open( 'SAVED_DATA/Opt_'+str(shape)+'_L1_'+str(iter)+'_0.pkl', "rb" ) )
    x = Func.dimensions[0]
    y = Func.dimensions[1]
    res = 300
    xx,yy = np.meshgrid(np.linspace(x[0],x[1],res),
                        np.linspace(y[0],y[1],res))
    pts = np.stack((xx.flatten(),yy.flatten()),axis=1)
    phi = KS_eval(pts,KDTree(Func.exact[0]),Func.exact[1],15,10)
    phi = phi.reshape(res,res)
    ax.pcolormesh(xx,yy,phi,shading='gouraud',cmap='viridis') #, vmin=phi.min(), vmax=phi.max())
    ax.plot(Func.exact[0][:,0],Func.exact[0][:,1],'k-',label='Boundaries')
    ax.set_title('Model')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('equal')
    
    ax1 = plt.subplot(2,2,2)
    plt.xlabel('x')
    plt.ylabel('$\phi$')
    for i,L1 in enumerate(L1_data):
        Func = pickle.load( open( 'SAVED_DATA/Opt_'+str(shape)+'_L1_'+str(iter)+'_'+str(i)+'.pkl', "rb" ) )
        x = Func.dimensions[0]
        y = Func.dimensions[1]
        res = 1000
        xspan = np.linspace(x[0],x[1],res)
        yspan = np.zeros(res)
        pts = np.stack((xspan,yspan),axis=1)
        u,v = Func.spatial_to_parametric(pts)
        b = Func.Surface.get_basis_matrix(u,v,0,0)
        phi = b.dot(Func.cps[:,2])
        ax1.plot(xspan,phi,styles[i],linewidth=2,label='L1={}'.format(L1))
        ax1.axis('equal')
        ax1.set_title('Varying $\lambda_1$')
        print('Finished L1='+str(L1)+' dataset')
    ax1.legend()

    ax2 = plt.subplot(2,2,3,sharex=ax1)
    plt.xlabel('x')
    plt.ylabel('$\phi$')
    for i,L2 in enumerate(L2_data):
        Func = pickle.load( open( 'SAVED_DATA/Opt_'+str(shape)+'_L2_'+str(iter)+'_'+str(i)+'.pkl', "rb" ) )
        x = Func.dimensions[0]
        y = Func.dimensions[1]
        res = 1000
        xspan = np.linspace(x[0],x[1],res)
        yspan = np.zeros(res)
        pts = np.stack((xspan,yspan),axis=1)
        u,v = Func.spatial_to_parametric(pts)
        b = Func.Surface.get_basis_matrix(u,v,0,0)
        phi = b.dot(Func.cps[:,2])
        ax2.plot(xspan,phi,styles[i],linewidth=2,label='L2={}'.format(L2))
        ax2.axis('equal')
        ax2.set_title('Varying $\lambda_2$')
        print('Finished L2='+str(L2)+' dataset')
    ax2.legend()

    ax3 = plt.subplot(2,2,4,sharex=ax1)
    plt.xlabel('x')
    plt.ylabel('$\phi$')
    for i,L3 in enumerate(L3_data):
        Func = pickle.load( open( 'SAVED_DATA/Opt_'+str(shape)+'_L3_'+str(iter)+'_'+str(i)+'.pkl', "rb" ) )
        x = Func.dimensions[0]
        y = Func.dimensions[1]
        res = 1000
        xspan = np.linspace(x[0],x[1],res)
        yspan = np.zeros(res)
        pts = np.stack((xspan,yspan),axis=1)
        u,v = Func.spatial_to_parametric(pts)
        b = Func.Surface.get_basis_matrix(u,v,0,0)
        phi = b.dot(Func.cps[:,2])
        ax3.plot(xspan,phi,styles[i],linewidth=2,label='L3={}'.format(L3))
        ax3.axis('equal')
        ax3.set_title('Varying $\lambda_3$')
        print('Finished L3='+str(L3)+' dataset')
    ax3.legend()

plt.show()