from modules.Bspline_Volume import BSplineVolume
from modules.read_stl import extract_stl_info
from modules.Fred_method import Freds_Method
from skimage.measure import marching_cubes
from modules.ellipsoid import Ellipsoid
from modules.Hicken import KS_eval
from modules.Analyze import model
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from stl.mesh import Mesh
import seaborn as sns
import numpy as np
import pickle
import time

def set_fonts(legendfont=12,axesfont=16):
    from matplotlib import rc
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
    rc('text', usetex=True)
    plt.rc('legend', fontsize=legendfont)    # legend fontsize
    plt.rc('axes', labelsize=axesfont)    # fontsize of the x and y labels
    return

sns.set()
set_fonts()

save_figures = True

### Select Shape ###
# Penalization
# file = 'o3Bunny_pen40'
# file = 'o4Bunny_pen35'
file = 'o4Bunny_pen40'
# file = 'o5Bunny_pen40'
# file = 'o6Bunny_pen40' 
# file = 'o4Bunny_pen34'
# file = 'o4Bunny_pen30'
# Constraints
# file = 'o4Bunny36'
# file = 'o5Bunny36'
# file = 'o6Bunny36' # Fails 377

# Best Heart
# file = 'o4Heart_pen42'

# file = 'o4Ellipsoid_pen38'

### Plot Data Mode ###
# mode = 'Hicken_analysis'
# mode = 'Bspline_analysis'
# mode = 'Bspline_analysis_vary_L1'
# mode = 'Bspline_analysis_vary_L2'
# mode = 'Bspline_analysis_vary_L3'
# mode = 'Visualize_lambdas_energies'
# mode = 'Visualize_lambdas'
# mode = 'Plot_data'
# mode = 'Comp_pen_strict'
# mode = 'Comp_err_order'
# mode = 'Hicken_v_Splines'
# mode = 'normalized_Hicken_v_Splines'
# mode = 'Comp_time'
mode = 'plot_point_cloud'
print(mode)

# BSpline Volume Parameters #
dim = 3
R = 1
order = int(file[1])
border = 0.15
max_cps = int(file[-2:])
soft_const = True
iter = 1
L = [1e-2, 10., 100.]
# iter = 2
# L = [1e-1, 1e0, 1e5]
# iter = 3
# L = [1e-3, 1e1, 1e-1]

data = np.logspace(-8,4,13)

if file[2:-2]=='Bunny':
    tol = 1e-4
    exact = pickle.load( open( "SAVED_DATA/_Bunny_data_exact_.pkl", "rb" ) )
    main_name = 'stl-files/Bunny_'
    pt_data = [77,108,252,297,327,377]
elif file[2:-2]=='Heart':
    tol = 1e-4
    exact = pickle.load( open( "SAVED_DATA/_Heart_data_exact_.pkl", "rb" ) )
    main_name = 'stl-files/Heart_'
    pt_data = [127,177,214,252,302,352]
elif file[2:-2]=='Ellipsoid':
    a = 8
    b = 7.25
    c = 5.75
    tol = 1e-4
    res_exact = 10000
    e = Ellipsoid(a,b,c)
    exact = np.stack((e.points(res_exact),e.unit_pt_normals(res_exact)))
    pt_data = [100,200,300,400,500]
elif file[2:-2]=='Bunny_pen':
    tol = 5e-5
    soft_const = True
    exact = pickle.load( open( "SAVED_DATA/_Bunny_data_exact_.pkl", "rb" ) )
    main_name = 'stl-files/Bunny_'
    pt_data = [677,1002,2002,3002,4002,5002,10002,25002,40802,63802,100002] # 77,108,201,252,412,

if mode=='Hicken_analysis':
    file = file[2:-2]
    res = 140
    k = 10
    rho = 10
    for num_pts in pt_data:
        if file[2:-2] =='Ellipsoid':
            surf_pts = e.points(num_pts)
            normals = e.unit_pt_normals(num_pts)
        else:
            filename = main_name+str(num_pts)+'.stl'
            surf_pts, normals = extract_stl_info(filename)
        lower = np.min(surf_pts,axis=0)
        upper = np.max(surf_pts,axis=0)
        diff = upper-lower
        dimensions = np.stack((lower-diff*border, upper+diff*border),axis=1)
        x = dimensions[0]
        y = dimensions[1]
        z = dimensions[2]
        pts = np.zeros((np.product(res**3), 3))
        pts[:, 0] = np.einsum('i,j,k->ijk', np.linspace(x[0],x[1],res), np.ones(res),np.ones(res)).flatten()
        pts[:, 1] = np.einsum('i,j,k->ijk', np.ones(res), np.linspace(y[0],y[1],res),np.ones(res)).flatten()
        pts[:, 2] = np.einsum('i,j,k->ijk', np.ones(res), np.ones(res),np.linspace(z[0],z[1],res)).flatten()
        phi = KS_eval(pts,surf_pts,normals,k,rho)
        phi = phi.reshape((res,res,res))
        verts, faces,_,_ = marching_cubes(phi, 0)
        verts = verts*np.diff(dimensions).flatten()/(res-1) + dimensions[:,0]
        surf = Mesh(np.zeros(faces.shape[0], dtype=Mesh.dtype))
        for i, f in enumerate(faces):
                for j in range(3):
                        surf.vectors[i][j] = verts[f[j],:]
        surf.save('SAVED_DATA/Hick_'+file+'_' + str(num_pts) + '.stl')
        print('Finished ',str(num_pts),' point Hicken File')

if mode=='Bspline_analysis':
    m = model(max_cps,R,border,dim,tol,exact,soft_const)
    for num_pts in pt_data:
        if file[2:-2] =='Ellipsoid':
            surf_pts = e.points(num_pts)
            normals = e.unit_pt_normals(num_pts)
        elif file[2:-2] =='Bunny_pen' and num_pts==100002:
            data = pickle.load( open( "SAVED_DATA/_Bunny_data_100002.pkl", "rb" ) )
            surf_pts = data[0]
            normals = data[1]
        elif file[2:-2] =='Bunny_pen' and num_pts==63802:
            data = pickle.load( open( "SAVED_DATA/_Bunny_data_63802.pkl", "rb" ) )
            surf_pts = data[0]
            normals = data[1]
        else:
            filename = main_name+str(num_pts)+'.stl'
            surf_pts, normals = extract_stl_info(filename)
        Func = m.inner_solve(surf_pts, normals, L[0], L[1], L[2], order)
        pickle.dump(Func, open( "SAVED_DATA/Opt_"+file+"_"+str(num_pts)+".pkl","wb"))
        del surf_pts,normals,Func
        print('Finished ',file,str(num_pts),' Optimization')

if mode == 'Bspline_analysis_vary_L1':
    L1_data = L[0]*data
    print(L1_data)
    L2 = L[1]
    L3 = L[2]
    m = model(max_cps,R,border,dim,tol,exact,soft_const)
    surf_pts, normals = extract_stl_info('stl-files/Bunny_5002.stl')
    for i,L1 in enumerate(L1_data):
        Func = m.inner_solve(surf_pts, normals, L1, L2, L3, order)
        pickle.dump(Func, open( "SAVED_DATA/Opt_Bunny_L1_"+str(iter)+"_"+str(i)+".pkl","wb"))
        del Func
        print('Finished L1 =',str(L1),'Optimization')

if mode == 'Bspline_analysis_vary_L2':
    L1 = L[0]
    L2_data = L[1]*data
    print(L2_data)
    L3 = L[2]
    m = model(max_cps,R,border,dim,tol,exact,soft_const)
    surf_pts, normals = extract_stl_info('stl-files/Bunny_5002.stl')
    for i,L2 in enumerate(L2_data):
        Func = m.inner_solve(surf_pts, normals, L1, L2, L3, order)
        pickle.dump(Func, open( "SAVED_DATA/Opt_Bunny_L2_"+str(iter)+"_"+str(i)+".pkl","wb"))
        del Func
        print('Finished L2 =',str(L2),'Optimization')

if mode == 'Bspline_analysis_vary_L3':
    L1 = L[0]
    L2 = L[1]
    L3_data = L[2]*data
    print(L3_data)
    m = model(max_cps,R,border,dim,tol,exact,soft_const)
    surf_pts, normals = extract_stl_info('stl-files/Bunny_5002.stl')
    for i,L3 in enumerate(L3_data):
        Func = m.inner_solve(surf_pts, normals, L1, L2, L3, order)
        pickle.dump(Func, open( "SAVED_DATA/Opt_Bunny_L3_"+str(iter)+"_"+str(i)+".pkl","wb"))
        del Func
        print('Finished L3 =',str(L3),'Optimization')

if mode == 'Visualize_lambdas_energies':
    L1_data = L[0]*data
    L2_data = L[1]*data
    L3_data = L[2]*data
    i1 = np.argwhere(data==1e0)[0][0] -4
    i2 = np.argwhere(data==1e0)[0][0] -4 
    i3 = np.argwhere(data==1e0)[0][0] -4

    RMS_surf_L1 = np.ones(len(L1_data))
    MAX_surf_L1 = np.ones(len(L1_data))
    RMS_local_L1 = np.ones(len(L1_data))
    max_Fnorm_L1 = np.ones(len(L1_data))
    Energy1_L1 = np.ones(len(L1_data))
    Energy2_L1 = np.ones(len(L1_data))
    Energy3_L1 = np.ones(len(L1_data))
    Runtime_L1 = np.ones(len(L1_data))
    for i,L1 in enumerate(L1_data):
        if i<4:
            continue
        Func = pickle.load( open( "SAVED_DATA/Opt_Bunny_L1_"+str(iter)+"_"+str(i)+".pkl", "rb" ) )
        Energy1_L1[i] = Func.E_norm[0] # Measurement of the curvature energy
        Energy2_L1[i] = Func.E_norm[1] # Local energy
        Energy3_L1[i] = Func.E_norm[2] # Surf energy
        ep_range,data = Func.check_local_RMS_error(1,2) # 1% both ways, average the error
        RMS_local_L1[i] = np.mean(data)
        phi = Func.eval_surface()
        MAX_surf_L1[i] = np.max(abs(phi))/Func.Bbox_diag
        RMS_surf_L1[i] = np.sqrt(np.mean(phi**2))/Func.Bbox_diag
        max_Fnorm_L1[i] = Func.get_max_Fnorm()
        Runtime_L1[i] = Func.runtime
        print('Finished L1='+str(L1)+' dataset')
    RMS_surf_L2 = np.ones(len(L2_data))
    MAX_surf_L2 = np.ones(len(L2_data))
    RMS_local_L2 = np.ones(len(L2_data))
    max_Fnorm_L2 = np.ones(len(L2_data))
    Energy1_L2 = np.ones(len(L2_data))
    Energy2_L2 = np.ones(len(L2_data))
    Energy3_L2 = np.ones(len(L2_data))
    Runtime_L2 = np.ones(len(L2_data))
    for i,L2 in enumerate(L2_data):
        if i<4:
            continue
        Func = pickle.load( open( "SAVED_DATA/Opt_Bunny_L2_"+str(iter)+"_"+str(i)+".pkl", "rb" ) )
        Energy1_L2[i] = Func.E_norm[0] # Measurement of the curvature energy
        Energy2_L2[i] = Func.E_norm[1] # Surf energy
        Energy3_L2[i] = Func.E_norm[2] # local energy
        ep_range,data = Func.check_local_RMS_error(1,2) # 1% both ways, average the error
        RMS_local_L2[i] = np.mean(data)
        phi = Func.eval_surface()
        MAX_surf_L2[i] = np.max(abs(phi))/Func.Bbox_diag
        RMS_surf_L2[i] = np.sqrt(np.mean(phi**2))/Func.Bbox_diag
        max_Fnorm_L2[i] = Func.get_max_Fnorm()
        Runtime_L2[i] = Func.runtime
        print('Finished L2='+str(L2)+' dataset')
    RMS_surf_L3 = np.ones(len(L3_data))
    MAX_surf_L3 = np.ones(len(L3_data))
    RMS_local_L3 = np.ones(len(L3_data))
    max_Fnorm_L3 = np.ones(len(L2_data))
    Energy1_L3 = np.ones(len(L3_data))
    Energy2_L3 = np.ones(len(L3_data))
    Energy3_L3 = np.ones(len(L3_data))
    Runtime_L3 = np.ones(len(L3_data))
    for i,L3 in enumerate(L3_data):
        if i<4:
            continue
        Func = pickle.load( open( "SAVED_DATA/Opt_Bunny_L3_"+str(iter)+"_"+str(i)+".pkl", "rb" ) )
        Energy1_L3[i] = Func.E_norm[0] # Measurement of the curvature energy
        Energy2_L3[i] = Func.E_norm[1] # Surf energy
        Energy3_L3[i] = Func.E_norm[2] # local energy
        ep_range,data = Func.check_local_RMS_error(1,2) # 1% both ways, average the error
        RMS_local_L3[i] = np.mean(data)
        phi = Func.eval_surface()
        MAX_surf_L3[i] = np.max(abs(phi))/Func.Bbox_diag
        RMS_surf_L3[i] = np.sqrt(np.mean(phi**2))/Func.Bbox_diag
        max_Fnorm_L3[i] = Func.get_max_Fnorm()
        Runtime_L3[i] = Func.runtime
        print('Finished L3='+str(L3)+' dataset')

    L1_data = L1_data[4:]
    L2_data = L2_data[4:]
    L3_data = L3_data[4:]
    Energy1_L1 = Energy1_L1[4:]
    Energy1_L2 = Energy1_L2[4:]
    Energy1_L3 = Energy1_L3[4:]
    Energy2_L1 = Energy2_L1[4:]
    Energy2_L2 = Energy2_L2[4:]
    Energy2_L3 = Energy2_L3[4:]
    Energy3_L1 = Energy3_L1[4:]
    Energy3_L2 = Energy3_L2[4:]
    Energy3_L3 = Energy3_L3[4:]
    RMS_surf_L1 = RMS_surf_L1[4:]
    RMS_surf_L2 = RMS_surf_L2[4:]
    RMS_surf_L3 = RMS_surf_L3[4:]
    MAX_surf_L1 = MAX_surf_L1[4:]
    MAX_surf_L2 = MAX_surf_L2[4:]
    MAX_surf_L3 = MAX_surf_L3[4:]
    RMS_local_L1 = RMS_local_L1[4:]
    RMS_local_L2 = RMS_local_L2[4:]
    RMS_local_L3 = RMS_local_L3[4:]
    max_Fnorm_L1 = max_Fnorm_L1[4:]
    max_Fnorm_L2 = max_Fnorm_L2[4:]
    max_Fnorm_L3 = max_Fnorm_L3[4:]
    Runtime_L1 = Runtime_L1[4:]
    Runtime_L2 = Runtime_L2[4:]
    Runtime_L3 = Runtime_L3[4:]

    sns.set(style='ticks')
    fig1, axs1 = plt.subplots(2,1,sharex=True,figsize=(6,7),dpi=180)
    fig1.subplots_adjust(hspace=0)
    axs1[0].loglog(L1_data,Energy3_L1/Energy3_L1[i1],'*--',linewidth=2,markersize=12,color='tab:orange',label='$\mathcal{E}_1$')
    axs1[0].loglog(L1_data,Energy2_L1/Energy2_L1[i1],'D--',linewidth=2,markersize=6,color='tab:red',label='$\mathcal{E}_2$')
    axs1[0].loglog(L1_data,Energy1_L1/Energy1_L1[i1],'o--',linewidth=2,markersize=6,color='tab:blue',label='$\mathcal{E}_3$')
    axs1[0].set_ylabel('Energy about $\lambda_1={}$'.format(L[0]))
    axs1[0].set_xticks(L1_data)
    axs1[0].legend(loc='upper left',framealpha=1,edgecolor='black',facecolor='white')
    axs1[0].set_ylim(2e-2,1e3)
    axs1[0].grid()

    axs1[1].loglog(L1_data,max_Fnorm_L1,'*-',linewidth=2,markersize=12,color='tab:orange',label='Curvature')
    axs1[1].loglog(L1_data,RMS_local_L1,'D-',linewidth=2,markersize=6,color='tab:red',label='Local')
    axs1[1].loglog(L1_data,RMS_surf_L1,'o-', linewidth=2,markersize=6,color='tab:blue',label='Surface')
    # axs1[1].loglog(L1_data,Runtime_L1/Runtime_L1[i1],'.--',markersize=8,label='Optimization time')
    # ax4.loglog(L1_data,MAX_surf_L1,'.-',markersize=8,color='tab:cyan',label='Max Surface Error')
    axs1[1].set_xlabel('$\lambda_1$')
    axs1[1].set_ylabel('Error')
    axs1[1].set_xticks(L1_data)
    axs1[1].legend(loc='upper left',framealpha=1,edgecolor='black',facecolor='white')
    axs1[1].set_ylim(3e-4,1e1)
    axs1[1].grid()
    sns.despine()
    fig1.subplots_adjust(hspace=0)
    plt.tight_layout()
    plt.savefig('PDF_figures/L1.pdf',bbox_inches='tight')

    sns.set(style='ticks')
    fig2, axs2 = plt.subplots(2,1,sharex=True,figsize=(6,7),dpi=180)
    axs2[0].loglog(L2_data,Energy3_L2/Energy3_L2[i2],'*--',linewidth=2,markersize=12,color='tab:orange',label='$\mathcal{E}_1$')
    axs2[0].loglog(L2_data,Energy2_L2/Energy2_L2[i2],'D--',linewidth=2,markersize=6,color='tab:red',label='$\mathcal{E}_2$')
    axs2[0].loglog(L2_data,Energy1_L2/Energy1_L2[i2],'o--',linewidth=2,markersize=6,color='tab:blue',label='$\mathcal{E}_3$')
    axs2[0].set_ylabel('Energy about $\lambda_2={}$'.format(L[1]))
    axs1[0].set_xticks(L2_data)
    axs2[0].legend(loc='upper left',framealpha=1,edgecolor='black',facecolor='white')
    axs2[0].set_ylim(2e-2,1e3)
    axs2[0].grid()
    axs2[1].loglog(L2_data,max_Fnorm_L2,'*-',linewidth=2,markersize=12,color='tab:orange',label='Curvature')
    axs2[1].loglog(L2_data,RMS_local_L2,'D-',linewidth=2,markersize=6,color='tab:red',label='Local')
    axs2[1].loglog(L2_data,RMS_surf_L2, 'o-',linewidth=2,markersize=6,color='tab:blue',label='Surface')
    # axs2[1].loglog(L2_data,Runtime_L2/Runtime_L2[i2],'.--',markersize=8,label='Optimization time')
    # axs2[1].loglog(L2_data,MAX_surf_L2,'.-',markersize=8,color='tab:cyan',label='Max Surface')
    axs2[1].set_xlabel('$\lambda_2$')
    axs2[1].set_ylabel('Error')
    axs2[1].set_xticks(L2_data)
    axs2[1].legend(loc='upper left',framealpha=1,edgecolor='black',facecolor='white')
    axs2[1].set_ylim(3e-4,1e1)
    axs2[1].grid()
    sns.despine()
    fig2.subplots_adjust(hspace=0)
    plt.tight_layout()
    plt.savefig('PDF_figures/L2.pdf',bbox_inches='tight')

    sns.set(style='ticks')
    fig3, axs3 = plt.subplots(2,1,sharex=True,figsize=(6,7),dpi=180)
    axs3[0].loglog(L3_data,Energy3_L3/Energy3_L3[i3],'*--',linewidth=2,markersize=12,color='tab:orange',label='$\mathcal{E}_1$')
    axs3[0].loglog(L3_data,Energy2_L3/Energy2_L3[i3],'D--',linewidth=2,markersize=6,color='tab:red',label='$\mathcal{E}_2$')
    axs3[0].loglog(L3_data,Energy1_L3/Energy1_L3[i3],'o--',linewidth=2,markersize=6,color='tab:blue',label='$\mathcal{E}_3$')
    axs3[0].set_ylabel('Energy about $\lambda_3={}$'.format(L[2]))
    axs3[0].set_xticks(L3_data)
    axs3[0].legend(loc='upper left',framealpha=1,edgecolor='black',facecolor='white')
    axs3[0].set_ylim(2e-2,1e3)
    axs3[0].grid()
    axs3[1].loglog(L3_data,max_Fnorm_L3,'*-',linewidth=2,markersize=12,color='tab:orange',label='Curvature')
    axs3[1].loglog(L3_data,RMS_local_L3,'D-',linewidth=2,markersize=6,color='tab:red',label='Local')
    axs3[1].loglog(L3_data,RMS_surf_L3, 'o-',linewidth=2,markersize=6,color='tab:blue',label='Surface')
    # axs3[1].loglog(L3_data,Runtime_L3/Runtime_L3[i3],'.--',markersize=8,label='Optimization time')
    # axs3[1].loglog(L3_data,MAX_surf_L3,'.-',markersize=8,color='tab:cyan',label='Max Surface')
    axs3[1].set_xlabel('$\lambda_3$')
    axs3[1].set_ylabel('Error')
    axs3[1].set_xticks(L3_data)
    axs3[1].legend(loc='upper left',framealpha=1,edgecolor='black',facecolor='white')
    axs3[1].set_ylim(3e-4,1e1)
    axs3[1].grid()
    sns.despine()
    fig3.subplots_adjust(hspace=0)
    plt.tight_layout()

    from matplotlib import rc
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
    rc('text', usetex=True)
    plt.rc('legend', fontsize=12)    # legend fontsize
    plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
    
    plt.savefig('PDF_figures/L3.pdf',bbox_inches='tight')

if mode == 'Visualize_lambdas':
    L1_data = L[0]*data
    L2_data = L[1]*data
    L3_data = L[2]*data
    i1 = np.argwhere(data==1e0)[0][0] -4
    i2 = np.argwhere(data==1e0)[0][0] -4 
    i3 = np.argwhere(data==1e0)[0][0] -4
    RMS_surf_L1 = np.ones(len(L1_data))
    MAX_surf_L1 = np.ones(len(L1_data))
    RMS_local_L1 = np.ones(len(L1_data))
    max_Fnorm_L1 = np.ones(len(L1_data))
    Energy1_L1 = np.ones(len(L1_data))
    Energy2_L1 = np.ones(len(L1_data))
    Energy3_L1 = np.ones(len(L1_data))
    Runtime_L1 = np.ones(len(L1_data))
    for i,L1 in enumerate(L1_data):
        if i<4:
            continue
        Func = pickle.load( open( "SAVED_DATA/Opt_Bunny_L1_"+str(iter)+"_"+str(i)+".pkl", "rb" ) )
        Energy1_L1[i] = Func.E_norm[0] # Measurement of the curvature energy
        Energy2_L1[i] = Func.E_norm[1] # Local energy
        Energy3_L1[i] = Func.E_norm[2] # Surf energy
        ep_range,data = Func.check_local_RMS_error(1,2) # 1% both ways, average the error
        RMS_local_L1[i] = np.mean(data)
        phi = Func.eval_surface()
        MAX_surf_L1[i] = np.max(abs(phi))/Func.Bbox_diag
        RMS_surf_L1[i] = np.sqrt(np.mean(phi**2))/Func.Bbox_diag
        max_Fnorm_L1[i] = Func.get_RMS_Fnorm()
        Runtime_L1[i] = Func.runtime
        print('Finished L1='+str(L1)+' dataset')
    RMS_surf_L2 = np.ones(len(L2_data))
    MAX_surf_L2 = np.ones(len(L2_data))
    RMS_local_L2 = np.ones(len(L2_data))
    max_Fnorm_L2 = np.ones(len(L2_data))
    Energy1_L2 = np.ones(len(L2_data))
    Energy2_L2 = np.ones(len(L2_data))
    Energy3_L2 = np.ones(len(L2_data))
    Runtime_L2 = np.ones(len(L2_data))
    for i,L2 in enumerate(L2_data):
        if i<4:
            continue
        Func = pickle.load( open( "SAVED_DATA/Opt_Bunny_L2_"+str(iter)+"_"+str(i)+".pkl", "rb" ) )
        Energy1_L2[i] = Func.E_norm[0] # Measurement of the curvature energy
        Energy2_L2[i] = Func.E_norm[1] # Surf energy
        Energy3_L2[i] = Func.E_norm[2] # local energy
        ep_range,data = Func.check_local_RMS_error(1,2) # 1% both ways, average the error
        RMS_local_L2[i] = np.mean(data)
        phi = Func.eval_surface()
        MAX_surf_L2[i] = np.max(abs(phi))/Func.Bbox_diag
        RMS_surf_L2[i] = np.sqrt(np.mean(phi**2))/Func.Bbox_diag
        max_Fnorm_L2[i] = Func.get_RMS_Fnorm()
        Runtime_L2[i] = Func.runtime
        print('Finished L2='+str(L2)+' dataset')
    RMS_surf_L3 = np.ones(len(L3_data))
    MAX_surf_L3 = np.ones(len(L3_data))
    RMS_local_L3 = np.ones(len(L3_data))
    max_Fnorm_L3 = np.ones(len(L2_data))
    Energy1_L3 = np.ones(len(L3_data))
    Energy2_L3 = np.ones(len(L3_data))
    Energy3_L3 = np.ones(len(L3_data))
    Runtime_L3 = np.ones(len(L3_data))
    for i,L3 in enumerate(L3_data):
        if i<4:
            continue
        Func = pickle.load( open( "SAVED_DATA/Opt_Bunny_L3_"+str(iter)+"_"+str(i)+".pkl", "rb" ) )
        Energy1_L3[i] = Func.E_norm[0] # Measurement of the curvature energy
        Energy2_L3[i] = Func.E_norm[1] # Surf energy
        Energy3_L3[i] = Func.E_norm[2] # local energy
        ep_range,data = Func.check_local_RMS_error(1,2) # 1% both ways, average the error
        RMS_local_L3[i] = np.mean(data)
        phi = Func.eval_surface()
        MAX_surf_L3[i] = np.max(abs(phi))/Func.Bbox_diag
        RMS_surf_L3[i] = np.sqrt(np.mean(phi**2))/Func.Bbox_diag
        max_Fnorm_L3[i] = Func.get_RMS_Fnorm()
        Runtime_L3[i] = Func.runtime
        print('Finished L3='+str(L3)+' dataset')
    L1_data = L1_data[4:]
    L2_data = L2_data[4:]
    L3_data = L3_data[4:]
    Energy1_L1 = Energy1_L1[4:]
    Energy1_L2 = Energy1_L2[4:]
    Energy1_L3 = Energy1_L3[4:]
    Energy2_L1 = Energy2_L1[4:]
    Energy2_L2 = Energy2_L2[4:]
    Energy2_L3 = Energy2_L3[4:]
    Energy3_L1 = Energy3_L1[4:]
    Energy3_L2 = Energy3_L2[4:]
    Energy3_L3 = Energy3_L3[4:]
    RMS_surf_L1 = RMS_surf_L1[4:]
    RMS_surf_L2 = RMS_surf_L2[4:]
    RMS_surf_L3 = RMS_surf_L3[4:]
    MAX_surf_L1 = MAX_surf_L1[4:]
    MAX_surf_L2 = MAX_surf_L2[4:]
    MAX_surf_L3 = MAX_surf_L3[4:]
    RMS_local_L1 = RMS_local_L1[4:]
    RMS_local_L2 = RMS_local_L2[4:]
    RMS_local_L3 = RMS_local_L3[4:]
    max_Fnorm_L1 = max_Fnorm_L1[4:]
    max_Fnorm_L2 = max_Fnorm_L2[4:]
    max_Fnorm_L3 = max_Fnorm_L3[4:]
    Runtime_L1 = Runtime_L1[4:]
    Runtime_L2 = Runtime_L2[4:]
    Runtime_L3 = Runtime_L3[4:]

    sns.set(style='ticks')
    fig1= plt.figure(figsize=(5,4),dpi=180)
    ax1 = plt.axes()
    ax1.loglog(L1_data,max_Fnorm_L1,'*-',linewidth=2,markersize=12,color='tab:orange',label='Curvature')
    ax1.loglog(L1_data,RMS_local_L1,'D-',linewidth=2,markersize=6,color='tab:red',label='Local $(\pm 0.01)$')
    ax1.loglog(L1_data,RMS_surf_L1,'o-', linewidth=2,markersize=6,color='tab:blue',label='Surface')
    # ax1.loglog(L1_data,Runtime_L1/Runtime_L1[i1],'.--',markersize=8,label='Optimization time')
    # ax1.loglog(L1_data,MAX_surf_L1,'.-',markersize=8,color='tab:cyan',label='Max Surface Error')
    ax1.set_xlabel('$\lambda_1$',fontsize=16)
    ax1.set_ylabel('RMS Error',fontsize=16)
    ax1.set_xticks(L1_data)
    ax1.legend(loc='upper left',framealpha=1,edgecolor='black',facecolor='white',fontsize=12)
    ax1.set_ylim(3e-4,1e2)
    ax1.grid()
    sns.despine()
    plt.tight_layout()
    set_fonts()
    plt.savefig('PDF_figures/L1.pdf',bbox_inches='tight')

    sns.set(style='ticks')
    fig2= plt.figure(figsize=(5,4),dpi=180)
    ax2 = plt.axes()
    ax2.loglog(L2_data,max_Fnorm_L2,'*-',linewidth=2,markersize=12,color='tab:orange',label='Curvature')
    ax2.loglog(L2_data,RMS_local_L2,'D-',linewidth=2,markersize=6,color='tab:red',label='Local $(\pm 0.01)$')
    ax2.loglog(L2_data,RMS_surf_L2, 'o-',linewidth=2,markersize=6,color='tab:blue',label='Surface')
    # ax2.loglog(L2_data,Runtime_L2/Runtime_L2[i2],'.--',markersize=8,label='Optimization time')
    # ax2.loglog(L2_data,MAX_surf_L2,'.-',markersize=8,color='tab:cyan',label='Max Surface')
    ax2.set_xlabel('$\lambda_2$',fontsize=16)
    ax2.set_ylabel('RMS Error',fontsize=16)
    ax2.set_xticks(L2_data)
    ax2.legend(loc='upper left',framealpha=1,edgecolor='black',facecolor='white',fontsize=12)
    ax2.set_ylim(3e-4,1e2)
    ax2.grid()
    sns.despine()
    plt.tight_layout()
    set_fonts()
    plt.savefig('PDF_figures/L2.pdf',bbox_inches='tight')

    sns.set(style='ticks')
    fig3= plt.figure(figsize=(5,4),dpi=180)
    ax3 = plt.axes()
    ax3.loglog(L3_data,max_Fnorm_L3,'*-',linewidth=2,markersize=12,color='tab:orange',label='Curvature')
    ax3.loglog(L3_data,RMS_local_L3,'D-',linewidth=2,markersize=6,color='tab:red',label='Local $(\pm 0.01)$')
    ax3.loglog(L3_data,RMS_surf_L3, 'o-',linewidth=2,markersize=6,color='tab:blue',label='Surface')
    # ax3.loglog(L3_data,Runtime_L3/Runtime_L3[i3],'.--',markersize=8,label='Optimization time')
    # ax3.loglog(L3_data,MAX_surf_L3,'.-',markersize=8,color='tab:cyan',label='Max Surface')
    ax3.set_xlabel('$\lambda_3$',fontsize=16)
    ax3.set_ylabel('RMS Error',fontsize=16)
    ax3.set_xticks(L3_data)
    ax3.legend(loc='upper left',framealpha=1,edgecolor='black',facecolor='white',fontsize=12)
    ax3.set_ylim(3e-4,1e2)
    ax3.grid()
    sns.despine()
    plt.tight_layout()
    set_fonts()
    plt.savefig('PDF_figures/L3.pdf',bbox_inches='tight')

if mode == 'Plot_data':
    res = 30
    ep_max = 5 # BBox diag %

    max_surf_error = np.zeros(len(pt_data))
    RMS_surf_error = np.zeros(len(pt_data))

    plt.figure()
    ax = plt.axes()
    for i,num_pts in enumerate(pt_data):
        Func = pickle.load( open( "SAVED_DATA/Opt_"+file+"_"+str(num_pts)+".pkl", "rb" ) )
        ep_range,data = Func.check_local_RMS_error(ep_max,res)
        ax.plot(ep_range,data,'-',label=("$N_{\Gamma}$ = "+str(num_pts)))
        phi = Func.eval_surface()
        max_surf_error[i] = np.max(abs(phi))/Func.Bbox_diag
        RMS_surf_error[i] = np.sqrt(np.mean(phi**2))/Func.Bbox_diag
        print('plotted '+str(num_pts)+' point dataset')
    ax.set_xlabel("$\epsilon$ as a $\%$ of BBox diag")
    ax.set_ylabel("Normalized RMS")
    ax.set_title('Local distance error for '+file+' model (diag ='+str(np.round(Func.Bbox_diag,decimals=2))+')')
    ax.set_xlim(-ep_max,ep_max)
    ax.legend(loc='upper left')

    plt.figure()
    ax = plt.axes()
    ax.plot(pt_data,RMS_surf_error,'.-',label=('RMS'))
    ax.plot(pt_data,max_surf_error,'.-',label=('Maximum'))
    ax.set_xlabel("$N_{\Gamma}$")
    ax.set_ylabel("Normalized Error")
    ax.set_title('Surface Error for '+file+' model (diag ='+str(np.round(Func.Bbox_diag,decimals=2))+')')
    ax.legend(loc='upper center')

if mode == 'Comp_pen_strict':
    # Bunny ONLY
    pt_data_strict = [77,108,252,297,327,377]
    pt_data_pen = [77,108,201,252,412,677,1002,2002,3002,4002,5002,10002,25002,40802,63802,100002]

    orders = ['o4']

    max_err_strict = np.zeros(len(pt_data_strict))
    RMS_err_strict = np.zeros(len(pt_data_strict))
    max_err_pen = np.zeros((len(orders),len(pt_data_pen)))
    RMS_err_pen = np.zeros((len(orders),len(pt_data_pen)))
    styles = ['.-','.--','.:']
    fig = plt.figure(figsize=(6,5),dpi=140)
    ax = plt.axes()
    for j,order in enumerate(orders):
        for i,num_pts in enumerate(pt_data_pen):
            Func = pickle.load( open( "SAVED_DATA/Opt_"+order+"Bunny_pen36_"+str(num_pts)+".pkl", "rb" ) )
            phi = Func.eval_surface()
            max_err_pen[j,i] = np.max(abs(phi))/Func.Bbox_diag
            RMS_err_pen[j,i] = np.sqrt(np.mean(phi**2))/Func.Bbox_diag
        ax.loglog(pt_data_pen,RMS_err_pen[j,:],('b'+styles[j]),label=('Penalization RMS'))
        ax.loglog(pt_data_pen,max_err_pen[j,:],('r'+styles[j]),label=('Penalization Max'))
        print('plotted '+str(order)+' dataset')
    for i,num_pts in enumerate(pt_data_strict):
        Func = pickle.load( open( "SAVED_DATA/Opt_o4Bunny36_"+str(num_pts)+".pkl", "rb" ) )
        phi = Func.eval_surface()
        max_err_strict[i] = np.max(abs(phi))/Func.Bbox_diag
        RMS_err_strict[i] = np.sqrt(np.mean(phi**2))/Func.Bbox_diag
        print('plotted '+str(num_pts)+' point dataset')

    ax.loglog(pt_data_strict,RMS_err_strict,'b.--',label=('Constrained RMS'))
    ax.loglog(pt_data_strict,max_err_strict,'r.--',label=('Constrained Max'))
    ax.set_xlabel("$N_{\Gamma}$")
    ax.set_ylabel("Normalized Surface Error")
    # ax.set_title('Surface Error for '+file+' model (diag ='+str(np.round(Func.Bbox_diag,decimals=2))+')')
    ax.legend(loc='upper right')
    print('Bounding Box Diagonal: ',Func.Bbox_diag)
    plt.tight_layout()
    set_fonts()
    plt.savefig('PDF_figures/Error_Pen_Const.pdf',bbox_inches='tight')

if mode == 'Comp_err_order':
    # Bunny ONLY
    pt_data_pen = [77,108,201,252,412,677,1002,2002,3002,4002,5002,10002] #,25002,40802,63802,100002]

    orders = ['o3','o4','o5']

    max_err_pen = np.zeros((len(orders),len(pt_data_pen)))
    RMS_err_pen = np.zeros((len(orders),len(pt_data_pen)))
    styles = ['-','--',':']
    sns.set(style='ticks')
    plt.figure(figsize=(7,6),dpi=180)
    ax = plt.axes()
    for j,order in enumerate(orders):
        for i,num_pts in enumerate(pt_data_pen):
            Func = pickle.load( open( "SAVED_DATA/Opt_"+order+"Bunny_pen40_"+str(num_pts)+".pkl", "rb" ) )
            phi = Func.eval_surface()
            max_err_pen[j,i] = np.max(abs(phi))/Func.Bbox_diag
            RMS_err_pen[j,i] = np.sqrt(np.mean(phi**2))/Func.Bbox_diag
        print('Finished order '+str(order[1])+' dataset')
        ax.loglog(pt_data_pen,RMS_err_pen[j,:],('bs'+styles[j]),linewidth=2,markersize=6,label=('Degree '+str(int(order[1])-1)+' RMS'))
    for j,order in enumerate(orders):
        ax.loglog(pt_data_pen,max_err_pen[j,:],('r.'+styles[j]),linewidth=2,markersize=9,label=('Degree '+str(int(order[1])-1)+' Max'))
    ax.set_xlabel("$N_{\Gamma}$",fontsize=14)
    ax.set_ylabel("Normalized Error",fontsize=14)
    # ax.set_title('Surface Error for '+file+' model (diag ='+str(np.round(Func.Bbox_diag,decimals=2))+')')
    ax.legend(fontsize=12,loc='upper right',framealpha=1,edgecolor='black',facecolor='white')
    ax.grid()
    sns.despine()
    print('Bounding Box Diagonal: ',Func.Bbox_diag)
    plt.tight_layout()
    set_fonts()
    plt.savefig('PDF_figures/Error_v_Order.pdf',bbox_inches='tight')

if mode == 'Hicken_v_Splines':

    ep_data = [0.5, 1]
    k = 20
    rho = 20
    num_samples = 50000
    bunny_exact = pickle.load( open( "SAVED_DATA/_Bunny_data_exact_.pkl", "rb" ) )
    exact_dataset = KDTree(bunny_exact[0])
    np.random.seed(1)
    indx = np.random.randint(np.size(bunny_exact[0],0), size=num_samples)
    down_exact_pts = bunny_exact[0][indx,:]
    down_exact_nrm = bunny_exact[1][indx,:]
    pt_data = [677,1002,2002,3002,4002,5002,10002,25002,40802,63802,100002] # 252,412

    RMS_err_Bsplines_fine = np.zeros(len(pt_data))
    RMS_err_KSmethod = np.zeros(len(pt_data))
    ep_error_KSmethod = np.zeros((len(pt_data),len(ep_data)))
    ep_error_Bsplines = np.zeros((len(pt_data),len(ep_data)))
    for i,num_pts in enumerate(pt_data):
        Func = pickle.load( open( "SAVED_DATA/Opt_o4Bunny_pen40_"+str(num_pts)+".pkl", "rb" ) )
        phi = Func.eval_pts(down_exact_pts)
        RMS_err_Bsplines_fine[i] = np.sqrt(np.mean(phi**2))/Func.Bbox_diag

        dataset = KDTree(Func.surf_pts)
        phi = KS_eval(down_exact_pts,dataset,Func.normals,k,rho)
        RMS_err_KSmethod[i] = np.sqrt(np.mean(phi**2))/Func.Bbox_diag

        for j,ep in enumerate(ep_data):
            i_pts1 = down_exact_pts + ep/100 * Func.Bbox_diag * down_exact_nrm
            i_pts2 = down_exact_pts - ep/100 * Func.Bbox_diag * down_exact_nrm

            i_pts = np.vstack((i_pts1,i_pts2))
            phi_ex,_ = exact_dataset.query(i_pts,k=1)
            phi = Func.eval_pts(i_pts)
            ep_error_Bsplines[i,j] = np.sqrt(np.mean( (abs(phi)-phi_ex)**2 ))/Func.Bbox_diag
            phi = KS_eval(i_pts,dataset,Func.normals,k,rho)
            ep_error_KSmethod[i,j] = np.sqrt(np.mean( (abs(phi)-phi_ex)**2 ))/Func.Bbox_diag
        print('finished ng={}'.format(num_pts))

    ##################################################################

    sns.set(style='ticks')
    set_fonts()
    fig1 = plt.figure(figsize=(5,5),dpi=160)
    ax1 = plt.axes()
    ax1.loglog(pt_data,RMS_err_Bsplines_fine,'.-',color='tab:blue',markersize=14,linewidth=2,label=('Our Method'))
    ax1.loglog(pt_data,RMS_err_KSmethod,'.--',color='tab:orange',markersize=14,linewidth=2,label=('Explicit Method'))
    ax1.set_xlabel('$N_{\Gamma}$',fontsize=14)
    ax1.set_ylabel('RMS Error',fontsize=14)
    ax1.legend(fontsize=12,framealpha=1,edgecolor='black',facecolor='white')
    ax1.set_ylim(2e-5,2e-2)
    ax1.grid()
    sns.despine()
    plt.tight_layout()
    if save_figures:
        plt.savefig('PDF_figures/EXvBa.pdf',bbox_inches='tight')

    sns.set(style='ticks')
    set_fonts(legendfont=12,axesfont=18)
    fig2 = plt.figure(figsize=(5,5),dpi=160)
    ax2 = plt.axes()
    styles_bspline = ['.-','s-']
    styles_KSmethod = ['.--','s--']
    for i,ep in enumerate(ep_data):
        if i==0:
            ax2.loglog(pt_data,ep_error_Bsplines[:,i],styles_bspline[i],markersize=14,linewidth=2,color='tab:blue',
                label=('Our method ($\pm${})'.format(ep/100)))
        elif i==1:
            ax2.loglog(pt_data,ep_error_Bsplines[:,i],styles_bspline[i],markersize=7,linewidth=2,color='tab:blue',
                label=('Our method ($\pm${})'.format(ep/100)))
    for i,ep in enumerate(ep_data):
        if i==0:
            ax2.loglog(pt_data,ep_error_KSmethod[:,i],styles_KSmethod[i],markersize=14,linewidth=2,color='tab:orange',
                label=('Explicit method ($\pm${})'.format(ep/100)))
        elif i==1:
            ax2.loglog(pt_data,ep_error_KSmethod[:,i],styles_KSmethod[i],markersize=7,linewidth=2,color='tab:orange',
                label=('Explicit method ($\pm${})'.format(ep/100)))
    ax2.set_xlabel('$N_{\Gamma}$',fontsize=14)
    ax2.set_ylabel('RMS Error',fontsize=14)
    ax2.set_ylim(9e-5,2e-2)
    ax2.legend(fontsize=12,framealpha=1,edgecolor='black',facecolor='white')
    ax2.grid()
    sns.despine()
    plt.tight_layout()
    if save_figures:
        plt.savefig('PDF_figures/EXvBb.pdf',bbox_inches='tight')

    # print("RMS Bsplines")
    # print(RMS_err_Bsplines_fine)
    # print("RMS Hicken")
    # print(RMS_err_KSmethod)
    
if mode == 'Comp_time':
    ep_data = [0.5, 1.0]
    k = 40
    rho = 20
    num_samples = 2000
    bunny_exact = pickle.load( open( "SAVED_DATA/_Bunny_data_exact_.pkl", "rb" ) )
    down_exact_pts = bunny_exact[0][::2]
    down_exact_nrm = bunny_exact[1][::2]

    np.random.seed(1)
    indx = np.random.randint(np.size(bunny_exact[0],0), size=num_samples)
    down_exact_pts = bunny_exact[0][indx,:]
    down_exact_nrm = bunny_exact[1][indx,:]
    pt_data = [677,1002,2002,3002,4002,5002,10002,25002,40802,63802,100002]

    # RMS_err_Bsplines_fine = np.zeros(len(pt_data))
    # RMS_err_KSmethod = np.zeros(len(pt_data))
    # time_KSmethod = np.zeros(len(pt_data))
    # time_Bsplines_1000 = np.zeros(len(pt_data))
    # time_FredMethod = np.zeros(len(pt_data))
    # ep_error_KSmethod = np.zeros((len(pt_data),len(ep_data)))
    # ep_error_Bsplines = np.zeros((len(pt_data),len(ep_data)))
    # for i,num_pts in enumerate(pt_data):
    #     Func = pickle.load( open( "SAVED_DATA/Opt_o4Bunny_pen40_"+str(num_pts)+".pkl", "rb" ) )
    #     t1 = time.perf_counter()
    #     phi = Func.eval_pts(down_exact_pts)
    #     t2 = time.perf_counter()
    #     time_Bsplines_1000[i] = (t2-t1) / len(phi)

    #     t1 = time.perf_counter()
    #     f = Freds_Method(down_exact_pts,Func.surf_pts,Func.normals)
    #     t2 = time.perf_counter()
    #     time_FredMethod[i] = (t2-t1) / len(down_exact_pts)

    #     dataset = KDTree(Func.surf_pts)
    #     t1 = time.perf_counter()
    #     phi = KS_eval(down_exact_pts,dataset,Func.normals,k,rho)
    #     t2 = time.perf_counter()
    #     time_KSmethod[i] = (t2-t1) / len(phi)

    #     print('finished ng={}'.format(num_pts))
    
    time_Bsplines_1000 = [3.03260e-06,2.82635e-06,2.77640e-06,3.01960e-06,
        3.11460e-06,2.92115e-06,2.94500e-06,3.05460e-06,3.09575e-06,3.25665e-06,2.64320e-06]
    time_KSmethod = [6.931800e-06,7.282700e-06,8.335100e-06,
        7.654400e-06,7.553100e-06,7.468300e-06,7.896700e-06,8.100350e-06,
        9.152785e-06,8.331000e-06,9.393050e-06]
    time_FredMethod = [6.82422000e-05,1.00678750e-04,
        2.05626600e-04,3.19813600e-04,4.11701150e-04,5.57543150e-04,
        1.00118260e-03,2.50914620e-03,4.08993435e-03,6.66776455e-03,
        1.09231676e-02]

    P_OM = np.polyfit(np.log(pt_data),np.log(time_Bsplines_1000),1)
    P_EM = np.polyfit(np.log(pt_data),np.log(time_KSmethod),1)
    P_CM = np.polyfit(np.log(pt_data),np.log(time_FredMethod),1)

    bf_OM = np.poly1d(P_OM)
    bf_EM = np.poly1d(P_EM)
    bf_CM = np.poly1d(P_CM)

    sns.set(style='ticks')
    fig = plt.figure(figsize=(5.2,5),dpi=160)
    ax1 = plt.axes()
    ax1.loglog(pt_data,np.exp(bf_OM(np.log(pt_data))), 'k-',linewidth=6,alpha=0.13)
    ax1.loglog(pt_data,np.exp(bf_EM(np.log(pt_data))), 'k-',linewidth=6,alpha=0.13)
    ax1.loglog(pt_data,np.exp(bf_CM(np.log(pt_data))), 'k-',linewidth=6,alpha=0.13)

    ax1.loglog(pt_data,time_Bsplines_1000,'.-',label=('Our Method'),color='tab:blue',markersize=10,linewidth=2)
    ax1.loglog(pt_data,time_KSmethod,'.:',label=('Explicit Method'),color='tab:orange',markersize=10,linewidth=2)
    ax1.loglog(pt_data,time_FredMethod,'.--',label=('Previous Method'),color='tab:green',markersize=10,linewidth=2)

    plt.text(1.5e3,5e-4,'$\mathcal{O}(N_{\Gamma})$',        fontsize=14)
    plt.text(1.7e4,1.2e-5,'$\mathcal{O}(k$log$(N_{\Gamma}))$',fontsize=14)
    plt.text(5e4,4e-6,'$\mathcal{O}(1)$',            fontsize=14)

    ax1.set_xlabel('$N_{\Gamma}$',fontsize=14)
    ax1.set_ylabel('Evaluation Time per point (sec)',fontsize=14)
    ax1.legend(fontsize=14,framealpha=1,edgecolor='black',facecolor='white')
    ax1.grid()
    sns.despine()
    plt.tight_layout()

    set_fonts()

    plt.savefig('PDF_figures/Comp_time.pdf',bbox_inches='tight')

if mode == 'normalized_Hicken_v_Splines':

    ep_data = [0.5, 1]
    k = 20
    rho = 20
    num_samples = 10000
    bunny_exact = pickle.load( open( "SAVED_DATA/_Bunny_data_exact_.pkl", "rb" ) )
    exact_dataset = KDTree(bunny_exact[0])
    np.random.seed(1)
    indx = np.random.randint(np.size(bunny_exact[0],0), size=num_samples)
    down_exact_pts = bunny_exact[0][indx,:]
    down_exact_nrm = bunny_exact[1][indx,:]
    pt_data = [252,412,677,1002,2002,3002,4002,5002,10002,25002,40802,63802,100002]

    RMS_err_Bsplines_fine = np.zeros(len(pt_data))
    RMS_err_KSmethod = np.zeros(len(pt_data))
    ep_error_KSmethod = np.zeros((len(pt_data),len(ep_data)))
    ep_error_Bsplines = np.zeros((len(pt_data),len(ep_data)))
    Hicken_normalizer = np.zeros(len(pt_data))
    for i,num_pts in enumerate(pt_data):
        Func = pickle.load( open( "SAVED_DATA/Opt_o4Bunny_pen40_"+str(num_pts)+".pkl", "rb" ) )
        normalizer = np.linalg.norm(np.diff(Func.dimensions).flatten()/Func.num_cps)
        phi = Func.eval_pts(down_exact_pts)
        RMS_err_Bsplines_fine[i] = np.sqrt(np.mean(phi**2))/Func.Bbox_diag

        dataset = KDTree(Func.surf_pts)
        runsum = 0
        all_dh,_ = dataset.query(Func.surf_pts,k=2)
        Hicken_normalizer[i] = np.mean(all_dh[:,1])
        phi = KS_eval(down_exact_pts,dataset,Func.normals,k,rho)
        RMS_err_KSmethod[i] = np.sqrt(np.mean(phi**2))/Func.Bbox_diag

        for j,ep in enumerate(ep_data):
            i_pts1 = down_exact_pts + ep/100 * Func.Bbox_diag * down_exact_nrm
            i_pts2 = down_exact_pts - ep/100 * Func.Bbox_diag * down_exact_nrm

            i_pts = np.vstack((i_pts1,i_pts2))
            phi_ex,_ = exact_dataset.query(i_pts,k=1)
            phi = Func.eval_pts(i_pts)
            ep_error_Bsplines[i,j] = np.sqrt(np.mean( (abs(phi)-phi_ex)**2 ))/Func.Bbox_diag
            phi = KS_eval(i_pts,dataset,Func.normals,k,rho)
            ep_error_KSmethod[i,j] = np.sqrt(np.mean( (abs(phi)-phi_ex)**2 ))/Func.Bbox_diag
        print('finished ng={}'.format(num_pts))

    ##################################################################

    sns.set(style='ticks')
    set_fonts()
    fig1 = plt.figure(figsize=(5.2,5),dpi=160)
    ax1 = plt.axes()
    ax1.loglog(pt_data,RMS_err_Bsplines_fine/normalizer,'.-',color='tab:blue',markersize=14,linewidth=2,label=('Our Method'))
    ax1.loglog(pt_data,RMS_err_KSmethod/Hicken_normalizer,'.--',color='tab:orange',markersize=14,linewidth=2,label=('Explicit Method'))
    ax1.set_xlabel('$N_{\Gamma}$',fontsize=14)
    ax1.set_ylabel('Normalized RMS Error',fontsize=14)
    ax1.legend(fontsize=12,framealpha=1,edgecolor='black',facecolor='white')
    ax1.set_ylim(8e-6,2e-2)
    ax1.grid()
    sns.despine()
    plt.tight_layout()
    if save_figures:
        plt.savefig('PDF_figures/normalized_EXvBa.pdf',bbox_inches='tight')

    sns.set(style='ticks')
    set_fonts(legendfont=12,axesfont=18)
    fig2 = plt.figure(figsize=(5.2,5),dpi=160)
    ax2 = plt.axes()
    styles_bspline = ['.-','s-']
    styles_KSmethod = ['.--','s--']
    for i,ep in enumerate(ep_data):
        if i==0:
            ax2.loglog(pt_data,ep_error_Bsplines[:,i]/normalizer,styles_bspline[i],markersize=14,linewidth=2,color='tab:blue',
                label=('Our method ($\pm${})'.format(ep/100)))
        elif i==1:
            ax2.loglog(pt_data,ep_error_Bsplines[:,i]/normalizer,styles_bspline[i],markersize=7,linewidth=2,color='tab:blue',
                label=('Our method ($\pm${})'.format(ep/100)))
    for i,ep in enumerate(ep_data):
        if i==0:
            ax2.loglog(pt_data,ep_error_KSmethod[:,i]/Hicken_normalizer,styles_KSmethod[i],markersize=14,linewidth=2,color='tab:orange',
                label=('Explicit method ($\pm${})'.format(ep/100)))
        elif i==1:
            ax2.loglog(pt_data,ep_error_KSmethod[:,i]/Hicken_normalizer,styles_KSmethod[i],markersize=7,linewidth=2,color='tab:orange',
                label=('Explicit method ($\pm${})'.format(ep/100)))
    ax2.set_xlabel('$N_{\Gamma}$',fontsize=14)
    ax2.set_ylabel('Normalized RMS Error',fontsize=14)
    ax2.set_ylim(8e-6,2e-2)
    ax2.legend(fontsize=12,framealpha=1,edgecolor='black',facecolor='white')
    ax2.grid()
    sns.despine()
    plt.tight_layout()
    if save_figures:
        plt.savefig('PDF_figures/normalized_EXvBb.pdf',bbox_inches='tight')

    # print("RMS Bsplines")
    # print(RMS_err_Bsplines_fine)
    # print("RMS Hicken")
    # print(RMS_err_KSmethod)

if mode == 'plot_point_cloud':
    surf_pts, normals = extract_stl_info('stl-files/Bunny_2002.stl')
    ax = plt.axes(projection='3d')
    ax.plot3D(surf_pts[:,0],surf_pts[:,1],surf_pts[:,2],'k.',markersize=10)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

plt.show()