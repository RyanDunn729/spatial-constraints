from modules.Bspline_Volume import BSplineVolume
from utils.read_stl import extract_stl_info
from utils.Lin_et_al import Lin_et_al_Method
from skimage.measure import marching_cubes
from geom_shapes.ellipsoid import Ellipsoid
from utils.Hicken_Kaur import KS_eval, Continuous_Hicken_eval
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
# file = 'o3Bunny40'
# file = 'o4Bunny35'
# file = 'o3Bunny28'
file = 'o4Bunny28'
# file = 'o5Bunny28'
# file = 'o5Bunny40'
# file = 'o6Bunny40' 
# file = 'o4Bunny34'
# file = 'o4Bunny30'

# flag = 'Dragon'

### Plot Data Mode ###
mode = 'Hicken_analysis'
isocontour = 0
mode = 'Bspline_analysis'
# mode = 'Bspline_analysis_vary_L1'
# mode = 'Bspline_analysis_vary_L2'
# mode = 'Bspline_analysis_vary_L3'
# mode = 'Visualize_lambdas_energies'
mode = 'Visualize_lambdas'
# mode = 'Plot_data'
# mode = 'Comp_pen_strict'
# mode = 'Comp_err_order'
mode = 'Hicken_v_Splines'
# mode = 'normalized_Hicken_v_Splines'
mode = 'Comp_time'
# mode = 'plot_point_cloud'
print(mode)

# BSpline Volume Parameters #
dim = 3
order = int(file[1])
border = 0.15
max_cps = int(file[-2:])
iter = 1
# Curvature, normals, level set
L = [1e-1, 10., 1000.]

data = np.logspace(-4,4,9)

if file[2:-2]=='Heart':
    tol = 1e-4
    exact = extract_stl_info("geom_shapes/Heart_exact.stl")
    main_name = 'geom_shapes/Heart_'
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
elif file[2:-2]=='Bunny':
    tol = 1e-4
    exact = extract_stl_info("geom_shapes/Bunny_exact.stl")
    main_name = 'geom_shapes/Bunny_'
    group = 6
    pt_data = [500,808,1310,2120,3432,5555,9000,14560,25000,38160,64000,100000]

if mode=='Hicken_analysis':
    file = file[2:-2]
    res = 180
    k = 20
    rho = 1e-1
    for num_pts in [100000]:
        if file[2:-2] =='Ellipsoid':
            surf_pts = e.points(num_pts)
            normals = e.unit_pt_normals(num_pts)
        # elif flag == 'Dragon':
        #     filename = 'geom_shapes/dragon_100k.stl'
        #     surf_pts, normals = extract_stl_info(filename)
        else:
            filename = main_name+str(num_pts)+'.stl'
            surf_pts, normals = extract_stl_info(filename)
        lower = np.min(surf_pts,axis=0)
        upper = np.max(surf_pts,axis=0)
        diff = upper-lower
        dimensions = np.stack((lower, upper),axis=1)
        x = dimensions[0]
        y = dimensions[1]
        z = dimensions[2]
        pts = np.zeros((np.product(res**3), 3))
        pts[:, 0] = np.einsum('i,j,k->ijk', np.linspace(x[0],x[1],res), np.ones(res),np.ones(res)).flatten()
        pts[:, 1] = np.einsum('i,j,k->ijk', np.ones(res), np.linspace(y[0],y[1],res),np.ones(res)).flatten()
        pts[:, 2] = np.einsum('i,j,k->ijk', np.ones(res), np.ones(res),np.linspace(z[0],z[1],res)).flatten()
        dataset = KDTree(surf_pts)
        phi = KS_eval(pts,dataset,normals,k,rho)
        phi = phi.reshape((res,res,res))/np.linalg.norm(diff)
        for isocontour in [-0.005,-0.01,0,0.005,0.01]:
            verts, faces,_,_ = marching_cubes(phi, isocontour)
            verts = verts*np.diff(dimensions).flatten()/(res-1) + dimensions[:,0]
            # d_i,_ = dataset.query(verts,k=1)
            # print('Exact representation RMS Error: ',np.sqrt(np.mean((abs(isocontour)-d_i))**2)/len(d_i))
            surf = Mesh(np.zeros(faces.shape[0], dtype=Mesh.dtype))
            for i, f in enumerate(faces):
                    for j in range(3):
                            surf.vectors[i][j] = verts[f[j],:]
            surf.save('SAVED_DATA/Hick_'+file+'_' + str(isocontour) + '.stl')
            print('Finished ',str(isocontour),' point Hicken File')

if mode=='Bspline_analysis':
    # pt_data = [pt_data[group-1], pt_data[-group]]
    m = model(max_cps,border,dim,tol,exact)
    for num_pts in pt_data:
        if file[2:-2] =='Ellipsoid':
            surf_pts = e.points(num_pts)
            normals = e.unit_pt_normals(num_pts)
        else:
            surf_pts, normals = extract_stl_info(main_name+str(num_pts)+'.stl')
        Func = m.inner_solve(surf_pts, normals, L[0], L[1], L[2], order)
        pickle.dump(Func, open( "SAVED_DATA/Opt_"+file+"_"+str(num_pts)+".pkl","wb"))
        del surf_pts,normals,Func
        print('Finished ',file,str(num_pts),' Optimization')

if mode == 'Bspline_analysis_vary_L1':
    L = [1., 1., 1.]
    L1_data = L[0]*data
    print(L1_data)
    L2 = L[1]
    L3 = L[2]
    m = model(max_cps,border,dim,tol,exact)
    surf_pts, normals = extract_stl_info('geom_shapes/Bunny_25000.stl')
    for i,L1 in enumerate(L1_data):
        if i<6:
            continue
        Func = m.inner_solve(surf_pts, normals, L1, L2, L3, order)
        pickle.dump(Func, open( "SAVED_DATA/Opt_Bunny_L1_"+str(i)+".pkl","wb"))
        del Func
        print('Finished L1 =',str(L1),'Optimization')

if mode == 'Bspline_analysis_vary_L2':
    L = [1., 1., 1.]
    L1 = L[0]
    L2_data = L[1]*data
    print(L2_data)
    L3 = L[2]
    m = model(max_cps,border,dim,tol,exact)
    surf_pts, normals = extract_stl_info('geom_shapes/Bunny_25000.stl')
    for i,L2 in enumerate(L2_data):
        Func = m.inner_solve(surf_pts, normals, L1, L2, L3, order)
        pickle.dump(Func, open( "SAVED_DATA/Opt_Bunny_L2_"+str(i)+".pkl","wb"))
        del Func
        print('Finished L2 =',str(L2),'Optimization')

if mode == 'Bspline_analysis_vary_L3':
    L = [1., 1., 1.]
    L1 = L[0]
    L2 = L[1]
    L3_data = L[2]*data
    print(L3_data)
    m = model(max_cps,border,dim,tol,exact)
    surf_pts, normals = extract_stl_info('geom_shapes/Bunny_25000.stl')
    for i,L3 in enumerate(L3_data):
        Func = m.inner_solve(surf_pts, normals, L1, L2, L3, order)
        pickle.dump(Func, open( "SAVED_DATA/Opt_Bunny_L3_"+str(i)+".pkl","wb"))
        del Func
        print('Finished L3 =',str(L3),'Optimization')

if mode == 'Visualize_lambdas_energies':
    L1_data = data
    L2_data = data
    L3_data = data
    RMS_surf_L1 = np.ones(len(L1_data))
    MAX_surf_L1 = np.ones(len(L1_data))
    RMS_local_L1 = np.ones(len(L1_data))
    max_Fnorm_L1 = np.ones(len(L1_data))
    Energy1_L1 = np.ones(len(L1_data))
    Energy2_L1 = np.ones(len(L1_data))
    Energy3_L1 = np.ones(len(L1_data))
    Runtime_L1 = np.ones(len(L1_data))
    for i,L1 in enumerate(L1_data):
        Func = pickle.load( open( "SAVED_DATA/Opt_Bunny_L1_"+str(i)+".pkl", "rb" ) )
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
        Func = pickle.load( open( "SAVED_DATA/Opt_Bunny_L2_"+str(i)+".pkl", "rb" ) )
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
        Func = pickle.load( open( "SAVED_DATA/Opt_Bunny_L3_"+str(i)+".pkl", "rb" ) )
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

    sns.set(style='ticks')
    fig1, axs1 = plt.subplots(2,1,sharex=True,figsize=(6,7),dpi=180)
    fig1.subplots_adjust(hspace=0)
    axs1[0].loglog(L1_data,Energy3_L1/Energy3_L1[4],'*--',linewidth=2,markersize=12,color='tab:orange',label='$\mathcal{E}_1$')
    axs1[0].loglog(L1_data,Energy2_L1/Energy2_L1[4],'D--',linewidth=2,markersize=6,color='tab:red',label='$\mathcal{E}_2$')
    axs1[0].loglog(L1_data,Energy1_L1/Energy1_L1[4],'o--',linewidth=2,markersize=6,color='tab:blue',label='$\mathcal{E}_3$')
    axs1[0].set_ylabel('Energy about $\lambda_1={}$'.format(L[0]))
    axs1[0].set_xticks(L1_data)
    axs1[0].legend(loc='upper left',framealpha=1,edgecolor='black',facecolor='white')
    axs1[0].set_ylim(2e-2,1e3)
    axs1[0].grid()

    axs1[1].loglog(L1_data,max_Fnorm_L1,'*-',linewidth=2,markersize=12,color='tab:orange',label='Curvature')
    axs1[1].loglog(L1_data,RMS_local_L1,'D-',linewidth=2,markersize=6,color='tab:red',label='Local')
    axs1[1].loglog(L1_data,RMS_surf_L1,'o-', linewidth=2,markersize=6,color='tab:blue',label='Surface')
    # axs1[1].loglog(L1_data,Runtime_L1/Runtime_L1[4],'.--',markersize=8,label='Optimization time')
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
    axs2[0].loglog(L2_data,Energy3_L2/Energy3_L2[4],'*--',linewidth=2,markersize=12,color='tab:orange',label='$\mathcal{E}_1$')
    axs2[0].loglog(L2_data,Energy2_L2/Energy2_L2[4],'D--',linewidth=2,markersize=6,color='tab:red',label='$\mathcal{E}_2$')
    axs2[0].loglog(L2_data,Energy1_L2/Energy1_L2[4],'o--',linewidth=2,markersize=6,color='tab:blue',label='$\mathcal{E}_3$')
    axs2[0].set_ylabel('Energy about $\lambda_2={}$'.format(L[1]))
    axs1[0].set_xticks(L2_data)
    axs2[0].legend(loc='upper left',framealpha=1,edgecolor='black',facecolor='white')
    axs2[0].set_ylim(2e-2,1e3)
    axs2[0].grid()
    axs2[1].loglog(L2_data,max_Fnorm_L2,'*-',linewidth=2,markersize=12,color='tab:orange',label='Curvature')
    axs2[1].loglog(L2_data,RMS_local_L2,'D-',linewidth=2,markersize=6,color='tab:red',label='Local')
    axs2[1].loglog(L2_data,RMS_surf_L2, 'o-',linewidth=2,markersize=6,color='tab:blue',label='Surface')
    # axs2[1].loglog(L2_data,Runtime_L2/Runtime_L2[4],'.--',markersize=8,label='Optimization time')
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
    axs3[0].loglog(L3_data,Energy3_L3/Energy3_L3[4],'*--',linewidth=2,markersize=12,color='tab:orange',label='$\mathcal{E}_1$')
    axs3[0].loglog(L3_data,Energy2_L3/Energy2_L3[4],'D--',linewidth=2,markersize=6,color='tab:red',label='$\mathcal{E}_2$')
    axs3[0].loglog(L3_data,Energy1_L3/Energy1_L3[4],'o--',linewidth=2,markersize=6,color='tab:blue',label='$\mathcal{E}_3$')
    axs3[0].set_ylabel('Energy about $\lambda_3={}$'.format(L[2]))
    axs3[0].set_xticks(L3_data)
    axs3[0].legend(loc='upper left',framealpha=1,edgecolor='black',facecolor='white')
    axs3[0].set_ylim(2e-2,1e3)
    axs3[0].grid()
    axs3[1].loglog(L3_data,max_Fnorm_L3,'*-',linewidth=2,markersize=12,color='tab:orange',label='Curvature')
    axs3[1].loglog(L3_data,RMS_local_L3,'D-',linewidth=2,markersize=6,color='tab:red',label='Local')
    axs3[1].loglog(L3_data,RMS_surf_L3, 'o-',linewidth=2,markersize=6,color='tab:blue',label='Surface')
    # axs3[1].loglog(L3_data,Runtime_L3/Runtime_L3[4],'.--',markersize=8,label='Optimization time')
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
    
    plt.savefig('PDF_figures/L3.pdf',bbox_inches='tight')

if mode == 'Visualize_lambdas':
    L1_data = data
    L2_data = data
    L3_data = data
    RMS_surf_L1 = np.ones(len(L1_data))
    MAX_surf_L1 = np.ones(len(L1_data))
    RMS_local_L1 = np.ones(len(L1_data))
    max_Fnorm_L1 = np.ones(len(L1_data))
    Energy1_L1 = np.ones(len(L1_data))
    Energy2_L1 = np.ones(len(L1_data))
    Energy3_L1 = np.ones(len(L1_data))
    Runtime_L1 = np.ones(len(L1_data))
    for i,L1 in enumerate(L1_data):
        Func = pickle.load( open( "SAVED_DATA/Opt_Bunny_L1_"+str(i)+".pkl", "rb" ) )
        Energy1_L1[i] = Func.E_norm[0] # Measurement of the curvature energy
        Energy2_L1[i] = Func.E_norm[1] # Local energy
        Energy3_L1[i] = Func.E_norm[2] # Surf energy
        ep_range,data = Func.check_local_RMS_error(1,2) # 1% both ways, average the error
        RMS_local_L1[i] = np.mean(data)
        phi = Func.eval_surface()
        MAX_surf_L1[i] = np.max(abs(phi))/Func.Bbox_diag
        RMS_surf_L1[i] = np.sqrt(np.mean(phi**2))/Func.Bbox_diag
        # max_Fnorm_L1[i] = Func.get_RMS_Fnorm()
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
        Func = pickle.load( open( "SAVED_DATA/Opt_Bunny_L2_"+str(i)+".pkl", "rb" ) )
        Energy1_L2[i] = Func.E_norm[0] # Measurement of the curvature energy
        Energy2_L2[i] = Func.E_norm[1] # Surf energy
        Energy3_L2[i] = Func.E_norm[2] # local energy
        ep_range,data = Func.check_local_RMS_error(1,2) # 1% both ways, average the error
        RMS_local_L2[i] = np.mean(data)
        phi = Func.eval_surface()
        MAX_surf_L2[i] = np.max(abs(phi))/Func.Bbox_diag
        RMS_surf_L2[i] = np.sqrt(np.mean(phi**2))/Func.Bbox_diag
        # max_Fnorm_L2[i] = Func.get_RMS_Fnorm()
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
        Func = pickle.load( open( "SAVED_DATA/Opt_Bunny_L3_"+str(i)+".pkl", "rb" ) )
        Energy1_L3[i] = Func.E_norm[0] # Measurement of the curvature energy
        Energy2_L3[i] = Func.E_norm[1] # Surf energy
        Energy3_L3[i] = Func.E_norm[2] # local energy
        ep_range,data = Func.check_local_RMS_error(1,2) # 1% both ways, average the error
        RMS_local_L3[i] = np.mean(data)
        phi = Func.eval_surface()
        MAX_surf_L3[i] = np.max(abs(phi))/Func.Bbox_diag
        RMS_surf_L3[i] = np.sqrt(np.mean(phi**2))/Func.Bbox_diag
        # max_Fnorm_L3[i] = Func.get_RMS_Fnorm()
        Runtime_L3[i] = Func.runtime
        print('Finished L3='+str(L3)+' dataset')

    sns.set(style='ticks')
    fig1= plt.figure(figsize=(5,4),dpi=180)
    ax1 = plt.axes()
    # ax1.loglog(L1_data,max_Fnorm_L1,'*-',linewidth=2,markersize=12,color='tab:orange',label='Curvature')
    ax1.loglog(L1_data,RMS_local_L1,'D-',linewidth=2,markersize=6,color='tab:red',label='Off-Surface $(\pm 0.01)$')
    ax1.loglog(L1_data,RMS_surf_L1,'o-', linewidth=2,markersize=6,color='tab:blue',label='On-Surface')
    # ax1.loglog(L1_data,Runtime_L1/Runtime_L1[4],'.--',markersize=8,label='Optimization time')
    # ax1.loglog(L1_data,MAX_surf_L1,'.-',markersize=8,color='tab:cyan',label='Max Surface Error')
    ax1.set_xlabel('$\lambda_r$',fontsize=16)
    ax1.set_ylabel('RMS Error',fontsize=16)
    ax1.set_xticks(L1_data)
    ax1.legend(loc='upper left',framealpha=1,edgecolor='black',facecolor='white',fontsize=12)
    ax1.set_ylim(3e-4,1e2)
    ax1.set_ylim(1e-4,1e0)
    ax1.grid()
    sns.despine()
    plt.tight_layout()
    set_fonts()
    plt.savefig('PDF_figures/Lr.pdf',bbox_inches='tight')

    sns.set(style='ticks')
    fig2= plt.figure(figsize=(5,4),dpi=180)
    ax2 = plt.axes()
    # ax2.loglog(L2_data,max_Fnorm_L2,'*-',linewidth=2,markersize=12,color='tab:orange',label='Curvature')
    ax2.loglog(L2_data,RMS_local_L2,'D-',linewidth=2,markersize=6,color='tab:red',label='Off-Surface $(\pm 0.01)$')
    ax2.loglog(L2_data,RMS_surf_L2, 'o-',linewidth=2,markersize=6,color='tab:blue',label='On-Surface')
    # ax2.loglog(L2_data,Runtime_L2/Runtime_L2[4],'.--',markersize=8,label='Optimization time')
    # ax2.loglog(L2_data,MAX_surf_L2,'.-',markersize=8,color='tab:cyan',label='Max Surface')
    ax2.set_xlabel('$\lambda_n$',fontsize=16)
    ax2.set_ylabel('RMS Error',fontsize=16)
    ax2.set_xticks(L2_data)
    ax2.legend(loc='upper left',framealpha=1,edgecolor='black',facecolor='white',fontsize=12)
    ax2.set_ylim(3e-4,1e2)
    ax2.set_ylim(1e-4,1e0)
    ax2.grid()
    sns.despine()
    plt.tight_layout()
    set_fonts()
    plt.savefig('PDF_figures/Ln.pdf',bbox_inches='tight')

    sns.set(style='ticks')
    fig3= plt.figure(figsize=(5,4),dpi=180)
    ax3 = plt.axes()
    # ax3.loglog(L3_data,max_Fnorm_L3,'*-',linewidth=2,markersize=12,color='tab:orange',label='Curvature')
    ax3.loglog(L3_data,RMS_local_L3,'D-',linewidth=2,markersize=6,color='tab:red',label='Off-Surface $(\pm 0.01)$')
    ax3.loglog(L3_data,RMS_surf_L3, 'o-',linewidth=2,markersize=6,color='tab:blue',label='On-Surface')
    # ax3.loglog(L3_data,Runtime_L3/Runtime_L3[4],'.--',markersize=8,label='Optimization time')
    # ax3.loglog(L3_data,MAX_surf_L3,'.-',markersize=8,color='tab:cyan',label='Max Surface')
    ax3.set_xlabel('$\lambda_p$',fontsize=16)
    ax3.set_ylabel('RMS Error',fontsize=16)
    ax3.set_xticks(L3_data)
    ax3.legend(loc='upper left',framealpha=1,edgecolor='black',facecolor='white',fontsize=12)
    ax3.set_ylim(3e-4,1e2)
    ax3.set_ylim(1e-4,1e0)
    ax3.grid()
    sns.despine()
    plt.tight_layout()
    set_fonts()
    plt.savefig('PDF_figures/Lp.pdf',bbox_inches='tight')

    fig6_data = {}
    fig6_data["RMS_local_L1"] = RMS_local_L1
    fig6_data["RMS_surf_L1"] = RMS_surf_L1
    fig6_data["RMS_local_L2"] = RMS_local_L2
    fig6_data["RMS_surf_L2"] = RMS_surf_L2
    fig6_data["RMS_local_L3"] = RMS_local_L3
    fig6_data["RMS_surf_L3"] = RMS_surf_L3
    fig6_data["lambda_range"] = L1_data
    pickle.dump(fig6_data, open("fig6_data.pkl","wb"))

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

if mode == 'Comp_err_order':
    # Bunny ONLY
    pt_data = [500,808,1310,2120,3432,5555,9000,14560,25000,38160,64000,100000]

    orders = ['o3','o4','o5']

    max_err_pen = np.zeros((len(orders),len(pt_data)))
    RMS_err_pen = np.zeros((len(orders),len(pt_data)))
    styles = ['-','--',':']
    sns.set(style='ticks')
    plt.figure(figsize=(7,6),dpi=180)
    ax = plt.axes()
    for j,order in enumerate(orders):
        for i,num_pts in enumerate(pt_data):
            Func = pickle.load( open( "SAVED_DATA/Opt_"+order+"Bunny28_"+str(num_pts)+".pkl", "rb" ) )
            phi = Func.eval_surface()
            max_err_pen[j,i] = np.max(abs(phi))/Func.Bbox_diag
            RMS_err_pen[j,i] = np.sqrt(np.mean(phi**2))/Func.Bbox_diag
        print('Finished order '+str(order[1])+' dataset')
        ax.loglog(pt_data,RMS_err_pen[j,:],('bs'+styles[j]),linewidth=2,markersize=6,label=('Degree '+str(int(order[1])-1)+' RMS'))
    for j,order in enumerate(orders):
        ax.loglog(pt_data,max_err_pen[j,:],('r.'+styles[j]),linewidth=2,markersize=9,label=('Degree '+str(int(order[1])-1)+' Max'))
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
    rho = 1e-3
    num_samples = 50000
    bunny_exact = extract_stl_info( "geom_shapes/Bunny_exact.stl" )
    exact_dataset = KDTree(bunny_exact[0])
    np.random.seed(1)
    rng = np.random.default_rng()
    indx = rng.choice(np.size(bunny_exact[0],0), size=num_samples, replace=False)

    down_exact_pts = bunny_exact[0][indx,:]
    down_exact_nrm = bunny_exact[1][indx,:]
    pt_data = [500,808,1310,2120,3432,5555,9000,14560,25000,38160,64000,100000]

    RMS_err_Bsplines_fine = np.zeros(len(pt_data))
    RMS_err_KSmethod = np.zeros(len(pt_data))
    ep_error_KSmethod = np.zeros((len(pt_data),len(ep_data)))
    ep_error_Bsplines = np.zeros((len(pt_data),len(ep_data)))
    for i,num_pts in enumerate(pt_data):
        Func = pickle.load( open( "SAVED_DATA/Opt_o4Bunny28_"+str(num_pts)+".pkl", "rb" ) )
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
    ax1.set_ylabel('On-surface RMS error',fontsize=14)
    ax1.legend(fontsize=12,framealpha=1,edgecolor='black',facecolor='white')
    ax1.set_ylim(1e-4,2e-2)
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
    ax2.set_ylabel('Off-surface RMS error',fontsize=14)
    ax2.set_ylim(1e-4,2e-2)
    ax2.legend(fontsize=12,framealpha=1,edgecolor='black',facecolor='white')
    ax2.grid()
    sns.despine()
    plt.tight_layout()
    if save_figures:
        plt.savefig('PDF_figures/EXvBb.pdf',bbox_inches='tight')
    
    try:
        fig8_data = pickle.load(open("fig8_data.pkl","rb"))
    except:
        fig8_data = {}
    fig8_data["our_onsurf"] = RMS_err_Bsplines_fine
    fig8_data["explicit_onsurf"] = RMS_err_KSmethod
    fig8_data["our_offsurf"] = ep_error_Bsplines
    fig8_data["explicit_offsurf"] = ep_error_KSmethod
    fig8_data["Ngamma_range"] = pt_data
    pickle.dump(fig8_data,open("fig8_data.pkl","wb"))
    
if mode == 'Comp_time':
    ep_data = [0.5, 1.0]
    k = 10
    rho = 20
    num_samples = 100000
    bunny_exact = extract_stl_info( "geom_shapes/Bunny_exact.stl" )

    ### Evaluate points on the surface (MAY FAVOR KDTREES) ###
    np.random.seed(1)
    rng = np.random.default_rng()
    indx = rng.choice(np.size(bunny_exact[0],0), size=num_samples, replace=False)
    down_exact_pts = bunny_exact[0][indx,:]

    ### Evaluate points all across the domain ###
    Func = pickle.load( open( "SAVED_DATA/Opt_o4Bunny28_500.pkl", "rb" ) )
    res = int(num_samples**(1/3))
    lower = np.min(Func.exact[0],axis=0)
    upper = np.max(Func.exact[0],axis=0)
    xx, yy, zz = np.meshgrid(
        np.linspace(lower[0], upper[0], res),
        np.linspace(lower[1], upper[1], res),
        np.linspace(lower[2], upper[2], res),
        indexing='ij')
    down_exact_pts = np.vstack((xx.flatten(),yy.flatten(),zz.flatten())).T

    pt_data = [500,808,1310,2120,3432,5555,9000,14560,25000,38160,64000,100000]

    RMS_err_Bsplines_fine = np.zeros(len(pt_data))
    RMS_err_KSmethod = np.zeros(len(pt_data))
    time_KSmethod = np.zeros(len(pt_data))
    time_Bsplines_1000 = np.zeros(len(pt_data))
    time_FredMethod = np.zeros(len(pt_data))
    ep_error_KSmethod = np.zeros((len(pt_data),len(ep_data)))
    ep_error_Bsplines = np.zeros((len(pt_data),len(ep_data)))
    for i,num_pts in enumerate(pt_data):
        Func = pickle.load( open( "SAVED_DATA/Opt_o4Bunny28_"+str(num_pts)+".pkl", "rb" ) )
        t1 = time.perf_counter()
        phi = Func.eval_pts(down_exact_pts)
        t2 = time.perf_counter()
        time_Bsplines_1000[i] = (t2-t1) / len(phi)

        # t1 = time.perf_counter()
        # phi = Lin_et_al_Method(down_exact_pts,Func.surf_pts,Func.normals)
        # t2 = time.perf_counter()
        # time_FredMethod[i] = (t2-t1) / len(phi)

        dataset = KDTree(Func.surf_pts, leafsize=10, compact_nodes=False, balanced_tree=False)
        t1 = time.perf_counter()
        # distances,indices = dataset.query(down_exact_pts,k=k)
        # phi = KS_eval(down_exact_pts,dataset,Func.normals,k,rho)
        # phi = Continuous_Hicken_eval(down_exact_pts,Func.surf_pts,Func.normals,k,rho)
        t2 = time.perf_counter()
        time_KSmethod[i] = (t2-t1) / len(phi)

        print('finished ng={}'.format(num_pts))
    
    time_FredMethod = [5.28685000e-05, 8.26599000e-05, 1.31702000e-04, 2.13472650e-04, 3.46353100e-04, 
        5.79892700e-04, 9.10037800e-04, 1.48618920e-03, 2.57041075e-03, 3.84414815e-03, 6.56379180e-03, 1.04882249e-02]
    time_KSmethod = [1.67347051e-05,2.60991770e-05,4.15005487e-05,6.73784636e-05,1.14200274e-04,1.74571742e-04,
        2.72697531e-04,4.89094513e-04,7.49370782e-04,1.21390604e-03,2.07241920e-03,3.11257037e-03]
    # print(time_KSmethod)
    P_OM = np.polyfit(pt_data,time_Bsplines_1000,1)
    P_EM = np.polyfit(np.log(pt_data),np.log(time_KSmethod),1)
    P_CM = np.polyfit(np.log(pt_data),np.log(time_FredMethod),1)

    bf_OM = np.poly1d(P_OM)
    bf_EM = np.poly1d(P_EM)
    bf_CM = np.poly1d(P_CM)

    sns.set(style='ticks')
    set_fonts()
    fig = plt.figure(figsize=(5,5),dpi=160)
    ax1 = plt.axes()
    ax1.loglog(pt_data,bf_OM(pt_data), 'k-',linewidth=6,alpha=0.13)
    ax1.loglog(pt_data,np.exp(bf_EM(np.log(pt_data))), 'k-',linewidth=6,alpha=0.13)
    ax1.loglog(pt_data,np.exp(bf_CM(np.log(pt_data))), 'k-',linewidth=6,alpha=0.13)

    ax1.loglog(pt_data,time_Bsplines_1000,'.-',label=('Our Method'),color='tab:blue',markersize=10,linewidth=2)
    ax1.loglog(pt_data,time_KSmethod,'.--',label=('Explicit Method'),color='tab:orange',markersize=10,linewidth=2)
    ax1.loglog(pt_data,time_FredMethod,'.:',label=('Lin et al.'),color='tab:green',markersize=10,linewidth=2)

    # Worst Case (n^(1-1/d))
    # ax1.loglog(pt_data, np.power(np.array(pt_data),1-1/3)*time_KSmethod[0]/(pt_data[0]**(1-1/3)),'r-',linewidth=6,alpha=0.50)
    # Average (log(n))
    # ax1.loglog(pt_data, np.log(pt_data)*time_KSmethod[-3]/(np.log(pt_data[-3])),'k-',linewidth=6,alpha=0.13)

    plt.text(1.5e3,5e-4,'$\mathcal{O}(N_{\Gamma})$',        fontsize=14)
    # plt.text(1.5e4,1.6e-5,'$\mathcal{O}(\log(N_{\Gamma}))$',fontsize=14)
    plt.text(5e4,3.8e-6,'$\mathcal{O}(1)$',            fontsize=14)

    ax1.set_xlabel('$N_{\Gamma}$',fontsize=14)
    ax1.set_ylabel('Evaluation Time per point (sec)',fontsize=14)
    ax1.legend(fontsize=14,framealpha=1,edgecolor='black',facecolor='white')
    ax1.grid()
    ax1.set_ylim(1e-6,2e-2)
    sns.despine()
    plt.tight_layout()

    plt.savefig('PDF_figures/Comp_time.pdf',bbox_inches='tight')

    try:
        fig8_data = pickle.load(open("fig8_data.pkl","rb"))
    except:
        fig8_data = {}
    fig8_data["time_ours"] = time_Bsplines_1000
    fig8_data["time_explicit"] = time_KSmethod
    fig8_data["time_Lin_et_al"] = time_FredMethod
    pickle.dump(fig8_data,open("fig8_data.pkl","wb"))


if mode == 'normalized_Hicken_v_Splines':

    ep_data = [0.5, 1]
    k = 20
    rho = 1e-3
    num_samples = 10000
    bunny_exact = extract_stl_info( "geom_shapes/Bunny_exact.stl" )
    exact_dataset = KDTree(bunny_exact[0])
    np.random.seed(1)
    rng = np.random.default_rng()
    indx = rng.choice(np.size(bunny_exact[0],0), size=num_samples, replace=False)
    down_exact_pts = bunny_exact[0][indx,:]
    down_exact_nrm = bunny_exact[1][indx,:]
    pt_data = [500,808,1310,2120,3432,5555,9000,14560,25000,38160,64000,100000]

    RMS_err_Bsplines_fine = np.zeros(len(pt_data))
    RMS_err_KSmethod = np.zeros(len(pt_data))
    ep_error_KSmethod = np.zeros((len(pt_data),len(ep_data)))
    ep_error_Bsplines = np.zeros((len(pt_data),len(ep_data)))
    Hicken_normalizer = np.zeros(len(pt_data))
    for i,num_pts in enumerate(pt_data):
        Func = pickle.load( open( "SAVED_DATA/Opt_o4Bunny28_"+str(num_pts)+".pkl", "rb" ) )
        normalizer = np.linalg.norm(np.diff(Func.dimensions).flatten()/Func.num_cps)
        phi = Func.eval_pts(down_exact_pts)
        RMS_err_Bsplines_fine[i] = np.sqrt(np.mean(phi**2))/Func.Bbox_diag

        dataset = KDTree(Func.surf_pts)
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
    # ax1.set_ylim(8e-6,2e-2)
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
    # ax2.set_ylim(8e-6,2e-2)
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
    surf_pts, normals = extract_stl_info('geom_shapes/Bunny_2002.stl')
    ax = plt.axes(projection='3d')
    ax.plot3D(surf_pts[:,0],surf_pts[:,1],surf_pts[:,2],'k.',markersize=10)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

plt.show()