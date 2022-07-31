import numpy as np
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import marching_cubes
import seaborn as sns
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from lsdo_viz.api import BaseViz, Frame
from stl.mesh import Mesh
from scipy.spatial import KDTree

sns.set()

class Viz(BaseViz):
    def setup(self):
        # self.use_latex_fonts()
        self.frame_name_format = 'output_{}'

        self.add_frame(
            Frame(
                height_in=10.,
                width_in=15,
                nrows=4,
                ncols=3,
                wspace=0.4,
                hspace=0.4,
                keys3d=[((0,3,None),(1,None,None))]
            ), 1)

    def plot(self,
             data_dict_current,
             data_dict_all,
             limits_dict,
             ind,
             video=False):
        
        Func = pickle.load( open( "_Saved_Function.pkl", "rb" ) )
        gold = (198/255, 146/255, 20/255)
        blue = (24/255, 43/255, 73/255)

        self.get_frame(1).clear_all_axes()
        with self.get_frame(1)[0:3, 1:] as ax:
            x = Func.dimensions[0]
            y = Func.dimensions[1]
            z = Func.dimensions[2]
            res = 50
            u = np.einsum('i,j,k->ijk', np.linspace(0,1,res), np.ones(res),np.ones(res)).flatten()
            v = np.einsum('i,j,k->ijk', np.ones(res), np.linspace(0,1,res),np.ones(res)).flatten()
            w = np.einsum('i,j,k->ijk', np.ones(res), np.ones(res),np.linspace(0,1,res)).flatten()
            basis = Func.Volume.get_basis_matrix(u, v, w, 0, 0, 0)
            phi = basis.dot(data_dict_current['phi_cps']).reshape((res,res,res))
            verts, faces,_,_ = marching_cubes(phi, 0)
            verts = verts*np.diff(Func.dimensions).flatten()/(res-1) + Func.dimensions[:,0]
            level_set = Poly3DCollection(verts[faces],linewidth=0.25,alpha=1,facecolor=gold,edgecolor='k')
            ax.add_collection3d(level_set)
            # ax.plot(Func.surf_pts[:,0],Func.surf_pts[:,1],Func.surf_pts[:,2],
                    # 'k.',label='surface points')
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.set_title('Level Set (iter = %i)'%ind)
            ax.set_xticks([x[0],(x[1]+x[0])/2,x[1]])
            ax.set_yticks([y[0],(y[1]+y[0])/2,y[1]])
            ax.set_zticks([z[0],(z[1]+z[0])/2,z[1]])
            center = np.mean(Func.dimensions,axis=1)
            d = np.max(np.diff(Func.dimensions,axis=1))
            ax.set_xlim(center[0]-d/2, center[0]+d/2)
            ax.set_ylim(center[1]-d/2, center[1]+d/2)
            ax.set_zlim(center[2]-d/2, center[2]+d/2)

        with self.get_frame(1)[2:4, 0] as ax:
            norms = np.linalg.norm(data_dict_current['hessians'],axis=(1,2))
            sns.boxplot(ax=ax,x=norms)
            # ax.set_xlim(left=np.min(abs(limits_dict['initial']['hessians']['min'][0].ravel()))) #, right=np.max(limits_dict['initial']['hessians']['max'][0].ravel()))
            ax.set_xlabel('Forbenius Norm of Hesssian Samples')
        
        with self.get_frame(1)[0:2, 0] as ax:
            res = 200
            ones = np.ones(res)
            diag = np.linspace(0,1,res)

            basis = Func.Volume.get_basis_matrix(diag, 0.5*ones, 0.5*ones, 0, 0, 0)
            pts = basis.dot(data_dict_current['phi_cps'])
            ax.plot(diag, pts, '-', label='X-axis')
            basis = Func.Volume.get_basis_matrix(0.5*ones, diag, 0.5*ones, 0, 0, 0)
            pts = basis.dot(data_dict_current['phi_cps'])
            ax.plot(diag, pts, '-', label='Y-axis')
            basis = Func.Volume.get_basis_matrix(0.5*ones, 0.5*ones, diag, 0, 0, 0)
            pts = basis.dot(data_dict_current['phi_cps'])
            ax.plot(diag, pts, '-', label='Z-axis')

            ax.set_ylim(-1,1)
            ax.set_xlim(0,1)
            ax.set_xticks([0,0.5,1])
            ax.set_yticks([-1,0,1])
            ax.set_xlabel('Normalized Location')
            ax.set_ylabel('Phi')
            ax.set_title('Phi along 1D slices')
            ax.legend()

        with self.get_frame(1)[3, 1:] as ax:
            res = 10
            bbox_perc = 5

            ep_max = bbox_perc*Func.Bbox_diag / 100
            ep_range = np.linspace(-ep_max,ep_max,res)
            dataset = KDTree(Func.exact[0])
            RMS_local = np.zeros(len(ep_range))
            sample_pts = Func.exact[0][::10]
            sample_normals = Func.exact[1][::10]
            for i,ep in enumerate(ep_range):
                i_pts = sample_pts + ep*sample_normals
                u,v,w = Func.spatial_to_parametric(i_pts)
                b = Func.Volume.get_basis_matrix(u,v,w,0,0,0)
                phi = b.dot(data_dict_current['phi_cps'])*Func.Bbox_diag
                phi_exact,_ = dataset.query(i_pts,k=1)
                RMS_local[i] = np.sqrt(np.sum(  (abs(phi)-phi_exact)**2  )/len(phi_exact) )
            x_perc = np.linspace(-bbox_perc,bbox_perc,res)
            norm_error = RMS_local/Func.Bbox_diag
            ax.plot(x_perc,norm_error,'b.-')
            ax.set_xlabel("$\epsilon$ ($\%$ of BBox diag)")
            ax.set_ylabel("Normalized RMS")
            ax.set_title('Local distance error')
            ax.set_xlim(-bbox_perc,bbox_perc)
            # _,max_error = Func.check_local_RMS_error(bbox_perc,2)
            # ax.set_ylim(max_error[0]/1.25,1.25*max_error[1])

        self.get_frame(1).write()
