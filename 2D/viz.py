import numpy as np
import seaborn as sns

import pickle
from scipy.spatial import KDTree

from lsdo_viz.api import BaseViz, Frame

sns.set()

class Viz(BaseViz):
    def setup(self):
        # self.use_latex_fonts()
        self.frame_name_format = 'output_{}'

        self.add_frame(
            Frame(
                height_in=9.,
                width_in=15,
                nrows=3,
                ncols=6,
                wspace=0.4,
                hspace=0.4,
                keys3d=[((0,2,None),(4,None,None))]
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
        with self.get_frame(1)[0:, 2:4] as ax:
            x = Func.dimensions[0]
            y = Func.dimensions[1]
            res = 200
            uu,vv = np.meshgrid(np.linspace(0,1,res),
                                np.linspace(0,1,res))
            b = Func.Surface.get_basis_matrix(uu.flatten(),vv.flatten(),0,0)
            xx = b.dot(Func.cps[:,0]).reshape(res,res)
            yy = b.dot(Func.cps[:,1]).reshape(res,res)
            phi = b.dot(data_dict_current['phi_cps']).reshape(res,res)
            ax.contour(xx,yy,phi,levels=[-2,-1,0,1,2],colors=['red','orange','green','blue','purple'])
            # exact = Func.e.closed_pts(res)
            # ax.plot(exact[:,0],exact[:,1],'k-',label='exact')
            ax.plot(Func.surf_pts[:,0],Func.surf_pts[:,1],'k.',label='surface points')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Contour Plot for $n_{{\Gamma}}$ = {}'.format(len(Func.u['surf'])))
            ax.legend(loc='upper right')
            ax.set_xticks([x[0],np.sum(x)/2,x[1]])
            ax.set_yticks([y[0],np.sum(y)/2,y[1]])
            # ax.set_xlim(x[0], x[1])
            # ax.set_ylim(y[0], y[1])
            ax.axis('equal')

        with self.get_frame(1)[0:2, 4:] as ax:
            ax.plot(Func.cps[:,0],Func.cps[:,1],data_dict_current['phi_cps'],'k.')
            res = 200
            uu,vv = np.meshgrid(np.linspace(0,1,res),
                                np.linspace(0,1,res))
            b = Func.Surface.get_basis_matrix(uu.flatten(),vv.flatten(),0,0)
            xx = b.dot(Func.cps[:,0]).reshape(res,res)
            yy = b.dot(Func.cps[:,1]).reshape(res,res)
            phi = b.dot(data_dict_current['phi_cps']).reshape(res,res)
            ax.contour(xx, yy, phi, levels=0)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('$\Phi$')
            ax.set_title('Control Points for iter = {}'.format(ind))
            ax.set_xticks([Func.dimensions[0,0],np.sum(Func.dimensions[0,:])/2,Func.dimensions[0,1]])
            ax.set_yticks([Func.dimensions[1,0],np.sum(Func.dimensions[1,:])/2,Func.dimensions[1,1]])
            center = np.mean(Func.dimensions,axis=1)
            d = np.max(np.diff(Func.dimensions,axis=1))
            ax.set_xlim(center[0]-d/2, center[0]+d/2)
            ax.set_ylim(center[1]-d/2, center[1]+d/2)
            ax.set_zlim(limits_dict['initial']['phi_cps']['min'][0],
                        limits_dict['initial']['phi_cps']['max'][0])

        with self.get_frame(1)[2, 4:] as ax:
            res = 50
            bbox_perc = 5
            ep_max = bbox_perc*Func.Bbox_diag / 100
            ep_range = np.linspace(-ep_max,ep_max,res)
            dataset = KDTree(Func.exact[0])
            RMS_local = np.zeros(len(ep_range))
            sample_pts = Func.exact[0][::10]
            sample_normals = Func.exact[1][::10]
            for i,ep in enumerate(ep_range):
                i_pts = sample_pts + ep*sample_normals
                u,v = Func.spatial_to_parametric(i_pts)
                b = Func.Surface.get_basis_matrix(u,v,0,0)
                phi = b.dot(data_dict_current['phi_cps'])
                phi_ex,_ = dataset.query(i_pts,k=1)
                RMS_local[i] = np.sqrt(np.sum(  (abs(phi)-phi_ex)**2  )/len(phi_ex))
            ax.plot(np.linspace(-bbox_perc,bbox_perc,res),RMS_local/Func.Bbox_diag)
            ax.set_title('Local Distance Error')
            ax.set_xlabel('$\epsilon$ as a $\%$ of BBox diag')
            ax.set_ylabel('Normalized RMS')
            ax.set_xlim(-bbox_perc,bbox_perc)
            # _, max_error = Func.check_local_RMS_error(bbox_perc,res=2)
            # ax.set_ylim(max_error[0]/1.25,1.25*max_error[1])

        with self.get_frame(1)[2, 0:2] as ax:
            xlim = (-1,15)
            norms = np.linalg.norm(data_dict_current['hessians'],axis=(1,2))
            sns.boxplot(ax=ax,x=norms)
            ax.set_xlim(xlim)
            ax.set_xlabel('Forbenius Norm of Hesssian Samples')

        with self.get_frame(1)[0:2, 0:2] as ax:
            res = 150
            ones = np.ones(res)
            diag = np.linspace(0,1,res)
            basis = Func.Surface.get_basis_matrix(diag, 0.5*ones, 0, 0)
            phi = basis.dot(data_dict_current['phi_cps'])
            ax.plot(diag, phi, '-', label='X-axis')
            basis = Func.Surface.get_basis_matrix(0.5*ones, diag, 0, 0)
            phi = basis.dot(data_dict_current['phi_cps'])
            ax.plot(diag, phi, '-', label='Y-axis')
            
            ax.axis([0,1,-8,8])
            ax.set_xticks([0,0.5,1])
            ax.set_yticks([-5,0,5])
            ax.set_xlabel('Normalized Location')
            ax.set_ylabel('Phi')
            ax.set_title('Phi along 1D slices')
            ax.legend()
            
        self.get_frame(1).write()
