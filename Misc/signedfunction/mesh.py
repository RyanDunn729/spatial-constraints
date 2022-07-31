import open3d as o3d
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import scipy.io
from scipy.spatial import KDTree

class trianglemesh:
    
    def __init__(self,num_nodes,k,meshfile):
        # Read the mesh
        mesh = o3d.io.read_triangle_mesh(meshfile)
        self.num_nodes = num_nodes
        self.k = k
        self.mesh = mesh
        mesh.compute_vertex_normals(normalized=True)
        normals = np.asarray(mesh.vertex_normals)
        ana = np.asarray(mesh.vertices).T
        ana_x = ana[0,:]
        ana_y = ana[1,:]
        ana_z = ana[2,:]
        p = ana[:,:].T 
        anatomy = zip(ana_x.ravel(),ana_y.ravel(),ana_z.ravel())
        
        self.normals = normals
        self.normals_nn = normals
        self.p = p
        self.ana_x = ana_x
        self.ana_y = ana_y
        self.ana_z = ana_z
        self.tree = KDTree(list(anatomy))

        
    def nn(self,query):
        
        kk = self.k
        tree = self.tree
        ana_x = self.ana_x
        ana_y = self.ana_y
        ana_z = self.ana_z
        normals = self.normals_nn
        num_nodes = self.num_nodes
        
        
        nearest = np.array(tree.query(query,k=1))
        
        nearest_normals = np.zeros((num_nodes,kk,3))
        nearest_normals = np.reshape(normals[nearest[1,:,:].astype(int).flatten(),:],(num_nodes,kk,3))  
        nearest_pts = np.zeros((num_nodes,kk,3))
        nearest_pts[:,:,0] = np.reshape(ana_x[nearest[1,:,:].astype(int).flatten()],(num_nodes,kk))  
        nearest_pts[:,:,1] = np.reshape(ana_y[nearest[1,:,:].astype(int).flatten()],(num_nodes,kk))
        nearest_pts[:,:,2] = np.reshape(ana_z[nearest[1,:,:].astype(int).flatten()],(num_nodes,kk))
        
        return nearest_pts, -nearest_normals
    
    
