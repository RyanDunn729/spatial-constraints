import numpy as np
from stl.mesh import Mesh
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

def plot3d(pts,xlim=None,ylim=None,zlim=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    img = ax.scatter(pts[:,0],pts[:,1],pts[:,2])
    if xlim and ylim and zlim:
        ax.set(xlim=xlim,ylim=ylim,zlim=zlim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()
    exit()

def convert_pts(pts):
    new_pts = np.empty((len(pts),3)) 
    for i in range(len(pts)):
        for j in range(3):
            new_pts[i,j] = pts[i,j]/3 + pts[i,j+3]/3 + pts[i,j+6]/3
    return new_pts

def std_uniform_knot_vec(num_cps,order):
    knot_vector = np.zeros(num_cps + order)
    den = num_cps - order + 1
    
    for i in range(num_cps + order):
        knot_vector[i] = (i - order + 1) / den
    
    return knot_vector

def remove_duplicates(pts):
    duplicates = np.empty((0,3))
    for xyz in pts:
        is_duplicate = 0
        for xyz_dup in duplicates:
            if all(xyz == xyz_dup):
                is_duplicate = 1
        if is_duplicate == 0:
            duplicates = np.vstack((duplicates,xyz))
    return duplicates

def import_mesh(filename):
    mesh_import = Mesh.from_file(filename)

    temp = mesh_import.points.reshape(3*len(mesh_import.points),3)
    mesh_pts = np.unique(temp,axis=1)

    temp = mesh_import.points.reshape(len(mesh_import.points),3,3)
    mesh_midpts = np.mean(temp,axis=1)
    
    return mesh_import, mesh_pts, mesh_midpts, mesh_import.get_unit_normals()

def sdf_sample_points(filename,num_samp_const,k=1):
    pts_x = num_samp_const[0]
    pts_y = num_samp_const[1]
    pts_z = num_samp_const[2]

    num_samples = pts_x * pts_y * pts_z
    mesh_import = Mesh.from_file(filename)

    minx,miny,minz = mesh_import.min_
    maxx,maxy,maxz = mesh_import.max_

    x = np.mean(mesh_import.x,axis=1)
    y = np.mean(mesh_import.y,axis=1)
    z = np.mean(mesh_import.z,axis=1)
    mesh_midpts = np.column_stack((x,y,z))
    norm_vec = mesh_import.get_unit_normals() # (num_pts,3)

    pt_grid = np.empty((num_samples, 3))
    pt_grid[:, 0] = np.einsum('i,j,k->ijk', np.linspace(minx,maxx,pts_x+2)[1:pts_x+1], np.ones(pts_y), np.ones(pts_z)).flatten()
    pt_grid[:, 1] = np.einsum('i,j,k->ijk', np.ones(pts_x), np.linspace(miny,maxy,pts_y+2)[1:pts_y+1], np.ones(pts_z)).flatten()
    pt_grid[:, 2] = np.einsum('i,j,k->ijk', np.ones(pts_x), np.ones(pts_y), np.linspace(minz,maxz,pts_z+2)[1:pts_z+1]).flatten()

    dataset = KDTree(mesh_midpts)
    _,indeces = dataset.query(pt_grid,k=k)

    inside_pts = np.empty((0,3))
    outside_pts = np.empty((0,3))
    unused_pts = np.empty((0,3))
    for i,ind in enumerate(indeces):
        check = np.empty((k,))
        i_pt = pt_grid[i,:]
        for j in range(k):
            if k==1:
                index = ind
            else:
                index = ind[j]
            i_meshpt = mesh_midpts[index]
            dx = (i_meshpt-i_pt)
            if np.dot(dx,norm_vec[index])>0:
                check[j] = True
            else:
                check[j] = False

        if all(check):
            inside_pts = np.vstack((inside_pts,i_pt))
        elif not any(check):
            outside_pts = np.vstack((outside_pts,i_pt))
        else:
            unused_pts = np.vstack((unused_pts,i_pt))

    return inside_pts, outside_pts, np.float64(mesh_midpts), norm_vec

def init_cps_constant(num_cps,order,rangex,rangey,rangez,val=0,rand=1e-2):
    # Order 4, basis = [1/6, 4/6, 1/6]
    if order == 4:
        A = np.array([[num_cps[0]-2, 1],
                    [1, num_cps[0]-2]])
        b = np.array([rangex[0]*(num_cps[0]-1), rangex[1]*(num_cps[0]-1)])
        xn = np.linalg.solve(A,b)

        A = np.array([[num_cps[1]-2, 1],
                    [1, num_cps[1]-2]])
        b = np.array([rangey[0]*(num_cps[1]-1), rangey[1]*(num_cps[1]-1)])
        yn = np.linalg.solve(A,b)

        A = np.array([[num_cps[2]-2, 1],
                    [1, num_cps[2]-2]])
        b = np.array([rangez[0]*(num_cps[2]-1), rangez[1]*(num_cps[2]-1)])
        zn = np.linalg.solve(A,b)

    cps = np.empty((num_cps[0],num_cps[1],num_cps[2],4))
    cps[:, :, :, 0] = np.einsum('i,j,k->ijk', np.linspace(xn[0], xn[1], num_cps[0]), np.ones(num_cps[1]),np.ones(num_cps[2]))
    cps[:, :, :, 1] = np.einsum('i,j,k->ijk', np.ones(num_cps[0]), np.linspace(yn[0], yn[1], num_cps[1]),np.ones(num_cps[2]))
    cps[:, :, :, 2] = np.einsum('i,j,k->ijk', np.ones(num_cps[0]), np.ones(num_cps[1]),np.linspace(zn[0], zn[1], num_cps[2]))
    shape = [cps.shape[0],cps.shape[1],cps.shape[2]]
    cps[:,:,:,3] = val*np.ones(shape) + rand*np.random.rand(shape[0],shape[1],shape[2])
    return cps.reshape((num_cps[0]*num_cps[1]*num_cps[2], 4))