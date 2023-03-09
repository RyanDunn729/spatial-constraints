import numpy as np
from mesh import trianglemesh

# number of sample points
num_pts = 10
k = 1
path2mesh = ''
# 3D sample points 
sample_pts = np.ones((num_pts,3))
# Initialize the object, which uses the KDtree
mesh = trianglemesh(num_pts,k,path2mesh)

sample_pts =sample_pts.reshape(num_pts,k,3)
# output the closet point and it's normal
query,normals = mesh.nn(sample_pts)
#compute the vector of smaple point and closet point from the anatomy
vec =sample_pts-query

# check robot is inside or not (negative or positive) 
# depend on the normal is inwards or outwards
signed_function = np.einsum("ijk,ijk->ij", vec, normals)
    

    

