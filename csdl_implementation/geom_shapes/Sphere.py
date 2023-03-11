import numpy as np
from numpy import newaxis as na
from matplotlib.patches import FancyArrowPatch

class Sphere(object):

    def __init__(self,radius):
        self.radius = radius

    def points(self,num_pts):
        points = np.zeros((num_pts,3))
        phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

        for i in range(num_pts):
            y = 1 - (i / float(num_pts - 1)) * 2  # y goes from 1 to -1
            r = np.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = np.cos(theta) * r
            z = np.sin(theta) * r

            points[i] = self.radius*np.array([x, y, z])

        return points
    
    def unit_pt_normals(self,num_pts):
        pts = self.points(num_pts)
        normals = pts / np.linalg.norm(pts,axis=1)[:,na]
        return normals

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns

    radius = 2
    num_pts = 100

    e = Sphere(radius)

    pts = e.points(num_pts)
    normals = e.unit_pt_normals(num_pts)
    print(pts)
    print(normals)

    sns.set()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i,(i_pt,i_norm) in enumerate(zip(pts,normals)):
        i_norm *= 1
        if i == 0:
            pt1 = i_pt + i_norm
            pt = np.vstack((i_pt,pt1))
            ax.plot(pt[:,0],pt[:,1],pt[:,2],'k.-',label='Normals')
        else:
            pt1 = i_pt + i_norm
            pt = np.vstack((i_pt,pt1))
            ax.plot(pt[:,0],pt[:,1],pt[:,2],'k.-')
    ax.plot(pts[:,0],pts[:,1],pts[:,2],'b.',label='Surface Points')
    ax.set_title('$N_{\Gamma}$ = %i' %num_pts)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(-radius,radius)
    ax.set_ylim(-radius,radius)
    ax.set_zlim(-radius,radius)
    ax.legend()
    plt.show()

