import numpy as np
from matplotlib.patches import FancyArrowPatch

class Ellipsoid(object):

    def __init__(self,a,b,c):
        self.a = a
        self.b = b
        self.c = c

    def points(self,num_pts):
        indices = np.arange(0, num_pts, dtype=float) + 0.5

        phi = np.arccos(1 - 2*indices/num_pts)
        theta = np.pi * (1 + 5**0.5) * indices

        x = self.a*np.cos(theta)*np.sin(phi)
        y = self.b*np.sin(theta)*np.sin(phi)
        z = self.c*np.cos(phi)
        return np.stack((x.flatten(),y.flatten(),z.flatten()),axis=1)

    def old_points(self,num_pts):
        num_pts = int(np.sqrt(num_pts))
        u = np.linspace(0,2*np.pi,num_pts+1)[0:num_pts]
        v = np.linspace(0,np.pi,num_pts+1)[0:num_pts]
        x = self.a*np.outer(np.sin(u), np.cos(v))
        y = self.b*np.outer(np.sin(u), np.sin(v))
        z = self.c*np.outer(np.cos(u), np.ones(num_pts))
        return np.stack((x.flatten(),y.flatten(),z.flatten()),axis=1)

    def unit_pt_normals(self,num_pts):
        pts = self.points(num_pts)
        dx = 2*pts[:,0]/self.a
        dy = 2*pts[:,1]/self.b
        dz = 2*pts[:,2]/self.c
        vec = np.stack((dx,dy,dz),axis=1)
        norm = np.linalg.norm(vec,axis=1)
        return vec / np.tile(norm,(3,1)).T

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns

    a = 16
    b = 6
    c = 8
    num_pts = 100

    e = Ellipsoid(a,b,c)

    pts = e.points(num_pts)
    normals = e.unit_pt_normals(num_pts)

    num_pts = len(pts)

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
    ax.set_xlim(-a,a)
    ax.set_ylim(-a,a)
    ax.set_zlim(-a,a)
    ax.legend()
    plt.show()

