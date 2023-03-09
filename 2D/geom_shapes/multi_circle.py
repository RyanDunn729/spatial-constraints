import numpy as np

class multi_circle(object):
    
    def __init__(self,centers,radii):
        self.center = []
        self.radius = []
        for (c,r) in zip(centers,radii):
            self.add_circle(c,r)

    def add_circle(self,center,r):
        self.center.append(center)
        self.radius.append(r)

    def points(self,num_pts):
        theta = np.linspace(0,2*np.pi,num_pts,endpoint=False)
        pts = np.empty((0,2))
        for (cent,r) in zip(self.center,self.radius):
            px = cent[0]+r*np.cos(theta)
            py = cent[1]+r*np.sin(theta)
            pts = np.vstack((pts,np.stack((px,py),axis=1)))
        return pts

    def unit_pt_normals(self,num_pts):
        theta = np.linspace(0,2*np.pi,num_pts,endpoint=False)
        norms = np.empty((0,2))
        for i in range(len(self.radius)):
            nx = np.cos(theta)
            ny = np.sin(theta)
            norms = np.vstack((norms,np.stack((nx,ny),axis=1)))
        return norms


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns

    centers = [[-13.,-0.5],[-7.,0.5],[2.,0.],[10.,-4.]]
    radii = [2.,2.,4.,3.]

    m = multi_circle(centers,radii)

    surf_pts = m.points(8)
    normals = m.unit_pt_normals(8)

    sns.set()
    plt.plot(surf_pts[:,0],surf_pts[:,1],'b.',markersize=20,label='points')
    for i,(i_pt,i_norm) in enumerate(zip(surf_pts,normals)):
        if i == 0:
            plt.arrow(i_pt[0],i_pt[1],i_norm[0],i_norm[1],color='k',label='normals')
        else:
            plt.arrow(i_pt[0],i_pt[1],i_norm[0],i_norm[1],color='k')

    exact = m.points(1000)
    plt.plot(exact[:,0],exact[:,1],'k.',markersize=1,label='exact')

    plt.legend(loc='upper right')
    plt.title('Mutli-Circles')
    plt.axis('equal')
    plt.show()