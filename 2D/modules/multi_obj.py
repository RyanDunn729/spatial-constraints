import numpy as np

class multi_obj(object):
    
    def __init__(self,centers,radii,rect_dim):
        self.circle_centers = []
        self.rect_centers = []
        self.radius = []
        self.ab = []
        for i,c in enumerate(centers):
            if i<len(radii):
                self.add_circle(c,radii[i])
            else:
                ii = len(radii)-i
                self.add_rect(c,rect_dim[ii])

    def add_rect(self,center,dim):
        self.rect_centers.append(center)
        self.ab.append(dim)

    def add_circle(self,center,r):
        self.circle_centers.append(center)
        self.radius.append(r)

    def points(self,num_pts):
        theta = np.linspace(0,2*np.pi,num_pts,endpoint=False)
        pts = np.empty((0,2))
        for (cent,r) in zip(self.circle_centers,self.radius):
            px = cent[0]+r*np.cos(theta)
            py = cent[1]+r*np.sin(theta)
            pts = np.vstack((pts,np.stack((px,py),axis=1)))

        for (cent,ab) in zip(self.rect_centers,self.ab):
            w = ab[0]
            h = ab[1]
            rng = 2*w + 2*h
            b1 = w
            b2 = w+h
            b3 = 2*w+h
            theta = np.linspace(0,rng,2*(num_pts)+1)[1::2]
            i_pts = np.zeros((len(theta),2))
            for i,t in enumerate(theta):
                if t<b1:
                    i_pts[i,0] = t - b1/2
                    i_pts[i,1] = -h/2
                elif t>=b1 and t<b2:
                    i_pts[i,0] = w/2
                    i_pts[i,1] = (t-b1-h/2)
                elif t>=b2 and t<b3:
                    i_pts[i,0] = (b3-t-w/2)
                    i_pts[i,1] = h/2
                elif t<=rng:
                    i_pts[i,0] = -w/2
                    i_pts[i,1] = (rng-t-h/2)
            i_pts[:,0] += cent[0]
            i_pts[:,1] += cent[1]
            pts = np.vstack((pts,i_pts))
        return pts

    def unit_pt_normals(self,num_pts):
        theta = np.linspace(0,2*np.pi,num_pts,endpoint=False)
        norms = np.empty((0,2))
        for i in range(len(self.radius)):
            nx = np.cos(theta)
            ny = np.sin(theta)
            norms = np.vstack((norms,np.stack((nx,ny),axis=1)))


        for (cent,ab) in zip(self.rect_centers,self.ab):
            w = ab[0]
            h = ab[1]
            rng = 2*w + 2*h
            b1 = w
            b2 = w+h
            b3 = 2*w+h
            theta = np.linspace(0,rng,2*(num_pts)+1)[1::2]
            i_norm_vec = np.zeros((len(theta),2))
            for i,t in enumerate(theta):
                if t==0:
                    i_norm_vec[i] = np.array([-1,-1])/np.sqrt(2)
                elif t>0 and t<b1:
                    i_norm_vec[i] = np.array([0,-1])
                elif t==b1:
                    i_norm_vec[i] = np.array([1,-1])/np.sqrt(2)
                elif t>b1 and t<b2:
                    i_norm_vec[i] = np.array([1,0])
                elif t==b2:
                    i_norm_vec[i] = np.array([1,1])/np.sqrt(2)
                elif t>b2 and t<b3:
                    i_norm_vec[i] = np.array([0,1])
                elif t==b3:
                    i_norm_vec[i] = np.array([-1,1])/np.sqrt(2)
                elif t>b3 and t<rng:
                    i_norm_vec[i] = np.array([-1,0])
            norms = np.vstack((norms,i_norm_vec))
        return norms

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns

    centers = [[-13.,-0.5],[5.,-3.],[2.,6.],[-5.,-2]]
    radii = [2.5,4.]
    ab = [[12,4],[3,6]]

    m = multi_obj(centers,radii,ab)

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
    plt.title('Mutli-Objects')
    plt.axis('equal')
    plt.show()