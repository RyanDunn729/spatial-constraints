import numpy as np

class ellipse(object):
    
    def __init__(self,a,b):
        self.a = a
        self.b = b


    def points(self,num_pts):
        theta = np.linspace(0,2*np.pi,num_pts+1)[0:num_pts]
        pts = self.get_pts(theta)
        return pts

    def closed_pts(self,num_pts):
        theta = np.linspace(0,2*np.pi,num_pts+1)
        pts = self.get_pts(theta)
        return pts

    def midpts(self,num_pts):
        theta = np.linspace(0,2*np.pi,num_pts+1)[0:num_pts]
        pts = self.get_pts(theta)
        midpts = np.empty((num_pts,2))
        for i in range(num_pts-1):
            midpts[i] = np.mean(pts[i:i+2],axis=0)
        midpts[num_pts-1] = np.mean(np.vstack((pts[0],pts[num_pts-1])),axis=0)
        return midpts

    def shifted_pts(self,num_pts):
        shift = 2*np.pi / (2*num_pts)
        theta_shift = np.linspace(shift,2*np.pi+shift,num_pts+1)[0:num_pts]
        return self.get_pts(theta_shift)

    def unit_midpt_normals(self,num_pts):
        pts = self.closed_pts(num_pts)
        d = np.empty((num_pts,2))
        for i in range(num_pts):
            d[i] = (pts[i] - pts[i+1]) / np.linalg.norm(pts[i] - pts[i+1])
        d[:,1] *= -1
        ## Check
        # check = np.einsum('ij,ij->i',(pts[0:num_pts]-pts[1:num_pts+1]),np.fliplr(d)[0:num_pts])
        # print(np.linalg.norm(check))
        # exit()
        return np.fliplr(d)

    def unit_pt_normals(self,num_pts):
        theta = np.linspace(0,2*np.pi,num_pts+1)[0:num_pts]
        pts = self.get_pts(theta)
        nx = pts[:,0]*self.b/self.a
        ny = pts[:,1]*self.a/self.b
        vec = np.stack((nx,ny),axis=1)
        norm = np.linalg.norm(vec,axis=1)
        return vec/np.stack((norm,norm),axis=1)

    def get_princ_curvatures(self,num_pts):
        theta = np.linspace(0,2*np.pi,num_pts+1)[0:num_pts]
        pts = np.stack((self.a*np.sin(theta),self.b*np.cos(theta)),axis=1)
        curvatures = -self.a*self.b / np.sqrt(pts[:,0]**2 + pts[:,1]**2)**3
        return curvatures

    def get_pts(self,theta):
        return np.stack((self.a*np.cos(theta),self.b*np.sin(theta)),axis=1)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set()

    a = 18
    b = 5
    num_pts = 25

    e = ellipse(a,b)
    
    pts = e.closed_pts(num_pts)
    plt.plot(pts[:,0],pts[:,1],'b-',label='points')

    pts = e.midpts(num_pts)
    # plt.plot(pts[:,0],pts[:,1],'r-',label='midpoints')

    pts = e.points(num_pts)
    normals = e.unit_pt_normals(num_pts)
    for i in range(num_pts):
        if i == 0:
            plt.arrow(pts[i,0],pts[i,1],normals[i,0],normals[i,1],color='k',label='normals')
        else:
            plt.arrow(pts[i,0],pts[i,1],normals[i,0],normals[i,1],color='k')

    exact = e.points(1000)
    plt.plot(exact[:,0],exact[:,1],'k-',label='exact')

    plt.legend(loc='upper right')
    plt.title('Ellipse a={} b={}'.format(a,b))
    plt.axis('equal')

    curv = e.get_princ_curvatures(num_pts)
    print(curv)

    plt.show()