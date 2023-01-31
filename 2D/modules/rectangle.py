import numpy as np

class rectangle(object):
    
    def __init__(self,w,h):
        self.h = h
        self.w = w
        self.range = 2*w + 2*h
        self.b1 = w
        self.b2 = w+h
        self.b3 = 2*w+h

    def points(self,num_pts):
        theta = np.linspace(0,self.range,2*(num_pts)+1)[1::2]
        pts = self.get_pts(theta)
        return pts

    def closed_pts(self,num_pts):
        theta = np.linspace(0,self.range,num_pts)
        pts = self.get_pts(theta)
        return pts

    def unit_pt_normals(self,num_pts):
        theta = np.linspace(0,self.range,2*(num_pts)+1)[1::2]
        norm_vec = np.zeros((len(theta),2))
        for i,t in enumerate(theta):
            if t==0:
                norm_vec[i] = np.array([-1,-1])/np.sqrt(2)
            elif t>0 and t<self.b1:
                norm_vec[i] = np.array([0,-1])
            elif t==self.b1:
                norm_vec[i] = np.array([1,-1])/np.sqrt(2)
            elif t>self.b1 and t<self.b2:
                norm_vec[i] = np.array([1,0])
            elif t==self.b2:
                norm_vec[i] = np.array([1,1])/np.sqrt(2)
            elif t>self.b2 and t<self.b3:
                norm_vec[i] = np.array([0,1])
            elif t==self.b3:
                norm_vec[i] = np.array([-1,1])/np.sqrt(2)
            elif t>self.b3 and t<self.range:
                norm_vec[i] = np.array([-1,0])
        return norm_vec

    def get_pts(self,theta):
        pts = np.zeros((len(theta),2))
        for i,t in enumerate(theta):
            if t<self.b1:
                pts[i,0] = t - self.b1/2
                pts[i,1] = -self.h/2
            elif t>=self.b1 and t<self.b2:
                pts[i,0] = self.w/2
                pts[i,1] = (t-self.b1-self.h/2)
            elif t>=self.b2 and t<self.b3:
                pts[i,0] = (self.b3-t-self.w/2)
                pts[i,1] = self.h/2
            elif t<=self.range:
                pts[i,0] = -self.w/2
                pts[i,1] = (self.range-t-self.h/2)
        return pts

    def get_princ_curvatures(self,num_pts):
        curv = np.zeros(num_pts)

        # theta = np.linspace(0,self.range,2*(num_pts)+1)[1::2]
        # pts = self.get_pts(theta)
        # for i,t in enumerate(theta):
        #     if i==0:
        #         curv[i] = -1/(pts[i,0]-self.w/2)
        #     elif t<self.b1 and theta[i+1]>self.b1:
        #         curv[i] = -1/(self.w/2-pts[i,0])
        #     elif theta[i-1]<self.b1 and t>self.b1:
        #         curv[i] = -1/(pts[i,1]+self.h/2)
        #     elif t<self.b2 and theta[i+1]>self.b2:
        #         curv[i] = -1/(self.h/2-pts[i,1])
        #     elif theta[i-1]<self.b2 and t>self.b2:
        #         curv[i] = -1/(self.w/2-pts[i,0])
        #     elif t<self.b3 and theta[i+1]>self.b3:
        #         curv[i] = -1/(pts[i,0]+self.w/2)
        #     elif theta[i-1]<self.b3 and t>self.b3:
        #         curv[i] = -1/(self.h/2-pts[i,1])
        #     elif t == theta[-1]:
        #         curv[i] = -1/(self.h/2-pts[i,1])

        return curv

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set()

    w = 5
    h = 7
    num_pts = 78

    r = rectangle(w,h)
    
    pts = r.points(num_pts)
    plt.plot(pts[:,0],pts[:,1],'k.',label='points')

    pts = r.points(num_pts)
    normals = r.unit_pt_normals(num_pts)
    for i in range(num_pts):
        if i == 0:
            plt.arrow(pts[i,0],pts[i,1],normals[i,0],normals[i,1],color='k',label='normals')
        else:
            plt.arrow(pts[i,0],pts[i,1],normals[i,0],normals[i,1],color='k')

    exact = r.points(1000)
    plt.plot(exact[:,0],exact[:,1],'b-',label='exact')

    plt.legend(loc='upper right')
    plt.title('Rectangle w={} h={}'.format(w,h))
    plt.axis('equal')

    plt.show()