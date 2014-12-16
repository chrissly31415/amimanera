"""
Created on Thu Dec 29 16:41:13 2011

@author: Christoph Loschen
"""

from numpy import *

#print "Starting Downhill-Simplex Optimization"

#test function
def function(p):
    y=pow(p[0]-2,2)+pow(p[1]-2,2)+pow(p[2],2)+pow(p[3]-5.5,2)+pow(p[4]-3,2)
    y=y+5
    #r_solute[0]=p[0]
    #r_solute[1]=p[1]
    #r_solute[2]=p[2]
    #r_solute[3]=p[3]
    #A=p[4] 
    return y

def function2(p):
    y = -np.sum(np.sin(p))
    return y

#simplex optimization in 5 Dimensions
# p0: vector with 5 dimensions
# fn: function which takes 5dim vector as argument 
def simplex(p0, fn, fixA=True):
    #setup
    ndim=5
    delta=1
    nmax=5000
    ilow=0
    ihigh=0
    inhigh=0
    ftol=1e-10
      
    
    #unit vectors
    e=np.identity(ndim)
    
    global A_fix
    A_fix=0.0
    if fixA==True:        
        A_fix=p0[4]
        e5=array([0.0, 0.0, 0.0, 0.0, 0.0])
        
    #creating N other points, N+1 in total
    p0=p0
    p1=p0+delta*e[0]    
    p2=p0+delta*e[1]
    p3=p0+delta*e[2]
    p4=p0+delta*e[3]
    p5=p0+delta*e[4]
    
    p = vstack((p0,p1,p2,p3,p4,p5))
    y0=fn(p0)
    y1=fn(p1)
    y2=fn(p2)
    y3=fn(p3)
    y4=fn(p4)
    y5=fn(p5)
    y = vstack((y0,y1,y2,y3,y4,y5))

    print "Initial guess: ",
    print p0
    
    #start iteration
    for k in range(nmax):
        print "##iteration:%2d  r:%8.2f%8.2f%8.2f%8.2f y:%8.2f ##" %(k,p[ilow][0],p[ilow][1],p[ilow][2],p[ilow][3],y[ilow]),   
        #determine highest and lowest point index
        for i in range(size(y)):
            if y.min()==y[i]:#dangerous
                ilow=i
            if y.max()==y[i]:
                ihigh=i
            #2nd highest
        for i in range(size(y)):    
            temp=y.copy()
            temp[ihigh]=y[ilow]
            if temp.max()==y[i]:
                inhigh=i
        #double tolerance
        if ftol>abs(y[ilow]-y[ihigh]):                   
            print "ftol: %8.6e" %(ftol)
            print "Simplex optimization CONVERGED!"
            break
        else:
            print "tol: %8.6e" %(abs(y[ilow]-y[ihigh]))
        psum=get_psum(p)
        ytry=amoebamove(p,p0,y,ihigh,-1.0, ndim, psum,fn)
        if ytry<y[ilow]:
            print "New minimum, trying EXTRAPOLATION"
            ytry=amoebamove(p,p0,y,ihigh,2.0, ndim, psum, fn)
        elif ytry>y[inhigh]:
            print "No minimum, trying CONTRACTION"
            ysave=ytry
            ytry=amoebamove(p,p0,y,ihigh,0.5, ndim, psum, fn)
    return p[ilow]     

def amoebamove(p,p0,y,ihigh,factor,ndim,psum,fn):
    #print "Amoeba moves...."
    ptry=zeros(size(p[0]))
    ytry=0.0
    factor1=(1.0-factor)/ndim

    factor2=factor1-factor

    for j in range(size(ptry)):
        ptry[j]=psum[j]*factor1-p[ihigh][j]*factor2
        #Do not allow values smaller than zero -> use boundary here
        if ptry[j]<0:
            ptry[j]=0
    ytry=fn(ptry)
    if ytry<y[ihigh]:

        y[ihigh]=ytry
        for j in range(size(ptry)):
            psum[j]= psum[j]-p[ihigh][j]+ptry[j]
            p[ihigh][j]=ptry[j]
    return ytry   


def get_psum(p):
    psum=zeros(size(p[0]))
    psum=p.sum(axis=0)
    return psum
    

if __name__=="__main__":
    
    g = lambda x: np.power(x-2,2)    
    
    p0=zeros(5)
    p_opt=simplex(p0,function2, False)
    print p_opt
    
    
    
    
    
    
    
    
    
    