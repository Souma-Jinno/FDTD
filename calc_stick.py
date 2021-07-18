# from numba.np.ufunc import parallel
import numpy as np

from MakeSignal import *
from keisu_calc import keisu_calc
from numba import jit,prange

@jit(nopython=True,parallel=True)
def calc_stick(Signal, dx, dy, dz, dt, nx, ny, nz, nt, eps, mu, rho, PIX, PIY):
   
    #---------------------------------------------------------------------------------------------------------------------------------------------------
    #　係数の計算
    #---------------------------------------------------------------------------------------------------------------------------------------------------
    dhx, dhy, dhz, ce, dex, dey, dez, de= keisu_calc(dx, dy, dz, dt, eps, mu, rho)

    #---------------------------------------------------------------------------------------------------------------------------------------------------
    #　初期化
    #---------------------------------------------------------------------------------------------------------------------------------------------------
    E_x = np.zeros(shape=(nx, ny+1, nz+1, nt))
    E_y = np.zeros(shape=(nx+1, ny, nz+1, nt))
    E_z = np.zeros(shape=(nx+1, ny+1, nz, nt))
    H_x = np.zeros(shape=(nx+1, ny, nz, nt))
    H_y = np.zeros(shape=(nx, ny+1, nz, nt))
    H_z = np.zeros(shape=(nx, ny, nz+1, nt)) 
    J_x = np.zeros(shape=(nx, ny+1, nz+1, nt)) 
    J_y = np.zeros(shape=(nx+1, ny, nz+1, nt)) 
    J_z = np.zeros(shape=(nx+1, ny+1, nz, nt)) 

    for t in range(nt-1):
        if t%10 == 0:
            print(str(t%10),"%")
        #---------------------------------------------------------------------------------------------------------------------------------------------------
        #　入力信号
        #---------------------------------------------------------------------------------------------------------------------------------------------------
        for i in range(nt):
            x               = int(Signal[i,0])
            y               = int(Signal[i,1])
            z               = int(Signal[i,2])
            J_x[x,y,z,t]    = Signal[i,3+t]
       
        #---------------------------------------------------------------------------------------------------------------------------------------------------
        #　電磁界計算
        #---------------------------------------------------------------------------------------------------------------------------------------------------
        #Hx_calc
        for x in prange(nx+1):   
            for y in range(ny):
                for z in range(nz):                         
                    H_x[x,y,z,t+1] = H_x[x,y,z,t] + dhz[x,y,z] * (E_y[x,y,z+1,t] - E_y[x,y,z,t])\
                                                    - dhy[x,y,z] * (E_z[x,y+1,z,t] - E_z[x,y,z,t])

        #Hy_calc
        for x in prange(nx):
            for y in range(ny-1):
                for z in range(nz):
                    H_y[x,y,z,t+1] = H_y[x,y,z,t] + dhx[x,y,z] * (E_z[x+1,y,z,t] - E_z[x,y,z,t])\
                                                    - dhz[x,y,z] * (E_x[x,y,z+1,t] - E_x[x,y,z,t])

        #Hz_calc
        for x in prange(nx):
            for y in range(ny):
                for z in range(nz-1):
                    H_z[x,y,z,t+1] = H_z[x,y,z,t] + dhy[x,y,z] * (E_x[x,y+1,z,t] - E_x[x,y,z,t])\
                                                        - dhx[x,y,z] * (E_y[x+1,y,z,t] - E_y[x,y,z,t])

        # Ex_calc, shape=(nx, ny+1, nz+1, nt)
        for x in prange(nx):
            for y in range(1,ny):
                for z in range(1,nz):
                    if PIX[x,y,z] == 0.0:
                        E_x[x,y,z,t+1] = ce[x,y,z] * E_x[x,y,z,t] + dey[x,y,z] * (H_z[x,y,z,t+1] - H_z[x,y-1,z,t+1])\
                                                            - dez[x,y,z] * (H_y[x,y,z,t+1] - H_y[x,y,z-1,t+1])\
                                                            - de[x,y,z] * J_x[x,y,z,t+1]
                    else:
                        E_x[x,y,z,t+1] = 0.0
                        J_x[x,y,z,t+1] = (1/dy) * (H_z[x,y,z,t] - H_z[x,y-1,z,t])\
                                                                -(1/dz) * (H_y[x,y,z,t] - H_y[x,y,z-1,t])
        

        #Ey_calc, shape=(nx+1, ny, nz+1, nt)
        for x in prange(1,nx):
            for y in range(ny):
                for z in range(1,nz):   
                    if PIY[x,y,z] == 0:
                        E_y[x,y,z,t+1] = ce[x,y,z] * E_y[x,y,z,t] + dez[x,y,z] * (H_x[x,y,z,t+1] - H_x[x,y,z-1,t+1])\
                                                            - dex[x,y,z] * (H_z[x,y,z,t+1] - H_z[x-1,y,z,t+1])\
                                                            - de[x,y,z] * J_y[x,y,z,t+1]
                    else:
                        E_y[x,y,z,t+1] = 0           
                        J_y[x,y,z,t+1] = (1/dz) * (H_x[x,y,z,t] - H_x[x,y,z-1,t])\
                                                                -(1/dx) * (H_z[x,y,z,t] - H_z[x-1,y,z,t])
        # Ez_calc, shape=(nx+1, ny, nz+1, nt)
        for x in prange(1,nx):
            for y in range(1,ny):
                for z in range(nz):        
                    E_z[x,y,z,t+1] = ce[x,y,z] * E_z[x,y,z,t] + dex[x,y,z] * (H_y[x,y,z,t+1] - H_y[x-1,y,z,t+1])\
                                                        - dey[x,y,z] * (H_x[x,y,z,t+1] - H_x[x,y-1,z,t+1]) \
                                                        # - de[x,y,z] * J_z[x,y,z,t+1]

        

        #---------------------------------------------------------------------------------------------------------------------------------------------------
        #　境界条件の設定
        #---------------------------------------------------------------------------------------------------------------------------------------------------
        α1 = 1.0
        α2 = 1.0
        d1 = 0.005
        d2 = 0.005

        # xy平面（z = 0,-1）
        for x in range(1,nx-1):         # E_x = np.zeros(shape=(nx, ny+1, nz+1, nt))
            for y in range(1,ny):         
                # E_x[x,y,0,t+1] = 0
                v = 1/np.sqrt(eps[x,y,0]*mu[x,y,0])
                b1 = (α1*v*dt-dx)/(α1*v*dt+dx)
                b2 = (α2*v*dt-dx)/(α2*v*dt+dx)  
                E_x[x,y,0,t+1]  = (
                +(b1+b2)*(E_x[x,y,1,t+1]-E_x[x,y,0,t]) 
                - b1*b2*(E_x[x,y,2,t+1]-2*E_x[x,y,1,t]+E_x[x,y,0,t-1])
                -(b1*(1-d2)+b2*(1-d1))*(E_x[x,y,2,t]-E_x[x,y,1,t-1])
                +((1-d1)+(1-d2))*E_x[x,y,1,t]
                -(1-d1)*(1-d2)*E_x[x,y,2,t-1]
                )

                # E_x[x,y,nz-1,t+1] = 0
                v = 1/np.sqrt(eps[x,y,-1]*mu[x,y,-1])
                b1 = (α1*v*dt-dx)/(α1*v*dt+dx)
                b2 = (α2*v*dt-dx)/(α2*v*dt+dx) 
                E_x[x,y,-1,t+1]  = (
                +(b1+b2)*(E_x[x,y,-2,t+1]-E_x[x,y,-1,t]) 
                - b1*b2*(E_x[x,y,-3,t+1]-2*E_x[x,y,-2,t]+E_x[x,y,-1,t-1])
                -(b1*(1-d2)+b2*(1-d1))*(E_x[x,y,-3,t]-E_x[x,y,-2,t-1])
                +((1-d1)+(1-d2))*E_x[x,y,-2,t]
                -(1-d1)*(1-d2)*E_x[x,y,-3,t-1]
                )  
        for x in range(1,nx):       # E_y = np.zeros(shape=(nx+1, ny, nz+1, nt))
            for y in range(1,ny-1):
                # E_y[x,y,0,t+1] = 0
                v = 1/np.sqrt(eps[x,y,0]*mu[x,y,0])
                b1 = (α1*v*dt-dx)/(α1*v*dt+dx)
                b2 = (α2*v*dt-dx)/(α2*v*dt+dx)  
                E_y[x,y,0,t+1]  = (
                +(b1+b2)*(E_y[x,y,1,t+1]-E_y[x,y,0,t]) 
                - b1*b2*(E_y[x,y,2,t+1]-2*E_y[x,y,1,t]+E_y[x,y,0,t-1])
                -(b1*(1-d2)+b2*(1-d1))*(E_y[x,y,2,t]-E_y[x,y,1,t-1])
                +((1-d1)+(1-d2))*E_y[x,y,1,t]
                -(1-d1)*(1-d2)*E_y[x,y,2,t-1]
                )

                # E_y[x,y,nz-1,t+1] = 0
                v = 1/np.sqrt(eps[x,y,-1]*mu[x,y,-1])
                b1 = (α1*v*dt-dx)/(α1*v*dt+dx)
                b2 = (α2*v*dt-dx)/(α2*v*dt+dx) 
                E_y[x,y,-1,t+1]  = (
                +(b1+b2)*(E_y[x,y,-2,t+1]-E_y[x,y,-1,t]) 
                - b1*b2*(E_y[x,y,-3,t+1]-2*E_y[x,y,-2,t]+E_y[x,y,-1,t-1])
                -(b1*(1-d2)+b2*(1-d1))*(E_y[x,y,-3,t]-E_y[x,y,-2,t-1])
                +((1-d1)+(1-d2))*E_y[x,y,-2,t]
                -(1-d1)*(1-d2)*E_y[x,y,-3,t-1]
                )  
        
        # yz平面（x = 0,-1)
        for y in range(1,ny-1):        # E_y = np.zeros(shape=(nx+1, ny, nz+1, nt))
            for z in range(1,nz):
                # E_y[0,y,z,t+1] = 0
                v = 1/np.sqrt(eps[0,y,z]*mu[0,y,z])
                b1 = (α1*v*dt-dx)/(α1*v*dt+dx)
                b2 = (α2*v*dt-dx)/(α2*v*dt+dx)  
                E_y[0,y,z,t+1]  = (
                +(b1+b2)*(E_y[1,y,z,t+1]-E_y[0,y,z,t]) 
                - b1*b2*(E_y[2,y,z,t+1]-2*E_y[1,y,z,t]+E_y[0,y,z,t-1])
                -(b1*(1-d2)+b2*(1-d1))*(E_y[2,y,z,t]-E_y[1,y,z,t-1])
                +((1-d1)+(1-d2))*E_y[1,y,z,t]
                -(1-d1)*(1-d2)*E_y[2,y,z,t-1]
                )
                # E_y[nx-1,y,z,t+1] = 0
                v = 1/np.sqrt(eps[-1,y,z]*mu[-1,y,z])
                b1 = (α1*v*dt-dx)/(α1*v*dt+dx)
                b2 = (α2*v*dt-dx)/(α2*v*dt+dx)  
                E_y[-1,y,z,t+1]  = (
                +(b1+b2)*(E_y[-2,y,z,t+1]-E_y[-1,y,z,t]) 
                - b1*b2*(E_y[-3,y,z,t+1]-2*E_y[-2,y,z,t]+E_y[-1,y,z,t-1])
                -(b1*(1-d2)+b2*(1-d1))*(E_y[-3,y,z,t]-E_y[-2,y,z,t-1])
                +((1-d1)+(1-d2))*E_y[-2,y,z,t]
                -(1-d1)*(1-d2)*E_y[-3,y,z,t-1]
                )  
        for y in range(1,ny):       # E_z = np.zeros(shape=(nx+1, ny+1, nz, nt))
            for z in range(1,nz-1):
                # E_z[0,y,z,t+1] = 0
                v = 1/np.sqrt(eps[0,y,z]*mu[0,y,z])
                b1 = (α1*v*dt-dx)/(α1*v*dt+dx)
                b2 = (α2*v*dt-dx)/(α2*v*dt+dx)  
                E_z[0,y,z,t+1]  = (
                +(b1+b2)*(E_z[1,y,z,t+1]-E_z[0,y,z,t]) 
                - b1*b2*(E_z[2,y,z,t+1]-2*E_z[1,y,z,t]+E_z[0,y,z,t-1])
                -(b1*(1-d2)+b2*(1-d1))*(E_z[2,y,z,t]-E_z[1,y,z,t-1])
                +((1-d1)+(1-d2))*E_z[1,y,z,t]
                -(1-d1)*(1-d2)*E_z[2,y,z,t-1]
                )
                # E_z[nx-1,y,z,t+1] = 0
                v = 1/np.sqrt(eps[-1,y,z]*mu[-1,y,z])
                b1 = (α1*v*dt-dx)/(α1*v*dt+dx)
                b2 = (α2*v*dt-dx)/(α2*v*dt+dx)  
                E_z[-1,y,z,t+1]  = (
                +(b1+b2)*(E_z[-2,y,z,t+1]-E_z[-1,y,z,t]) 
                - b1*b2*(E_z[-3,y,z,t+1]-2*E_z[-2,y,z,t]+E_z[-1,y,z,t-1])
                -(b1*(1-d2)+b2*(1-d1))*(E_z[-3,y,z,t]-E_z[-2,y,z,t-1])
                +((1-d1)+(1-d2))*E_z[-2,y,z,t]
                -(1-d1)*(1-d2)*E_z[-3,y,z,t-1]
                )  
        # zx平面　（y＝0,-1）
        for z in range(1,nz):           # E_x = np.zeros(shape=(nx, ny+1, nz+1, nt))
            for x in range(1,nx-1):
                # E_x[x,0,z,t+1] = 0
                v = 1/np.sqrt(eps[x,0,z]*mu[x,0,z])
                b1 = (α1*v*dt-dx)/(α1*v*dt+dx)
                b2 = (α2*v*dt-dx)/(α2*v*dt+dx)  
                E_x[x,0,z,t+1]  = (
                +(b1+b2)*(E_x[x,1,z,t+1]-E_x[x,0,z,t]) 
                - b1*b2*(E_x[x,2,z,t+1]-2*E_x[x,1,z,t]+E_x[x,0,z,t-1])
                -(b1*(1-d2)+b2*(1-d1))*(E_x[x,2,z,t]-E_x[x,1,z,t-1])
                +((1-d1)+(1-d2))*E_x[x,1,z,t]
                -(1-d1)*(1-d2)*E_x[x,2,z,t-1]
                )
                # E_x[x,ny-1,z,t+1] = 0
                v = 1/np.sqrt(eps[x,-1,z]*mu[x,-1,z])
                b1 = (α1*v*dt-dx)/(α1*v*dt+dx)
                b2 = (α2*v*dt-dx)/(α2*v*dt+dx)  
                E_x[x,-1,z,t+1]  = (
                +(b1+b2)*(E_x[x,-2,z,t+1]-E_x[x,-1,z,t]) 
                - b1*b2*(E_x[x,-3,z,t+1]-2*E_x[x,-2,z,t]+E_x[x,-1,z,t-1])
                -(b1*(1-d2)+b2*(1-d1))*(E_x[x,-3,z,t]-E_x[x,-2,z,t-1])
                +((1-d1)+(1-d2))*E_x[x,-2,z,t]
                -(1-d1)*(1-d2)*E_x[x,-3,z,t-1]
                )  

        for z in range(1,nz-1):       # E_z = np.zeros(shape=(nx+1, ny+1, nz, nt))
            for x in range(1,nx):
                # E_z[x,0,z,t+1] = 0
                v = 1/np.sqrt(eps[x,0,z]*mu[x,0,z])
                b1 = (α1*v*dt-dx)/(α1*v*dt+dx)
                b2 = (α2*v*dt-dx)/(α2*v*dt+dx)  
                E_z[x,0,z,t+1]  = (
                +(b1+b2)*(E_z[x,1,z,t+1]-E_z[x,0,z,t]) 
                - b1*b2*(E_z[x,2,z,t+1]-2*E_z[x,1,z,t]+E_z[x,0,z,t-1])
                -(b1*(1-d2)+b2*(1-d1))*(E_z[x,2,z,t]-E_z[x,1,z,t-1])
                +((1-d1)+(1-d2))*E_z[x,1,z,t]
                -(1-d1)*(1-d2)*E_z[x,2,z,t-1]
                )
                # E_x[x,ny-1,z,t+1] = 0
                v = 1/np.sqrt(eps[x,-1,z]*mu[x,-1,z])
                b1 = (α1*v*dt-dx)/(α1*v*dt+dx)
                b2 = (α2*v*dt-dx)/(α2*v*dt+dx)  
                E_z[x,-1,z,t+1]  = (
                +(b1+b2)*(E_z[x,-2,z,t+1]-E_z[x,-1,z,t]) 
                - b1*b2*(E_z[x,-3,z,t+1]-2*E_z[x,-2,z,t]+E_z[x,-1,z,t-1])
                -(b1*(1-d2)+b2*(1-d1))*(E_z[x,-3,z,t]-E_z[x,-2,z,t-1])
                +((1-d1)+(1-d2))*E_z[x,-2,z,t]
                -(1-d1)*(1-d2)*E_z[x,-3,z,t-1]
                )  

        
        for x in range(nx):
            # E_x[x,0,0,t+1] = 0
            v = 1/np.sqrt(eps[x,0,0]*mu[x,0,0])
            b1 = (α1*v*dt-dx)/(α1*v*dt+dx)
            b2 = (α2*v*dt-dx)/(α2*v*dt+dx)
            E_x[x,0,0,t+1]  = (1/2)*(
                (
                +(b1+b2)*(E_x[x,0,1,t+1]-E_x[x,0,0,t]) 
                - b1*b2*(E_x[x,0,2,t+1]-2*E_x[x,0,1,t]+E_x[x,0,0,t-1])
                -(b1*(1-d2)+b2*(1-d1))*(E_x[x,0,2,t]-E_x[x,0,1,t-1])
                +((1-d1)+(1-d2))*E_x[x,0,1,t]
                -(1-d1)*(1-d2)*E_x[x,0,2,t-1]
                )
                +
                (
                +(b1+b2)*(E_x[x,1,0,t+1]-E_x[x,0,0,t]) 
                - b1*b2*(E_x[x,2,0,t+1]-2*E_x[x,1,0,t]+E_x[x,0,0,t-1])
                -(b1*(1-d2)+b2*(1-d1))*(E_x[x,2,0,t]-E_x[x,1,0,t-1])
                +((1-d1)+(1-d2))*E_x[x,1,0,t]
                -(1-d1)*(1-d2)*E_x[x,2,0,t-1]
                )
            )
            # E_x[x,-1,0,t+1] = 0
            v = 1/np.sqrt(eps[x,-1,0]*mu[x,-1,0])
            b1 = (α1*v*dt-dx)/(α1*v*dt+dx)
            b2 = (α2*v*dt-dx)/(α2*v*dt+dx)
            E_x[x,-1,0,t+1] = (1/2)*(
                (
                +(b1+b2)*(E_x[x,-1,1,t+1]-E_x[x,-1,0,t]) 
                - b1*b2*(E_x[x,-1,2,t+1]-2*E_x[x,-1,1,t]+E_x[x,-1,0,t-1])
                -(b1*(1-d2)+b2*(1-d1))*(E_x[x,-1,2,t]-E_x[x,-1,1,t-1])
                +((1-d1)+(1-d2))*E_x[x,-1,1,t]
                -(1-d1)*(1-d2)*E_x[x,-1,2,t-1]
                )
                +
                (
                +(b1+b2)*(E_x[x,-2,0,t+1]-E_x[x,-1,0,t]) 
                - b1*b2*(E_x[x,-3,0,t+1]-2*E_x[x,-2,0,t]+E_x[x,-1,0,t-1])
                -(b1*(1-d2)+b2*(1-d1))*(E_x[x,-3,0,t]-E_x[x,-2,0,t-1])
                +((1-d1)+(1-d2))*E_x[x,-2,0,t]
                -(1-d1)*(1-d2)*E_x[x,-3,0,t-1]
                )  
            )
            # E_x[x,0,-1,t+1] = 0
            v = 1/np.sqrt(eps[x,0,-1]*mu[x,0,-1])
            b1 = (α1*v*dt-dx)/(α1*v*dt+dx)
            b2 = (α2*v*dt-dx)/(α2*v*dt+dx)
            E_x[x,0,-1,t+1] = (1/2)*(
                (
                +(b1+b2)*(E_x[x,1,-1,t+1]-E_x[x,0,-1,t]) 
                - b1*b2*(E_x[x,2,-1,t+1]-2*E_x[x,1,-1,t]+E_x[x,0,-1,t-1])
                -(b1*(1-d2)+b2*(1-d1))*(E_x[x,2,-1,t]-E_x[x,1,-1,t-1])
                +((1-d1)+(1-d2))*E_x[x,1,-1,t]
                -(1-d1)*(1-d2)*E_x[x,2,-1,t-1]
                )
                +
                (
                +(b1+b2)*(E_x[x,0,-2,t+1]-E_x[x,0,-1,t]) 
                - b1*b2*(E_x[x,0,-3,t+1]-2*E_x[x,0,-2,t]+E_x[x,0,-1,t-1])
                -(b1*(1-d2)+b2*(1-d1))*(E_x[x,0,-3,t]-E_x[x,0,-2,t-1])
                +((1-d1)+(1-d2))*E_x[x,0,-2,t]
                -(1-d1)*(1-d2)*E_x[x,0,-3,t-1]
                )  
            )

            # E_x[x,ny-1,nz-1,t+1] = 0
            v = 1/np.sqrt(eps[x,-1,-1]*mu[x,-1,-1])
            b1 = (α1*v*dt-dx)/(α1*v*dt+dx)
            b2 = (α2*v*dt-dx)/(α2*v*dt+dx)
            E_x[x,-1,-1,t+1] =  (1/2)*(
                (
                +(b1+b2)*(E_x[x,-2,-1,t+1]-E_x[x,-1,-1,t]) 
                - b1*b2*(E_x[x,-3,-1,t+1]-2*E_x[x,-2,-1,t]+E_x[x,-1,-1,t-1])
                -(b1*(1-d2)+b2*(1-d1))*(E_x[x,-3,-1,t]-E_x[x,-2,-1,t-1])
                +((1-d1)+(1-d2))*E_x[x,-2,-1,t]
                -(1-d1)*(1-d2)*E_x[x,-3,-1,t-1]
                )
                +
                (
                +(b1+b2)*(E_x[x,-1,-2,t+1]-E_x[x,-1,-1,t]) 
                - b1*b2*(E_x[x,-1,-3,t+1]-2*E_x[x,-1,-2,t]+E_x[x,-1,-1,t-1])
                -(b1*(1-d2)+b2*(1-d1))*(E_x[x,-1,-3,t]-E_x[x,-1,-2,t-1])
                +((1-d1)+(1-d2))*E_x[x,-1,-2,t]
                -(1-d1)*(1-d2)*E_x[x,-1,-3,t-1]
                )  
            )

        #境界面のEy=0                     
        for y in range(ny):
            # E_y[0,y,0,t+1] = 0
            v = 1/np.sqrt(eps[0,y,0]*mu[0,y,0])
            b1 = (α1*v*dt-dx)/(α1*v*dt+dx)
            b2 = (α2*v*dt-dx)/(α2*v*dt+dx)
            E_y[0,y,0,t+1]  = (1/2)*(
                (
                +(b1+b2)*(E_y[0,y,1,t+1]-E_y[0,y,0,t]) 
                - b1*b2*(E_y[0,y,2,t+1]-2*E_y[0,y,1,t]+E_y[0,y,0,t-1])
                -(b1*(1-d2)+b2*(1-d1))*(E_y[0,y,2,t]-E_y[0,y,1,t-1])
                +((1-d1)+(1-d2))*E_y[0,y,1,t]
                -(1-d1)*(1-d2)*E_y[0,y,2,t-1]
                )
                +
                (
                +(b1+b2)*(E_y[1,y,0,t+1]-E_y[0,y,0,t]) 
                - b1*b2*(E_y[2,y,0,t+1]-2*E_y[1,y,0,t]+E_y[0,y,0,t-1])
                -(b1*(1-d2)+b2*(1-d1))*(E_y[2,y,0,t]-E_y[1,y,0,t-1])
                +((1-d1)+(1-d2))*E_y[1,y,0,t]
                -(1-d1)*(1-d2)*E_y[2,y,0,t-1]
                )
            )
            # E_y[nx-1,y,0,t+1] = 0
            v = 1/np.sqrt(eps[-1,y,0]*mu[-1,y,0])
            b1 = (α1*v*dt-dx)/(α1*v*dt+dx)
            b2 = (α2*v*dt-dx)/(α2*v*dt+dx)
            E_y[-1,y,0,t+1]  = (1/2)*(
                (
                +(b1+b2)*(E_y[-1,y,1,t+1]-E_y[-1,y,0,t]) 
                - b1*b2*(E_y[-1,y,2,t+1]-2*E_y[-1,y,1,t]+E_y[-1,y,0,t-1])
                -(b1*(1-d2)+b2*(1-d1))*(E_y[-1,y,2,t]-E_y[-1,y,1,t-1])
                +((1-d1)+(1-d2))*E_y[-1,y,1,t]
                -(1-d1)*(1-d2)*E_y[-1,y,2,t-1]
                )
                +
                (
                +(b1+b2)*(E_y[-2,y,0,t+1]-E_y[-1,y,0,t]) 
                - b1*b2*(E_y[-3,y,0,t+1]-2*E_y[-2,y,0,t]+E_y[-1,y,0,t-1])
                -(b1*(1-d2)+b2*(1-d1))*(E_y[-3,y,0,t]-E_y[-2,y,0,t-1])
                +((1-d1)+(1-d2))*E_y[-2,y,0,t]
                -(1-d1)*(1-d2)*E_y[-3,y,0,t-1]
                )
            )

            # E_y[0,y,nz-1,t+1] = 0
            v = 1/np.sqrt(eps[0,y,-1]*mu[0,y,-1])
            b1 = (α1*v*dt-dx)/(α1*v*dt+dx)
            b2 = (α2*v*dt-dx)/(α2*v*dt+dx)
            E_y[0,y,-1,t+1]  = (1/2)*(
                (
                +(b1+b2)*(E_y[0,y,-2,t+1]-E_y[0,y,-1,t]) 
                - b1*b2*(E_y[0,y,-3,t+1]-2*E_y[0,y,-2,t]+E_y[0,y,-1,t-1])
                -(b1*(1-d2)+b2*(1-d1))*(E_y[0,y,-3,t]-E_y[0,y,-2,t-1])
                +((1-d1)+(1-d2))*E_y[0,y,-2,t]
                -(1-d1)*(1-d2)*E_y[0,y,-3,t-1]
                )
                +
                (
                +(b1+b2)*(E_y[1,y,-1,t+1]-E_y[0,y,-1,t]) 
                - b1*b2*(E_y[2,y,-1,t+1]-2*E_y[1,y,-1,t]+E_y[0,y,-1,t-1])
                -(b1*(1-d2)+b2*(1-d1))*(E_y[2,y,-1,t]-E_y[1,y,-1,t-1])
                +((1-d1)+(1-d2))*E_y[1,y,-1,t]
                -(1-d1)*(1-d2)*E_y[2,y,-1,t-1]
                )
            )

            # E_y[-1,y,-1,t+1] = 0
            v = 1/np.sqrt(eps[0,y,-1]*mu[0,y,-1])
            b1 = (α1*v*dt-dx)/(α1*v*dt+dx)
            b2 = (α2*v*dt-dx)/(α2*v*dt+dx)
            E_y[-1,y,-1,t+1]  = (1/2)*(
                (
                +(b1+b2)*(E_y[0,y,-2,t+1]-E_y[0,y,-1,t]) 
                - b1*b2*(E_y[0,y,-3,t+1]-2*E_y[0,y,-2,t]+E_y[0,y,-1,t-1])
                -(b1*(1-d2)+b2*(1-d1))*(E_y[0,y,-3,t]-E_y[0,y,-2,t-1])
                +((1-d1)+(1-d2))*E_y[0,y,-2,t]
                -(1-d1)*(1-d2)*E_y[0,y,-3,t-1]
                )
                +
                (
                +(b1+b2)*(E_y[-2,y,-1,t+1]-E_y[-1,y,-1,t]) 
                - b1*b2*(E_y[-3,y,-1,t+1]-2*E_y[-2,y,-1,t]+E_y[-1,y,-1,t-1])
                -(b1*(1-d2)+b2*(1-d1))*(E_y[-3,y,-1,t]-E_y[-2,y,-1,t-1])
                +((1-d1)+(1-d2))*E_y[-2,y,-1,t]
                -(1-d1)*(1-d2)*E_y[-3,y,-1,t-1]
                )
            )

        #境界面のEz=0
        for z in range(nz):
            # E_z[0,0,z,t+1] = 0
            v = 1/np.sqrt(eps[0,0,z]*mu[0,0,z])
            b1 = (α1*v*dt-dx)/(α1*v*dt+dx)
            b2 = (α2*v*dt-dx)/(α2*v*dt+dx)
            E_z[0,0,z,t+1]  = (1/2)*(
                (
                +(b1+b2)*(E_z[0,1,z,t+1]-E_z[0,0,z,t]) 
                - b1*b2*(E_z[0,2,z,t+1]-2*E_z[0,1,z,t]+E_z[0,0,z,t-1])
                -(b1*(1-d2)+b2*(1-d1))*(E_z[0,2,z,t]-E_z[0,1,z,t-1])
                +((1-d1)+(1-d2))*E_z[0,1,z,t]
                -(1-d1)*(1-d2)*E_z[0,2,z,t-1]
                )
                +
                (
                +(b1+b2)*(E_z[1,0,z,t+1]-E_z[0,0,z,t]) 
                - b1*b2*(E_z[2,0,z,t+1]-2*E_z[1,0,z,t]+E_z[0,0,z,t-1])
                -(b1*(1-d2)+b2*(1-d1))*(E_z[2,0,z,t]-E_z[1,0,z,t-1])
                +((1-d1)+(1-d2))*E_z[1,0,z,t]
                -(1-d1)*(1-d2)*E_z[2,0,z,t-1]
                )
            )

            # E_z[nx-1,0,z,t+1] = 0
            v = 1/np.sqrt(eps[-1,0,z]*mu[-1,0,z])
            b1 = (α1*v*dt-dx)/(α1*v*dt+dx)
            b2 = (α2*v*dt-dx)/(α2*v*dt+dx)
            E_z[-1,0,z,t+1]  = (1/2)*(
                (
                +(b1+b2)*(E_z[-1,1,z,t+1]-E_z[-1,0,z,t]) 
                - b1*b2*(E_z[-1,2,z,t+1]-2*E_z[-1,1,z,t]+E_z[-1,0,z,t-1])
                -(b1*(1-d2)+b2*(1-d1))*(E_z[-1,2,z,t]-E_z[-1,1,z,t-1])
                +((1-d1)+(1-d2))*E_z[-1,1,z,t]
                -(1-d1)*(1-d2)*E_z[-1,2,z,t-1]
                )
                +
                (
                +(b1+b2)*(E_z[-2,0,z,t+1]-E_z[-1,0,z,t]) 
                - b1*b2*(E_z[-3,0,z,t+1]-2*E_z[-2,0,z,t]+E_z[-1,0,z,t-1])
                -(b1*(1-d2)+b2*(1-d1))*(E_z[-3,0,z,t]-E_z[-2,0,z,t-1])
                +((1-d1)+(1-d2))*E_z[-2,0,z,t]
                -(1-d1)*(1-d2)*E_z[-3,0,z,t-1]
                )
            )

            # E_z[0,ny-1,z,t+1] = 0
            v = 1/np.sqrt(eps[0,-1,z]*mu[0,-1,z])
            b1 = (α1*v*dt-dx)/(α1*v*dt+dx)
            b2 = (α2*v*dt-dx)/(α2*v*dt+dx)
            E_z[0,-1,z,t+1]  = (1/2)*(
                (
                +(b1+b2)*(E_z[0,-2,z,t+1]-E_z[0,-1,z,t]) 
                - b1*b2*(E_z[0,2,z,t+1]-2*E_z[0,1,z,t]+E_z[0,-1,z,t-1])
                -(b1*(1-d2)+b2*(1-d1))*(E_z[0,-3,z,t]-E_z[0,-2,z,t-1])
                +((1-d1)+(1-d2))*E_z[0,-2,z,t]
                -(1-d1)*(1-d2)*E_z[0,-3,z,t-1]
                )
                +
                (
                +(b1+b2)*(E_z[1,0,z,t+1]-E_z[0,0,z,t]) 
                - b1*b2*(E_z[2,0,z,t+1]-2*E_z[1,0,z,t]+E_z[0,0,z,t-1])
                -(b1*(1-d2)+b2*(1-d1))*(E_z[2,0,z,t]-E_z[1,0,z,t-1])
                +((1-d1)+(1-d2))*E_z[1,0,z,t]
                -(1-d1)*(1-d2)*E_z[2,0,z,t-1]
                )
            )

            # E_z[nx-1,ny-1,z,t+1] = 0 
            v = 1/np.sqrt(eps[-1,-1,z]*mu[-1,-1,z])
            b1 = (α1*v*dt-dx)/(α1*v*dt+dx)
            b2 = (α2*v*dt-dx)/(α2*v*dt+dx)
            E_z[-1,-1,z,t+1]  = (1/2)*(
                (
                +(b1+b2)*(E_z[-1,-2,z,t+1]-E_z[-1,-1,z,t]) 
                - b1*b2*(E_z[-1,2,z,t+1]-2*E_z[-1,1,z,t]+E_z[-1,-1,z,t-1])
                -(b1*(1-d2)+b2*(1-d1))*(E_z[-1,-3,z,t]-E_z[-1,-2,z,t-1])
                +((1-d1)+(1-d2))*E_z[-1,-2,z,t]
                -(1-d1)*(1-d2)*E_z[-1,-3,z,t-1]
                )
                +
                (
                +(b1+b2)*(E_z[-2,-1,z,t+1]-E_z[-1,-1,z,t]) 
                - b1*b2*(E_z[-3,-1,z,t+1]-2*E_z[-2,-1,z,t]+E_z[-1,-1,z,t-1])
                -(b1*(1-d2)+b2*(1-d1))*(E_z[-3,-1,z,t]-E_z[-2,-1,z,t-1])
                +((1-d1)+(1-d2))*E_z[-2,-1,z,t]
                -(1-d1)*(1-d2)*E_z[-3,-1,z,t-1]
                )
            )



    return H_x, H_y, H_z, E_x, E_y, E_z, J_z


    
    
  

