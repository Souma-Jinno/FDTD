# from numba.np.ufunc import parallel
import numpy as np

from MakeSignal import *
from numba import jit,prange

@jit(nopython=True,parallel=True)
# @jit(nopython=True)
def calc_arbitral(CirName,step_save,Signal, nInput, dx, dy, dz, dt, nx, ny, nz, nt, eps, mu, rho, PIX, PIY,PIZ,dhy_Hx,dhz_Hx,dhx_Hy,dhz_Hy,dhy_Hz,dhx_Hz,ce,dex,dey,dez,de):
   
    #---------------------------------------------------------------------------------------------------------------------------------------------------
    #　初期化
    #---------------------------------------------------------------------------------------------------------------------------------------------------
    E_x = np.zeros(shape=(nx,  ny+1, nz+1,3))
    E_y = np.zeros(shape=(nx+1,ny,   nz+1,3))
    E_z = np.zeros(shape=(nx+1,ny+1, nz,  3))
    H_x = np.zeros(shape=(nx+1,ny,   nz,  3))
    H_y = np.zeros(shape=(nx,  ny+1, nz,  3))
    H_z = np.zeros(shape=(nx,  ny,   nz+1,3)) 
    J_x = np.zeros(shape=(nx,  ny+1, nz+1,3)) 
    J_y = np.zeros(shape=(nx+1,ny,   nz+1,3)) 
    J_z = np.zeros(shape=(nx+1,ny+1, nz,  3)) 
    # E_x_save = np.zeros(shape=(nx,  ny+1, nz+1,nt))
    nt2 = int(nt/step_save)
    Jx_yz = np.zeros(shape=(ny+1, nz+1,nt2))
    Hx_yz = np.zeros(shape=(ny,   nz,  nt2))
    Hy_yz = np.zeros(shape=(ny+1, nz,  nt2))
    Hz_yz = np.zeros(shape=(ny,   nz+1,nt2))
    Ex_yz = np.zeros(shape=(ny+1, nz+1,nt2))
    Ey_yz = np.zeros(shape=(ny,   nz+1,nt2))
    Ez_yz = np.zeros(shape=(ny+1, nz,nt2)) 

    Jx_xz = np.zeros(shape=(nx,   nz+1,nt2))
    Hx_xz = np.zeros(shape=(nx+1, nz,  nt2))
    Hy_xz = np.zeros(shape=(nx,   nz,  nt2))
    Hz_xz = np.zeros(shape=(nx,   nz+1,nt2))
    Ex_xz = np.zeros(shape=(nx,   nz+1,nt2))
    Ey_xz = np.zeros(shape=(nx+1, nz+1,nt2))
    Ez_xz = np.zeros(shape=(nx+1, nz,nt2)) 

    for t in range(nt):
        if t% 100 == 0:
            print(int(t/nt *100),"%")
        #---------------------------------------------------------------------------------------------------------------------------------------------------
        #　入力信号
        #---------------------------------------------------------------------------------------------------------------------------------------------------
        for i in range(nInput):
            x               = int(Signal[i,0])-1
            y               = int(Signal[i,1])-1
            z               = int(Signal[i,2])-1
            J_x[x,y,z,0]    = 1/(dy*dz)*Signal[i,3+t]
       
        #---------------------------------------------------------------------------------------------------------------------------------------------------
        #　電磁界計算
        #---------------------------------------------------------------------------------------------------------------------------------------------------
        #Hx_calc
        for x in range(nx+1):   
            for y in range(ny):
                for z in range(nz):                         
                    H_x[x,y,z,0] = H_x[x,y,z,1] + dhz_Hx[x,y,z] * (E_y[x,y,z+1,1] - E_y[x,y,z,1])\
                                                    - dhy_Hx[x,y,z] * (E_z[x,y+1,z,1] - E_z[x,y,z,1])

        #Hy_calc
        for x in range(nx):
            for y in range(ny+1):
                for z in range(nz):
                    H_y[x,y,z,0] = H_y[x,y,z,1] + dhx_Hy[x,y,z] * (E_z[x+1,y,z,1] - E_z[x,y,z,1])\
                                                    - dhz_Hy[x,y,z] * (E_x[x,y,z+1,1] - E_x[x,y,z,1])

        #Hz_calc
        for x in range(nx):
            for y in range(ny):
                for z in range(nz+1):
                    H_z[x,y,z,0] = H_z[x,y,z,1] + dhy_Hz[x,y,z] * (E_x[x,y+1,z,1] - E_x[x,y,z,1])\
                                                        - dhx_Hz[x,y,z] * (E_y[x+1,y,z,1] - E_y[x,y,z,1])

        # Ex_calc, shape=(nx, ny+1, nz+1, nt)
        for x in range(nx):
            for y in range(1,ny):
                for z in range(1,nz):
                    if PIX[x,y,z] == 0.0:
                        E_x[x,y,z,0] =    (1/4)*(ce[x,y-1,z-1]+ce[x,y,z-1]+ce[x,y-1,z]+ce[x,y,z]) * E_x[x,y,z,1]\
                                        + (1/4)*(dey[x,y-1,z-1]+dey[x,y,z-1]+dey[x,y-1,z]+dey[x,y,z]) * (H_z[x,y,z,0] - H_z[x,y-1,z,0])\
                                        - (1/4)*(dez[x,y-1,z-1]+dez[x,y,z-1]+dez[x,y-1,z]+dez[x,y,z]) * (H_y[x,y,z,0] - H_y[x,y,z-1,0])\
                                        - (1/4)*(de[x,y-1,z-1]+de[x,y,z-1]+de[x,y-1,z]+de[x,y,z]) * J_x[x,y,z,0]
                    else:
                        E_x[x,y,z,0] = 0.0
                        J_x[x,y,z,0] = (1/dy) * (H_z[x,y,z,0] - H_z[x,y-1,z,0])\
                                                                -(1/dz) * (H_y[x,y,z,0] - H_y[x,y,z-1,0])
        

        #Ey_calc, shape=(nx+1, ny, nz+1, nt)
        for x in range(1,nx):
            for y in range(ny):
                for z in range(1,nz):   
                    if PIY[x,y,z] == 0:
                        E_y[x,y,z,0] =    (1/4)*(ce[x-1,y,z-1]+ce[x-1,y,z]+ce[x,y,z-1]+ce[x,y,z]) * E_y[x,y,z,1] \
                                        + (1/4)*(dez[x-1,y,z-1]+dez[x-1,y,z]+dez[x,y,z-1]+dez[x,y,z]) * (H_x[x,y,z,0] - H_x[x,y,z-1,0])\
                                        - (1/4)*(dex[x-1,y,z-1]+dex[x-1,y,z]+dex[x,y,z-1]+dex[x,y,z]) * (H_z[x,y,z,0] - H_z[x-1,y,z,0])\
                                        - (1/4)*(de[x-1,y,z]+de[x-1,y,z]+de[x,y,z-1]+de[x,y,z]) * J_y[x,y,z,0]
                    else:
                        E_y[x,y,z,0] = 0.0           
                        J_y[x,y,z,0] = (1/dz) * (H_x[x,y,z,0] - H_x[x,y,z-1,0])\
                                                                -(1/dx) * (H_z[x,y,z,0] - H_z[x-1,y,z,0])

        # Ez_calc, shape=(nx+1, ny, nz+1, nt)   # 薄さ0の平面導体としているため、z方向の電流は考えない
        for x in range(1,nx):
            for y in range(1,ny):
                for z in range(nz):
                    if PIZ[x,y,z] == 0:
                        E_z[x,y,z,0] = (1/4)*(ce[x-1,y-1,z]+ce[x-1,y,z]+ce[x,y-1,z]+ce[x,y,z]) * E_z[x,y,z,1]\
                                        + (1/4)*(dex[x-1,y-1,z]+dex[x-1,y,z]+dex[x,y-1,z]+dex[x,y,z]) * (H_y[x,y,z,0] - H_y[x-1,y,z,0])\
                                        - (1/4)*(dey[x-1,y-1,z]+dey[x-1,y,z]+dey[x,y-1,z]+dey[x,y,z]) * (H_x[x,y,z,0] - H_x[x,y-1,z,0]) \
                                        - (1/4)*(de[x-1,y-1,z]+de[x-1,y,z]+de[x,y-1,z]+de[x,y,z]) * J_z[x,y,z,0]
                    else:
                        E_z[x,y,z,0] = 0.0
                        J_z[x,y,z,0] = (1/dx) * (H_y[x,y,z,0]-H_y[x-1,y,z,0])\
                                        - (1/dy) * (H_x[x,y,z,0]-H_x[x,y-1,z,0])
        

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
                # E_x[x,y,0,0] = 0
                v = 1/np.sqrt(eps[x,y,0]*mu[x,y,0])
                b1 = (α1*v*dt-dx)/(α1*v*dt+dx)
                b2 = (α2*v*dt-dx)/(α2*v*dt+dx)  
                E_x[x,y,0,0]  = (
                +(b1+b2)*(E_x[x,y,1,0]-E_x[x,y,0,1]) 
                - b1*b2*(E_x[x,y,2,0]-2*E_x[x,y,1,1]+E_x[x,y,0,2])
                -(b1*(1-d2)+b2*(1-d1))*(E_x[x,y,2,1]-E_x[x,y,1,2])
                +((1-d1)+(1-d2))*E_x[x,y,1,1]
                -(1-d1)*(1-d2)*E_x[x,y,2,2]
                )

                # E_x[x,y,nz-1,0] = 0
                v = 1/np.sqrt(eps[x,y,-1]*mu[x,y,-1])
                b1 = (α1*v*dt-dx)/(α1*v*dt+dx)
                b2 = (α2*v*dt-dx)/(α2*v*dt+dx) 
                E_x[x,y,-1,0]  = (
                +(b1+b2)*(E_x[x,y,-2,0]-E_x[x,y,-1,1]) 
                - b1*b2*(E_x[x,y,-3,0]-2*E_x[x,y,-2,1]+E_x[x,y,-1,2])
                -(b1*(1-d2)+b2*(1-d1))*(E_x[x,y,-3,1]-E_x[x,y,-2,2])
                +((1-d1)+(1-d2))*E_x[x,y,-2,1]
                -(1-d1)*(1-d2)*E_x[x,y,-3,2]
                )  
        for x in range(1,nx):       # E_y = np.zeros(shape=(nx+1, ny, nz+1, nt))
            for y in range(1,ny-1):
                # E_y[x,y,0,0] = 0
                v = 1/np.sqrt(eps[x,y,0]*mu[x,y,0])
                b1 = (α1*v*dt-dx)/(α1*v*dt+dx)
                b2 = (α2*v*dt-dx)/(α2*v*dt+dx)  
                E_y[x,y,0,0]  = (
                +(b1+b2)*(E_y[x,y,1,0]-E_y[x,y,0,1]) 
                - b1*b2*(E_y[x,y,2,0]-2*E_y[x,y,1,1]+E_y[x,y,0,2])
                -(b1*(1-d2)+b2*(1-d1))*(E_y[x,y,2,1]-E_y[x,y,1,2])
                +((1-d1)+(1-d2))*E_y[x,y,1,1]
                -(1-d1)*(1-d2)*E_y[x,y,2,2]
                )

                # E_y[x,y,nz-1,0] = 0
                v = 1/np.sqrt(eps[x,y,-1]*mu[x,y,-1])
                b1 = (α1*v*dt-dx)/(α1*v*dt+dx)
                b2 = (α2*v*dt-dx)/(α2*v*dt+dx) 
                E_y[x,y,-1,0]  = (
                +(b1+b2)*(E_y[x,y,-2,0]-E_y[x,y,-1,1]) 
                - b1*b2*(E_y[x,y,-3,0]-2*E_y[x,y,-2,1]+E_y[x,y,-1,2])
                -(b1*(1-d2)+b2*(1-d1))*(E_y[x,y,-3,1]-E_y[x,y,-2,2])
                +((1-d1)+(1-d2))*E_y[x,y,-2,1]
                -(1-d1)*(1-d2)*E_y[x,y,-3,2]
                )  
        
        # yz平面（x = 0,-1)
        for y in range(1,ny-1):        # E_y = np.zeros(shape=(nx+1, ny, nz+1, nt))
            for z in range(1,nz):
                # E_y[0,y,z,0] = 0
                v = 1/np.sqrt(eps[0,y,z]*mu[0,y,z])
                b1 = (α1*v*dt-dx)/(α1*v*dt+dx)
                b2 = (α2*v*dt-dx)/(α2*v*dt+dx)  
                E_y[0,y,z,0]  = (
                +(b1+b2)*(E_y[1,y,z,0]-E_y[0,y,z,1]) 
                - b1*b2*(E_y[2,y,z,0]-2*E_y[1,y,z,1]+E_y[0,y,z,2])
                -(b1*(1-d2)+b2*(1-d1))*(E_y[2,y,z,1]-E_y[1,y,z,2])
                +((1-d1)+(1-d2))*E_y[1,y,z,1]
                -(1-d1)*(1-d2)*E_y[2,y,z,2]
                )
                # E_y[nx-1,y,z,0] = 0
                v = 1/np.sqrt(eps[-1,y,z]*mu[-1,y,z])
                b1 = (α1*v*dt-dx)/(α1*v*dt+dx)
                b2 = (α2*v*dt-dx)/(α2*v*dt+dx)  
                E_y[-1,y,z,0]  = (
                +(b1+b2)*(E_y[-2,y,z,0]-E_y[-1,y,z,1]) 
                - b1*b2*(E_y[-3,y,z,0]-2*E_y[-2,y,z,1]+E_y[-1,y,z,2])
                -(b1*(1-d2)+b2*(1-d1))*(E_y[-3,y,z,1]-E_y[-2,y,z,2])
                +((1-d1)+(1-d2))*E_y[-2,y,z,1]
                -(1-d1)*(1-d2)*E_y[-3,y,z,2]
                )  
        for y in range(1,ny):       # E_z = np.zeros(shape=(nx+1, ny+1, nz, nt))
            for z in range(1,nz-1):
                # E_z[0,y,z,0] = 0
                v = 1/np.sqrt(eps[0,y,z]*mu[0,y,z])
                b1 = (α1*v*dt-dx)/(α1*v*dt+dx)
                b2 = (α2*v*dt-dx)/(α2*v*dt+dx)  
                E_z[0,y,z,0]  = (
                +(b1+b2)*(E_z[1,y,z,0]-E_z[0,y,z,1]) 
                - b1*b2*(E_z[2,y,z,0]-2*E_z[1,y,z,1]+E_z[0,y,z,2])
                -(b1*(1-d2)+b2*(1-d1))*(E_z[2,y,z,1]-E_z[1,y,z,2])
                +((1-d1)+(1-d2))*E_z[1,y,z,1]
                -(1-d1)*(1-d2)*E_z[2,y,z,2]
                )
                # E_z[nx-1,y,z,0] = 0
                v = 1/np.sqrt(eps[-1,y,z]*mu[-1,y,z])
                b1 = (α1*v*dt-dx)/(α1*v*dt+dx)
                b2 = (α2*v*dt-dx)/(α2*v*dt+dx)  
                E_z[-1,y,z,0]  = (
                +(b1+b2)*(E_z[-2,y,z,0]-E_z[-1,y,z,1]) 
                - b1*b2*(E_z[-3,y,z,0]-2*E_z[-2,y,z,1]+E_z[-1,y,z,2])
                -(b1*(1-d2)+b2*(1-d1))*(E_z[-3,y,z,1]-E_z[-2,y,z,2])
                +((1-d1)+(1-d2))*E_z[-2,y,z,1]
                -(1-d1)*(1-d2)*E_z[-3,y,z,2]
                )  
        # zx平面　（y＝0,-1）
        for z in range(1,nz):           # E_x = np.zeros(shape=(nx, ny+1, nz+1, nt))
            for x in range(1,nx-1):
                # E_x[x,0,z,0] = 0
                v = 1/np.sqrt(eps[x,0,z]*mu[x,0,z])
                b1 = (α1*v*dt-dx)/(α1*v*dt+dx)
                b2 = (α2*v*dt-dx)/(α2*v*dt+dx)  
                E_x[x,0,z,0]  = (
                +(b1+b2)*(E_x[x,1,z,0]-E_x[x,0,z,1]) 
                - b1*b2*(E_x[x,2,z,0]-2*E_x[x,1,z,1]+E_x[x,0,z,2])
                -(b1*(1-d2)+b2*(1-d1))*(E_x[x,2,z,1]-E_x[x,1,z,2])
                +((1-d1)+(1-d2))*E_x[x,1,z,1]
                -(1-d1)*(1-d2)*E_x[x,2,z,2]
                )
                # E_x[x,ny-1,z,0] = 0
                v = 1/np.sqrt(eps[x,-1,z]*mu[x,-1,z])
                b1 = (α1*v*dt-dx)/(α1*v*dt+dx)
                b2 = (α2*v*dt-dx)/(α2*v*dt+dx)  
                E_x[x,-1,z,0]  = (
                +(b1+b2)*(E_x[x,-2,z,0]-E_x[x,-1,z,1]) 
                - b1*b2*(E_x[x,-3,z,0]-2*E_x[x,-2,z,1]+E_x[x,-1,z,2])
                -(b1*(1-d2)+b2*(1-d1))*(E_x[x,-3,z,1]-E_x[x,-2,z,2])
                +((1-d1)+(1-d2))*E_x[x,-2,z,1]
                -(1-d1)*(1-d2)*E_x[x,-3,z,2]
                )  

        for z in range(1,nz-1):       # E_z = np.zeros(shape=(nx+1, ny+1, nz, nt))
            for x in range(1,nx):
                # E_z[x,0,z,0] = 0
                v = 1/np.sqrt(eps[x,0,z]*mu[x,0,z])
                b1 = (α1*v*dt-dx)/(α1*v*dt+dx)
                b2 = (α2*v*dt-dx)/(α2*v*dt+dx)  
                E_z[x,0,z,0]  = (
                +(b1+b2)*(E_z[x,1,z,0]-E_z[x,0,z,1]) 
                - b1*b2*(E_z[x,2,z,0]-2*E_z[x,1,z,1]+E_z[x,0,z,2])
                -(b1*(1-d2)+b2*(1-d1))*(E_z[x,2,z,1]-E_z[x,1,z,2])
                +((1-d1)+(1-d2))*E_z[x,1,z,1]
                -(1-d1)*(1-d2)*E_z[x,2,z,2]
                )
                # E_x[x,ny-1,z,0] = 0
                v = 1/np.sqrt(eps[x,-1,z]*mu[x,-1,z])
                b1 = (α1*v*dt-dx)/(α1*v*dt+dx)
                b2 = (α2*v*dt-dx)/(α2*v*dt+dx)  
                E_z[x,-1,z,0]  = (
                +(b1+b2)*(E_z[x,-2,z,0]-E_z[x,-1,z,1]) 
                - b1*b2*(E_z[x,-3,z,0]-2*E_z[x,-2,z,1]+E_z[x,-1,z,2])
                -(b1*(1-d2)+b2*(1-d1))*(E_z[x,-3,z,1]-E_z[x,-2,z,2])
                +((1-d1)+(1-d2))*E_z[x,-2,z,1]
                -(1-d1)*(1-d2)*E_z[x,-3,z,2]
                )  

        
        for x in range(nx):
            # E_x[x,0,0,0] = 0
            v = 1/np.sqrt(eps[x,0,0]*mu[x,0,0])
            b1 = (α1*v*dt-dx)/(α1*v*dt+dx)
            b2 = (α2*v*dt-dx)/(α2*v*dt+dx)
            E_x[x,0,0,0]  = (1/2)*(
                (
                +(b1+b2)*(E_x[x,0,1,0]-E_x[x,0,0,1]) 
                - b1*b2*(E_x[x,0,2,0]-2*E_x[x,0,1,1]+E_x[x,0,0,2])
                -(b1*(1-d2)+b2*(1-d1))*(E_x[x,0,2,1]-E_x[x,0,1,2])
                +((1-d1)+(1-d2))*E_x[x,0,1,1]
                -(1-d1)*(1-d2)*E_x[x,0,2,2]
                )
                +
                (
                +(b1+b2)*(E_x[x,1,0,0]-E_x[x,0,0,1]) 
                - b1*b2*(E_x[x,2,0,0]-2*E_x[x,1,0,1]+E_x[x,0,0,2])
                -(b1*(1-d2)+b2*(1-d1))*(E_x[x,2,0,1]-E_x[x,1,0,2])
                +((1-d1)+(1-d2))*E_x[x,1,0,1]
                -(1-d1)*(1-d2)*E_x[x,2,0,2]
                )
            )
            # E_x[x,-1,0,0] = 0
            v = 1/np.sqrt(eps[x,-1,0]*mu[x,-1,0])
            b1 = (α1*v*dt-dx)/(α1*v*dt+dx)
            b2 = (α2*v*dt-dx)/(α2*v*dt+dx)
            E_x[x,-1,0,0] = (1/2)*(
                (
                +(b1+b2)*(E_x[x,-1,1,0]-E_x[x,-1,0,1]) 
                - b1*b2*(E_x[x,-1,2,0]-2*E_x[x,-1,1,1]+E_x[x,-1,0,2])
                -(b1*(1-d2)+b2*(1-d1))*(E_x[x,-1,2,1]-E_x[x,-1,1,2])
                +((1-d1)+(1-d2))*E_x[x,-1,1,1]
                -(1-d1)*(1-d2)*E_x[x,-1,2,2]
                )
                +
                (
                +(b1+b2)*(E_x[x,-2,0,0]-E_x[x,-1,0,1]) 
                - b1*b2*(E_x[x,-3,0,0]-2*E_x[x,-2,0,1]+E_x[x,-1,0,2])
                -(b1*(1-d2)+b2*(1-d1))*(E_x[x,-3,0,1]-E_x[x,-2,0,2])
                +((1-d1)+(1-d2))*E_x[x,-2,0,1]
                -(1-d1)*(1-d2)*E_x[x,-3,0,2]
                )  
            )
            # E_x[x,0,-1,0] = 0
            v = 1/np.sqrt(eps[x,0,-1]*mu[x,0,-1])
            b1 = (α1*v*dt-dx)/(α1*v*dt+dx)
            b2 = (α2*v*dt-dx)/(α2*v*dt+dx)
            E_x[x,0,-1,0] = (1/2)*(
                (
                +(b1+b2)*(E_x[x,1,-1,0]-E_x[x,0,-1,1]) 
                - b1*b2*(E_x[x,2,-1,0]-2*E_x[x,1,-1,1]+E_x[x,0,-1,2])
                -(b1*(1-d2)+b2*(1-d1))*(E_x[x,2,-1,1]-E_x[x,1,-1,2])
                +((1-d1)+(1-d2))*E_x[x,1,-1,1]
                -(1-d1)*(1-d2)*E_x[x,2,-1,2]
                )
                +
                (
                +(b1+b2)*(E_x[x,0,-2,0]-E_x[x,0,-1,1]) 
                - b1*b2*(E_x[x,0,-3,0]-2*E_x[x,0,-2,1]+E_x[x,0,-1,2])
                -(b1*(1-d2)+b2*(1-d1))*(E_x[x,0,-3,1]-E_x[x,0,-2,2])
                +((1-d1)+(1-d2))*E_x[x,0,-2,1]
                -(1-d1)*(1-d2)*E_x[x,0,-3,2]
                )  
            )

            # E_x[x,ny-1,nz-1,0] = 0
            v = 1/np.sqrt(eps[x,-1,-1]*mu[x,-1,-1])
            b1 = (α1*v*dt-dx)/(α1*v*dt+dx)
            b2 = (α2*v*dt-dx)/(α2*v*dt+dx)
            E_x[x,-1,-1,0] =  (1/2)*(
                (
                +(b1+b2)*(E_x[x,-2,-1,0]-E_x[x,-1,-1,1]) 
                - b1*b2*(E_x[x,-3,-1,0]-2*E_x[x,-2,-1,1]+E_x[x,-1,-1,2])
                -(b1*(1-d2)+b2*(1-d1))*(E_x[x,-3,-1,1]-E_x[x,-2,-1,2])
                +((1-d1)+(1-d2))*E_x[x,-2,-1,1]
                -(1-d1)*(1-d2)*E_x[x,-3,-1,2]
                )
                +
                (
                +(b1+b2)*(E_x[x,-1,-2,0]-E_x[x,-1,-1,1]) 
                - b1*b2*(E_x[x,-1,-3,0]-2*E_x[x,-1,-2,1]+E_x[x,-1,-1,2])
                -(b1*(1-d2)+b2*(1-d1))*(E_x[x,-1,-3,1]-E_x[x,-1,-2,2])
                +((1-d1)+(1-d2))*E_x[x,-1,-2,1]
                -(1-d1)*(1-d2)*E_x[x,-1,-3,2]
                )  
            )

        #境界面のEy=0                     
        for y in range(ny):
            # E_y[0,y,0,0] = 0
            v = 1/np.sqrt(eps[0,y,0]*mu[0,y,0])
            b1 = (α1*v*dt-dx)/(α1*v*dt+dx)
            b2 = (α2*v*dt-dx)/(α2*v*dt+dx)
            E_y[0,y,0,0]  = (1/2)*(
                (
                +(b1+b2)*(E_y[0,y,1,0]-E_y[0,y,0,1]) 
                - b1*b2*(E_y[0,y,2,0]-2*E_y[0,y,1,1]+E_y[0,y,0,2])
                -(b1*(1-d2)+b2*(1-d1))*(E_y[0,y,2,1]-E_y[0,y,1,2])
                +((1-d1)+(1-d2))*E_y[0,y,1,1]
                -(1-d1)*(1-d2)*E_y[0,y,2,2]
                )
                +
                (
                +(b1+b2)*(E_y[1,y,0,0]-E_y[0,y,0,1]) 
                - b1*b2*(E_y[2,y,0,0]-2*E_y[1,y,0,1]+E_y[0,y,0,2])
                -(b1*(1-d2)+b2*(1-d1))*(E_y[2,y,0,1]-E_y[1,y,0,2])
                +((1-d1)+(1-d2))*E_y[1,y,0,1]
                -(1-d1)*(1-d2)*E_y[2,y,0,2]
                )
            )
            # E_y[nx-1,y,0,0] = 0
            v = 1/np.sqrt(eps[-1,y,0]*mu[-1,y,0])
            b1 = (α1*v*dt-dx)/(α1*v*dt+dx)
            b2 = (α2*v*dt-dx)/(α2*v*dt+dx)
            E_y[-1,y,0,0]  = (1/2)*(
                (
                +(b1+b2)*(E_y[-1,y,1,0]-E_y[-1,y,0,1]) 
                - b1*b2*(E_y[-1,y,2,0]-2*E_y[-1,y,1,1]+E_y[-1,y,0,2])
                -(b1*(1-d2)+b2*(1-d1))*(E_y[-1,y,2,1]-E_y[-1,y,1,2])
                +((1-d1)+(1-d2))*E_y[-1,y,1,1]
                -(1-d1)*(1-d2)*E_y[-1,y,2,2]
                )
                +
                (
                +(b1+b2)*(E_y[-2,y,0,0]-E_y[-1,y,0,1]) 
                - b1*b2*(E_y[-3,y,0,0]-2*E_y[-2,y,0,1]+E_y[-1,y,0,2])
                -(b1*(1-d2)+b2*(1-d1))*(E_y[-3,y,0,1]-E_y[-2,y,0,2])
                +((1-d1)+(1-d2))*E_y[-2,y,0,1]
                -(1-d1)*(1-d2)*E_y[-3,y,0,2]
                )
            )

            # E_y[0,y,nz-1,0] = 0
            v = 1/np.sqrt(eps[0,y,-1]*mu[0,y,-1])
            b1 = (α1*v*dt-dx)/(α1*v*dt+dx)
            b2 = (α2*v*dt-dx)/(α2*v*dt+dx)
            E_y[0,y,-1,0]  = (1/2)*(
                (
                +(b1+b2)*(E_y[0,y,-2,0]-E_y[0,y,-1,1]) 
                - b1*b2*(E_y[0,y,-3,0]-2*E_y[0,y,-2,1]+E_y[0,y,-1,2])
                -(b1*(1-d2)+b2*(1-d1))*(E_y[0,y,-3,1]-E_y[0,y,-2,2])
                +((1-d1)+(1-d2))*E_y[0,y,-2,1]
                -(1-d1)*(1-d2)*E_y[0,y,-3,2]
                )
                +
                (
                +(b1+b2)*(E_y[1,y,-1,0]-E_y[0,y,-1,1]) 
                - b1*b2*(E_y[2,y,-1,0]-2*E_y[1,y,-1,1]+E_y[0,y,-1,2])
                -(b1*(1-d2)+b2*(1-d1))*(E_y[2,y,-1,1]-E_y[1,y,-1,2])
                +((1-d1)+(1-d2))*E_y[1,y,-1,1]
                -(1-d1)*(1-d2)*E_y[2,y,-1,2]
                )
            )

            # E_y[-1,y,-1,0] = 0
            v = 1/np.sqrt(eps[0,y,-1]*mu[0,y,-1])
            b1 = (α1*v*dt-dx)/(α1*v*dt+dx)
            b2 = (α2*v*dt-dx)/(α2*v*dt+dx)
            E_y[-1,y,-1,0]  = (1/2)*(
                (
                +(b1+b2)*(E_y[0,y,-2,0]-E_y[0,y,-1,1]) 
                - b1*b2*(E_y[0,y,-3,0]-2*E_y[0,y,-2,1]+E_y[0,y,-1,2])
                -(b1*(1-d2)+b2*(1-d1))*(E_y[0,y,-3,1]-E_y[0,y,-2,2])
                +((1-d1)+(1-d2))*E_y[0,y,-2,1]
                -(1-d1)*(1-d2)*E_y[0,y,-3,2]
                )
                +
                (
                +(b1+b2)*(E_y[-2,y,-1,0]-E_y[-1,y,-1,1]) 
                - b1*b2*(E_y[-3,y,-1,0]-2*E_y[-2,y,-1,1]+E_y[-1,y,-1,2])
                -(b1*(1-d2)+b2*(1-d1))*(E_y[-3,y,-1,1]-E_y[-2,y,-1,2])
                +((1-d1)+(1-d2))*E_y[-2,y,-1,1]
                -(1-d1)*(1-d2)*E_y[-3,y,-1,2]
                )
            )

        #境界面のEz=0
        for z in range(nz):
            # E_z[0,0,z,0] = 0
            v = 1/np.sqrt(eps[0,0,z]*mu[0,0,z])
            b1 = (α1*v*dt-dx)/(α1*v*dt+dx)
            b2 = (α2*v*dt-dx)/(α2*v*dt+dx)
            E_z[0,0,z,0]  = (1/2)*(
                (
                +(b1+b2)*(E_z[0,1,z,0]-E_z[0,0,z,1]) 
                - b1*b2*(E_z[0,2,z,0]-2*E_z[0,1,z,1]+E_z[0,0,z,2])
                -(b1*(1-d2)+b2*(1-d1))*(E_z[0,2,z,1]-E_z[0,1,z,2])
                +((1-d1)+(1-d2))*E_z[0,1,z,1]
                -(1-d1)*(1-d2)*E_z[0,2,z,2]
                )
                +
                (
                +(b1+b2)*(E_z[1,0,z,0]-E_z[0,0,z,1]) 
                - b1*b2*(E_z[2,0,z,0]-2*E_z[1,0,z,1]+E_z[0,0,z,2])
                -(b1*(1-d2)+b2*(1-d1))*(E_z[2,0,z,1]-E_z[1,0,z,2])
                +((1-d1)+(1-d2))*E_z[1,0,z,1]
                -(1-d1)*(1-d2)*E_z[2,0,z,2]
                )
            )

            # E_z[nx-1,0,z,0] = 0
            v = 1/np.sqrt(eps[-1,0,z]*mu[-1,0,z])
            b1 = (α1*v*dt-dx)/(α1*v*dt+dx)
            b2 = (α2*v*dt-dx)/(α2*v*dt+dx)
            E_z[-1,0,z,0]  = (1/2)*(
                (
                +(b1+b2)*(E_z[-1,1,z,0]-E_z[-1,0,z,1]) 
                - b1*b2*(E_z[-1,2,z,0]-2*E_z[-1,1,z,1]+E_z[-1,0,z,2])
                -(b1*(1-d2)+b2*(1-d1))*(E_z[-1,2,z,1]-E_z[-1,1,z,2])
                +((1-d1)+(1-d2))*E_z[-1,1,z,1]
                -(1-d1)*(1-d2)*E_z[-1,2,z,2]
                )
                +
                (
                +(b1+b2)*(E_z[-2,0,z,0]-E_z[-1,0,z,1]) 
                - b1*b2*(E_z[-3,0,z,0]-2*E_z[-2,0,z,1]+E_z[-1,0,z,2])
                -(b1*(1-d2)+b2*(1-d1))*(E_z[-3,0,z,1]-E_z[-2,0,z,2])
                +((1-d1)+(1-d2))*E_z[-2,0,z,1]
                -(1-d1)*(1-d2)*E_z[-3,0,z,2]
                )
            )

            # E_z[0,ny-1,z,0] = 0
            v = 1/np.sqrt(eps[0,-1,z]*mu[0,-1,z])
            b1 = (α1*v*dt-dx)/(α1*v*dt+dx)
            b2 = (α2*v*dt-dx)/(α2*v*dt+dx)
            E_z[0,-1,z,0]  = (1/2)*(
                (
                +(b1+b2)*(E_z[0,-2,z,0]-E_z[0,-1,z,1]) 
                - b1*b2*(E_z[0,2,z,0]-2*E_z[0,1,z,1]+E_z[0,-1,z,2])
                -(b1*(1-d2)+b2*(1-d1))*(E_z[0,-3,z,1]-E_z[0,-2,z,2])
                +((1-d1)+(1-d2))*E_z[0,-2,z,1]
                -(1-d1)*(1-d2)*E_z[0,-3,z,2]
                )
                +
                (
                +(b1+b2)*(E_z[1,0,z,0]-E_z[0,0,z,1]) 
                - b1*b2*(E_z[2,0,z,0]-2*E_z[1,0,z,1]+E_z[0,0,z,2])
                -(b1*(1-d2)+b2*(1-d1))*(E_z[2,0,z,1]-E_z[1,0,z,2])
                +((1-d1)+(1-d2))*E_z[1,0,z,1]
                -(1-d1)*(1-d2)*E_z[2,0,z,2]
                )
            )

            # E_z[nx-1,ny-1,z,0] = 0 
            v = 1/np.sqrt(eps[-1,-1,z]*mu[-1,-1,z])
            b1 = (α1*v*dt-dx)/(α1*v*dt+dx)
            b2 = (α2*v*dt-dx)/(α2*v*dt+dx)
            E_z[-1,-1,z,0]  = (1/2)*(
                (
                +(b1+b2)*(E_z[-1,-2,z,0]-E_z[-1,-1,z,1]) 
                - b1*b2*(E_z[-1,2,z,0]-2*E_z[-1,1,z,1]+E_z[-1,-1,z,2])
                -(b1*(1-d2)+b2*(1-d1))*(E_z[-1,-3,z,1]-E_z[-1,-2,z,2])
                +((1-d1)+(1-d2))*E_z[-1,-2,z,1]
                -(1-d1)*(1-d2)*E_z[-1,-3,z,2]
                )
                +
                (
                +(b1+b2)*(E_z[-2,-1,z,0]-E_z[-1,-1,z,1]) 
                - b1*b2*(E_z[-3,-1,z,0]-2*E_z[-2,-1,z,1]+E_z[-1,-1,z,2])
                -(b1*(1-d2)+b2*(1-d1))*(E_z[-3,-1,z,1]-E_z[-2,-1,z,2])
                +((1-d1)+(1-d2))*E_z[-2,-1,z,1]
                -(1-d1)*(1-d2)*E_z[-3,-1,z,2]
                )
            )
        # E_x_save[:,:,:,t] = E_x[:,:,:,0]
        if t%step_save == 0:
            t2 = int(t/step_save)
            Jx_yz[:,:,t2]   = J_x[100,:,:,0]
            Hx_yz[:,:,t2]   = H_x[100,:,:,0]
            Hy_yz[:,:,t2]   = H_y[100,:,:,0]
            Hz_yz[:,:,t2]   = H_z[100,:,:,0]
            Ex_yz[:,:,t2]   = E_x[100,:,:,0]
            Ey_yz[:,:,t2]   = E_y[100,:,:,0]
            Ez_yz[:,:,t2]   = E_z[100,:,:,0]
            Jx_xz[:,:,t2]   = J_x[:,30,:,0]
            Hx_xz[:,:,t2]   = H_x[:,30,:,0]
            Hy_xz[:,:,t2]   = H_y[:,30,:,0]
            Hz_xz[:,:,t2]   = H_z[:,30,:,0]
            Ex_xz[:,:,t2]   = E_x[:,30,:,0]
            Ey_xz[:,:,t2]   = E_y[:,30,:,0]
            Ez_xz[:,:,t2]   = E_z[:,30,:,0]
        E_x[:,:,:,2] = E_x[:,:,:,1]
        E_x[:,:,:,1] = E_x[:,:,:,0]
        E_x[:,:,:,0] = 0*E_x[:,:,:,0]
        E_y[:,:,:,2] = E_y[:,:,:,1]
        E_y[:,:,:,1] = E_y[:,:,:,0]
        E_y[:,:,:,0] = 0*E_y[:,:,:,0]
        E_z[:,:,:,2] = E_z[:,:,:,1]
        E_z[:,:,:,1] = E_z[:,:,:,0]
        E_z[:,:,:,0] = 0*E_z[:,:,:,0]
        H_x[:,:,:,2] = H_x[:,:,:,1]
        H_x[:,:,:,1] = H_x[:,:,:,0]
        H_x[:,:,:,0] = 0*H_x[:,:,:,0]
        H_y[:,:,:,2] = H_y[:,:,:,1]
        H_y[:,:,:,1] = H_y[:,:,:,0]
        H_y[:,:,:,0] = 0*H_y[:,:,:,0]
        H_z[:,:,:,2] = H_z[:,:,:,1]
        H_z[:,:,:,1] = H_z[:,:,:,0]
        H_z[:,:,:,0] = 0*H_z[:,:,:,0]
        J_x[:,:,:,2] = J_x[:,:,:,1]
        J_x[:,:,:,1] = J_x[:,:,:,0]
        J_x[:,:,:,0] = 0*J_x[:,:,:,0]
        J_y[:,:,:,2] = J_y[:,:,:,1]
        J_y[:,:,:,1] = J_y[:,:,:,0]
        J_y[:,:,:,0] = 0*J_y[:,:,:,0]
        J_z[:,:,:,2] = J_z[:,:,:,1]
        J_z[:,:,:,1] = J_z[:,:,:,0]
        J_z[:,:,:,0] = 0*J_z[:,:,:,0]
    return Jx_yz,Hx_yz,Hy_yz,Hz_yz,Ex_yz,Ey_yz,Ez_yz,Jx_xz,Hx_xz,Hy_xz,Hz_xz,Ex_xz,Ey_xz,Ez_xz
        


    
    
  

