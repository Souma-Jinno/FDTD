# from numba.np.ufunc import parallel
import numpy as np

from MakeSignal import *
from keisu_calc import keisu_calc
from numba import jit,prange

@jit(nopython=True,parallel=True)
def calc_board(in_board, len_board, wid_board, dis_board, in_current, Signal, dx, dy, dz, dt, nx, ny, nz, nt, eps, mu, sigma):
   
    #---------------------------------------------------------------------------------------------------------------------------------------------------
    #　係数の計算
    #---------------------------------------------------------------------------------------------------------------------------------------------------
    dhx, dhy, dhz, ce, dex, dey, dez, de= keisu_calc(dx, dy, dz, dt, eps, mu, sigma)

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
    J_z = np.zeros(shape=(nx+1, ny+1, nz, nt)) 

    for t in range(nt-1):
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------
        #　入力信号
        #---------------------------------------------------------------------------------------------------------------------------------------------------
        if dis_board % 2 != 0:
            dis_board = dis_board + 1

        J_z[nx//2+1,(ny//2+1)-(dis_board//2),in_current,t+1] = -Signal[t]
        J_z[nx//2+1,(ny//2+1)+(dis_board//2),in_current,t+1] = Signal[t]
       
        #---------------------------------------------------------------------------------------------------------------------------------------------------
        #　電磁界計算
        #---------------------------------------------------------------------------------------------------------------------------------------------------
        #Hx_calc
        for x in prange(nx-1):   
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

        #Ex_calc
        for x in prange(nx):
            for y in range(1,ny):
                for z in range(1,nz):
                    E_x[x,y,z,t+1] = ce[x,y,z] * E_x[x,y,z,t] + dey[x,y,z] * (H_z[x,y,z,t+1] - H_z[x,y-1,z,t+1])\
                                                        - dez[x,y,z] * (H_y[x,y,z,t+1] - H_y[x,y,z-1,t+1])

        #Ey_calc
        for x in prange(1,nx):
            for y in range(ny):
                for z in range(1,nz):   
                    E_y[x,y,z,t+1] = ce[x,y,z] * E_y[x,y,z,t] + dez[x,y,z] * (H_x[x,y,z,t+1] - H_x[x,y,z-1,t+1])\
                                                        - dex[x,y,z] * (H_z[x,y,z,t+1] - H_z[x-1,y,z,t+1])
           
        #Ez_calc
        for x in prange(1,nx):
            for y in range(1,ny):
                for z in range(nz):        
                    E_z[x,y,z,t+1] = ce[x,y,z] * E_z[x,y,z,t] + dex[x,y,z] * (H_y[x,y,z,t+1] - H_y[x-1,y,z,t+1])\
                                                            - dey[x,y,z] * (H_x[x,y,z,t+1] - H_x[x,y-1,z,t+1]) \
                                                            - de[x,y,z] * J_z[x,y,z,t+1]

        #---------------------------------------------------------------------------------------------------------------------------------------------------
        #　導体板における電流の計算(変更点)
        #---------------------------------------------------------------------------------------------------------------------------------------------------

        for x in range((nx//2+1)-wid_board//2, (nx//2+1)+wid_board//2+1):
            for z in range(in_board, in_board+len_board):
                E_z[x,(ny//2+1)+(dis_board//2),z,t+1] = 0
                J_z[x,(ny//2+1)+(dis_board//2),z,t+1] = (1/dx) * (H_y[nx//2+1,(ny//2+1)+dis_board//2,z,t] - H_y[nx//2,(ny//2+1)+dis_board//2,z,t])\
                                                                -(1/dy) * (H_x[nx//2+1,(ny//2+1)+dis_board//2,z,t] - H_x[nx//2+1,(ny//2)+dis_board//2,z,t])
        for x in range((nx//2+1)-wid_board//2, (nx//2+1)+wid_board//2):
            for z in range(in_board, in_board+len_board+1):
                E_x[x,(ny//2+1)+(dis_board//2),z,t+1] = 0
                J_x[x,(ny//2+1)+(dis_board//2),z,t+1] = (1/dy) * (H_z[nx//2+1,(ny//2+1)+dis_board//2,z,t] - H_z[nx//2+1,(ny//2+1)+dis_board//2-1,z,t])\
                                                                -(1/dz) * (H_y[nx//2+1,(ny//2+1)+dis_board//2,z,t] - H_y[nx//2+1,(ny//2)+dis_board//2,z-1,t])
        

        for x in range((nx//2+1)-wid_board//2, (nx//2+1)+wid_board//2+1):
            for z in range(in_board ,in_board+len_board):
                E_z[x,(ny//2+1)-(dis_board//2),z,t+1] = 0
                J_z[x,(ny//2+1)-(dis_board//2),z,t+1] = (1/dx) * (H_y[nx//2+1,(ny//2+1)-dis_board//2,z,t] - H_y[nx//2,(ny//2+1)-dis_board//2,z,t])\
                                                                -(1/dy) * (H_x[nx//2+1,(ny//2+1)-dis_board//2,z,t] - H_x[nx//2+1,(ny//2)-dis_board//2,z,t])
        for x in range((nx//2+1)-wid_board//2, (nx//2+1)+wid_board//2):
            for z in range(in_board, in_board+len_board+1):
                E_x[x,(ny//2+1)-(dis_board//2),z,t+1] = 0
                J_x[x,(ny//2+1)-(dis_board//2),z,t+1] = (1/dy) * (H_z[nx//2+1,(ny//2+1)-dis_board//2,z,t] - H_z[nx//2+1,(ny//2+1)-dis_board//2-1,z,t])\
                                                                -(1/dz) * (H_y[nx//2+1,(ny//2+1)-dis_board//2,z,t] - H_y[nx//2+1,(ny//2)-dis_board//2,z-1,t])

        
        #---------------------------------------------------------------------------------------------------------------------------------------------------
        #　境界条件の設定
        #---------------------------------------------------------------------------------------------------------------------------------------------------
        #境界面のEx=0
        for x in range(nx):
                '''
                #E_x元の式
                E_x[x,y,z,t+1] = ce * E_x[x,y,z,t] + dey * (H_z[x,y,z,t] - H_z[x,y-1,z,t])\
                                                    - dez * (H_y[x,y,z,t] - H_y[x,y,z-1,t])
                '''
                E_x[x,0,0,t+1] = 0
                E_x[x,ny-1,0,t+1] = 0
                E_x[x,0,nz-1,t+1] = 0
                E_x[x,ny-1,nz-1,t+1] = 0
        for x in range(nx):
            for y in range(1,ny-2):
                E_x[x,y,0,t+1] = 0
                E_x[x,y,nz-1,t+1] = 0
        for x in range(nx):
            for z in range(1,nz-2):
                E_x[x,0,z,t+1] = 0
                E_x[x,ny-1,z,t+1] = 0

        #境界面のEy=0                     
        for y in range(ny):
                '''
                E_y[x,y,z,t+1] = ce * E_y[x,y,z,t] + dez * (H_x[x,y,z,t] - H_x[x,y,z-1,t])\
                                                    - dex * (H_z[x,y,z,t] - H_z[x-1,y,z,t])
                '''
                E_y[0,y,0,t+1] = 0
                E_y[nx-1,y,0,t+1] = 0
                E_y[0,y,nz-1,t+1] = 0
                E_y[nx-1,y,nz-1,t+1] = 0
        for y in range(ny):
            for x in range(1,nx-2):
                E_y[x,y,0,t+1] = 0
                E_y[x,y,nz-1,t+1] = 0
        for y in range(ny):
            for z in range(1,nz-2):
                E_y[0,y,z,t+1] = 0
                E_y[nx-1,y,z,t+1] = 0        

        #境界面のEz=0
        for z in range(nz):
                '''
                #E_z元の式
                E_z[x,y,z,t+1] = ce * E_z[x,y,z,t] + dex * (H_y[x,y,z,t] - H_y[x-1,y,z,t])\
                                                    - dey * (H_x[x,y,z,t] - H_x[x,y-1,z,t])
                '''
                E_z[0,0,z,t+1] = 0
                E_z[nx-1,0,z,t+1] = 0
                E_z[0,ny-1,z,t+1] = 0
                E_z[nx-1,ny-1,z,t+1] = 0 
        for z in range(nz):
            for x in range(1,nx-2):
                E_z[x,0,z,t+1] = 0
                E_z[x,ny-1,z,t+1] = 0
        for z in range(nz):
            for y in range(1,ny-2):
                E_z[0,y,z,t+1] = 0
                E_z[nx-1,y,z,t+1] = 0


    return H_x, H_y, H_z, E_x, E_y, E_z, J_x, J_z


    
    
  


