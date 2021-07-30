import numpy as np
import pandas as pd

def MakeIXPosition(PC):
    [nx,ny,nz] = PC.shape
    PIX = np.zeros([nx,ny+1,nz+1])
    zero_zx  = np.zeros([nx,1,nz])
    temp1_y = np.concatenate((zero_zx,PC),axis=1)
    temp2_y = np.concatenate((PC,zero_zx),axis=1)
    temp    = temp1_y + temp2_y
    zero_xy  = np.zeros([nx,ny+1,1])
    temp1_z = np.concatenate((zero_xy,temp),axis=2)         
    temp2_z = np.concatenate((temp,zero_xy),axis=2)         
    temp    = temp1_z+temp2_z
    for i in range(nz+1):
        PIX[:,:,i] = np.array(pd.DataFrame(temp[:,:,i]).replace([1.0,2.0,3.0,4.0],1.0))
    return PIX

def MakeIYPosition(PC):
    [nx,ny,nz] = PC.shape
    PIY = np.zeros([nx+1,ny,nz+1])
    zero_yz  = np.zeros([1,ny,nz])
    temp1_x = np.concatenate((zero_yz,PC),axis=0)
    temp2_x = np.concatenate((PC,zero_yz),axis=0)
    temp    = temp1_x + temp2_x
    zero_xy  = np.zeros([nx+1,ny,1])
    temp1_z = np.concatenate((zero_xy,temp),axis=2)       
    temp2_z = np.concatenate((temp,zero_xy),axis=2)       
    temp    = temp1_z+temp2_z
    for i in range(nz+1):
        PIY[:,:,i] = np.array(pd.DataFrame(temp[:,:,i]).replace([1.0,2.0,3.0,4.0],1.0))
    return PIY



def MakeIZPosition(PC):
    [nx,ny,nz] = PC.shape
    PIZ = np.zeros([nx+1,ny+1,nz])
    zero_yz = np.zeros([1,ny,nz])
    temp1_x = np.concatenate((zero_yz,PC),axis=0)
    temp2_x = np.concatenate((PC,zero_yz),axis=0)
    temp    = temp1_x + temp2_x
    zero_zx = np.zeros([nx+1,1,nz])
    temp1_y = np.concatenate((zero_zx,temp),axis=1)       
    temp2_y = np.concatenate((temp,zero_zx),axis=1)       
    temp    = temp1_y+temp2_y
    for i in range(nz):
        PIZ[:,:,i] = np.array(pd.DataFrame(temp[:,:,i]).replace([1.0,2.0,3.0,4.0],1.0))
    return PIZ