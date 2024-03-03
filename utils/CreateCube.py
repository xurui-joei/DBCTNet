import numpy as np
def pad_data(x,patch_size=9):
    pad = patch_size // 2
    pad_data = np.zeros((x.shape[0]+2*pad,x.shape[1]+2*pad,x.shape[2]))
    pad_data[pad:x.shape[0]+pad,pad:x.shape[1]+pad,:] = x
    return pad_data

def create_patch(x,y,patch_size=9):
    pad = patch_size // 2
    h = x.shape[0]
    w = x.shape[1]
    b = x.shape[2]
    pad_x = pad_data(x,patch_size)
    print('finish pad data')
    data_cube = np.zeros((h*w,patch_size,patch_size,b))
    data_label = np.zeros(h*w)
    patch_id = 0
    for i in range(pad,h+pad):
        for j in range(pad,w+pad):
            data_cube[patch_id,:,:,:] = pad_x[i-pad:i+pad+1,j-pad:j+pad+1,:]
            data_label[patch_id] = y[i-pad,j-pad]
            patch_id += 1
    data_cube = np.expand_dims(data_cube,axis=1)
    return data_cube,data_label
