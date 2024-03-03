from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
def split_dataset(x,y,dataset_name,train_ratio=0.1):
    data_remove_zero = x[y>0,:,:,:]
    label_remove_zero = y[y>0]
    label_remove_zero -= 1
    test_ratio = 1 - 2*train_ratio
    x_train,x_test,y_train,y_test = train_test_split(data_remove_zero,label_remove_zero,
                                                     stratify=label_remove_zero,
                                                     test_size=test_ratio)
    x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,
                                                 stratify=y_train,
                                                 test_size=0.5)

    classes = int(np.max(y_train))+1
    total_w = len(y_train)/classes
    class_map = Counter(y_train)                            
    sampler = [total_w/class_map[i] for i in range(classes)]

    return x_train,y_train,x_val,y_val,x_test,y_test,sampler
