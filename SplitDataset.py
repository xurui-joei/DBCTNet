from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
def split_dataset(x,y,dataset_name,train_ratio=0.1,seed=42):
    data_remove_zero = x[y>0,:,:,:]
    label_remove_zero = y[y>0]
    label_remove_zero -= 1
    
    if dataset_name != 'IP' or (dataset_name == 'IP' and train_ratio > 0.1):
        #test_ratio = 1 - 3*train_ratio
        test_ratio = 1 - 2*train_ratio
        x_train,x_test,y_train,y_test = train_test_split(data_remove_zero,label_remove_zero,
                                                     random_state=seed,
                                                     stratify=label_remove_zero,
                                                     test_size=test_ratio)
        x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,
                                                 random_state=seed,
                                                 stratify=y_train,
                                                 test_size=0.5)
                                                 #test_size=2/3)
        classes = int(np.max(y_train))+1
        total_w = len(y_train)/classes
        class_map = Counter(y_train)                            
        sampler = [total_w/class_map[i] for i in range(classes)]
    else:
        np.random.seed(seed)
        x_train,y_train,x_val,y_val,x_test,y_test = 0,0,0,0,0,0
        if train_ratio == 0.05:
            #train_num = [5,71,42,12,24,36,5,24,5,49,109,30,10,63,19,5]
            #val_num =  [10,142,84,24,48,72,10,48,5,98,246,60,20,126,38,10]
            #test_num = [31,1215,704,201,411,622,13,406,10,825,2100,503,175,1076,329,78]
            train_num = [5,71,42,12,24,36,5,24,5,49,109,30,10,63,19,5]
            val_num =  [5,71,42,12,24,36,5,24,5,49,109,30,10,63,19,5]
            test_num = [36,1286,746,213,435,658,18,430,10,874,2237,533,185,1139,348,83]
        elif train_ratio == 0.1:
            #train_num = [5,143,83,24,48,73,6,48,4,97,246,59,21,127,39,9]
            #val_num = [10,143,83,24,48,73,6,48,4,97,246,59,21,127,39,18]
            #test_num = [31,1142,664,189,387,584,16,382,12,778,1964,474,163,1012,308,66]
            train_num = [5,143,83,24,48,73,6,48,4,97,246,59,21,127,39,9]
            val_num = [10,143,83,24,48,73,6,48,4,97,246,59,21,127,39,18]
            test_num = [31,1142,664,189,387,584,16,382,12,778,1964,474,163,1012,308,66]
        count = 0
        for i in range(16):
            x_c = data_remove_zero[label_remove_zero == i,:,:,:]
            l_c = label_remove_zero[label_remove_zero == i]
            index = np.random.permutation(x_c.shape[0])
            x_c = x_c[index,:,:,:]
            if count == 0:
                x_train,y_train= x_c[0:train_num[i],:,:,:],l_c[0:train_num[i]]
                x_val,y_val = x_c[train_num[i]:train_num[i]+val_num[i],:,:,:],l_c[0:val_num[i]]
                x_test,y_test = x_c[train_num[i]+val_num[i]:,:,:,:],l_c[0:test_num[i]]
                count = 1
            else:
                x_train = np.concatenate((x_train,x_c[0:train_num[i],:,:,:]))
                y_train = np.concatenate((y_train,l_c[0:train_num[i]]))
                x_val = np.concatenate((x_val,x_c[train_num[i]:train_num[i]+val_num[i],:,:,:]))
                y_val = np.concatenate((y_val,l_c[0:val_num[i]]))
                x_test = np.concatenate((x_test,x_c[train_num[i]+val_num[i]:,:,:,:]))
                y_test = np.concatenate((y_test,l_c[0:test_num[i]]))
        classes = int(np.max(y_train))+1
        total_w = len(y_train)
        class_map = Counter(y_train)
        #sampler = [total_train/i for i in map(lambda x:class_map[x],y_train)]
        sampler = [total_w/class_map[i] for i in range(classes)]
    return x_train,y_train,x_val,y_val,x_test,y_test,sampler
