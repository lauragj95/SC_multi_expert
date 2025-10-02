import numpy as np



def make_weights_for_balanced_classes(images, nclasses,argmax=False):              
    count = [0] * nclasses                                                      
    for img in images:
        if argmax:
            img = np.argmax(img,axis=2)

        labels,counts = np.unique(img,return_counts=True)
        if len(labels)==1 and labels[0]==0:
            pass
        else:
            for i,label in enumerate(labels):
                print(int(label))
                count[int(label)] += counts[i]
                
                                                             
    weight_per_class = [0.] * nclasses     
    weights_class = [0.] * nclasses                                
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                
        weight_per_class[i] = 1 - float(count[i])/N   
        weights_class[i] =  N/float(count[i])    
    return weight_per_class,weights_class     
