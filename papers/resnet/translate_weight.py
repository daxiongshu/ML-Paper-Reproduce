import numpy as np
import tensorflow as tf
from tensorflow.contrib.keras.api.keras.applications import resnet50
import re
import keras


def resnet50_name_map(name):
    tokens = re.findall('[0-9][a-g]', name)
    letters = {i:c+1 for c,i in enumerate('abcdefghijklmn')}
    if len(tokens)==2:
        s1,s2 = tokens # res4c_branch2a => 4c, 2a
        module = int(s1[0])-1
        block = letters[s1[1]]
        layer = letters[s2[1]] 
        if name.startswith('res'):
            result = 'resnet50/module%d/block%d/conv%d'%(module,block,layer)
        elif name.startswith('bn'):
            result = 'resnet50/module%d/block%d/batch_norm%d/batch_normalization'%(module,block,layer)
    elif len(tokens)==1 and name.endswith('branch1'):
        s1 = tokens[0]
        module = int(s1[0])-1
        block = letters[s1[1]]
        if name.startswith('res'):
            result = 'resnet50/module%d/shortcut%d/conv'%(module,block)
        elif name.startswith('bn'):
            result = 'resnet50/module%d/shortcut%d/batch_norm/batch_normalization'%(module,block)
    else:
        #print(name)
        return 0
    return result

def translate_resnet50_from_keras_app():
    print("translate resnet50's pretrained weights to tf format")
    model = resnet50.ResNet50()
    newweight = {}
    for layer in model.layers:
        if layer.name == 'conv1':
            w,b = layer.get_weights()
            newweight['resnet50/conv1/weights:0'] = w
            newweight['resnet50/conv1/bias:0'] = b
        elif layer.name == 'bn_conv1':
            #beta,gamma,mean,variance = layer.get_weights()
            gamma,beta,mean,variance = layer.get_weights()
            name = 'resnet50/batch_norm1/batch_normalization'
            newweight['%s/beta:0'%name] = beta
            newweight['%s/gamma:0'%name] = gamma
            newweight['%s/moving_mean:0'%name] = mean
            newweight['%s/moving_variance:0'%name] = variance
        elif layer.name == 'fc1000':
            w,b = layer.get_weights()
            newweight['resnet50/fc/weights:0'] = w
            newweight['resnet50/fc/bias:0'] = b
        elif layer.name.startswith('bn'):
            #beta,gamma,mean,variance = layer.get_weights()
            gamma,beta,mean,variance = layer.get_weights()
            name = resnet50_name_map(layer.name)
            print(layer.name, name)
            newweight['%s/beta:0'%name] = beta
            newweight['%s/gamma:0'%name] = gamma
            newweight['%s/moving_mean:0'%name] = mean
            newweight['%s/moving_variance:0'%name] = variance
        elif layer.name.startswith('res'):
            w,b = layer.get_weights()
            name = resnet50_name_map(layer.name)
            print(layer.name, name)
            newweight['%s/weights:0'%name] = w
            newweight['%s/bias:0'%name] = b
        elif len(layer.get_weights())>0:
            print(layer.name, len(layer.get_weights()))
    print (len(newweight),len([i for i in model.layers if len(i.get_weights())>0]))
    np.save("pretrain_weights/resnet50.npy",newweight)

        


if __name__ == "__main__":
    translate_resnet50_from_keras_app()
