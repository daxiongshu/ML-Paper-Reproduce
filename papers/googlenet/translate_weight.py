from tensorflow.contrib.keras.api.keras.applications import inception_v3
import numpy as np 

def inception_v3_name_map(name):
    name = int(name)+94
    if name in [95,96,97]:
        return "inception_v3/block1/conv%d"%(name-95)
    if name in [98,99]:
        return "inception_v3/block2/conv%d"%(name-98)

    # inception 0,1,2
    if name in [100,107,114]:
        block = (name - 100)//7
        return "inception_v3/inception%d/1x1/conv0"%(block)
    if name in [101,102,108,109,115,116]:
        block = (name - 101)//7
        count = (name-101)%7
        return "inception_v3/inception%d/5x5/conv%d"%(block,count)
    if name in [103,104,105,110,111,112,117,118,119]:
        block = (name - 103)//7
        count = (name-103)%7
        return "inception_v3/inception%d/3x3/conv%d"%(block,count)
    if name in [106,113,120]:
        block = (name - 106)//7
        return "inception_v3/inception%d/avg_1x1/conv0"%(block)

    # inception 3
    if name in [121]:
        block = 3
        return "inception_v3/inception%d/3x3/conv0"%(block)
    if name in [122,123,124]:
        block = 3
        count = name-122
        return "inception_v3/inception%d/3x3d/conv%d"%(block,count)

    # inception 4
    if name in [125]:
        block = 4
        return "inception_v3/inception%d/1x1/conv0"%(block)
    if name in [126,127,128]:
        block = 4
        count = name-126
        return "inception_v3/inception%d/7x7/conv%d"%(block,count)
    if name in range(129,129+5):
        block = 4
        count = name-129
        return "inception_v3/inception%d/7x7_db/conv%d"%(block,count)
    if name in [134]:
        block = 4
        return "inception_v3/inception%d/avg_1x1/conv0"%(block)

    #inception 5 and 6
    if name in [135,145]:
        block = (name - 135)//10 + 5
        return "inception_v3/inception%d/1x1/conv0"%(block)
    if name in [136,137,138,146,147,148]:
        block = (name - 136)//10 + 5
        count = (name-136)%10
        return "inception_v3/inception%d/7x7/conv%d"%(block,count)
    if name in list(range(139,139+5))+list(range(149,149+5)):
        block = (name - 139)//10 + 5
        count = (name-139)%10
        return "inception_v3/inception%d/7x7_db/conv%d"%(block,count)
    if name in [144,154]:
        block = (name - 144)//10 + 5 
        return "inception_v3/inception%d/avg_1x1/conv0"%(block)

    # inception 7
    if name in [155]:
        block = 7
        return "inception_v3/inception%d/1x1/conv0"%(block)
    if name in [156,157,158]:
        block = 7
        count = name-156
        return "inception_v3/inception%d/7x7/conv%d"%(block,count)
    if name in range(159,159+5):
        block = 7
        count = name-159
        return "inception_v3/inception%d/7x7_db/conv%d"%(block,count)
    if name in [164]:
        block = 7
        return "inception_v3/inception%d/avg_1x1/conv0"%(block)

    # inception 8
    if name in [165,166]:
        block = 8
        count = name - 165
        return "inception_v3/inception%d/3x3/conv%d"%(block,count)
    if name in [167,168,169,170]:
        block = 8
        count = name-167
        return "inception_v3/inception%d/7x7/conv%d"%(block,count)

    #inception 9 and 10
    x = 171
    if name in [x,x+9]:
        block = (name - x)//9 + 9
        return "inception_v3/inception%d/1x1/conv0"%(block)
    if name in [x+1,x+10]:
        block = (name - x)//9 + 9
        return "inception_v3/inception%d/3x3/conv0"%(block)
    if name in [x+2,x+11]:
        block = (name - x)//9 + 9
        return "inception_v3/inception%d/3x3_1/conv0"%(block)
    if name in [x+3,x+12]:
        block = (name - x)//9 + 9
        return "inception_v3/inception%d/3x3_2/conv0"%(block)
        
    if name in [x+4,x+5,x+13,x+14]:
        block = (name - x)//9 + 9
        count = (name- x -4)%9
        return "inception_v3/inception%d/3x3_db/conv%d"%(block,count)
    if name in [x+6,x+15]:
        block = (name - x)//9 + 9
        return "inception_v3/inception%d/3x3_db1/conv0"%(block)
    if name in [x+7,x+16]:
        block = (name - x)//9 + 9
        return "inception_v3/inception%d/3x3_db2/conv0"%(block)
    if name in [x+8,x+17]:
        block = (name - x)//9 + 9
        return "inception_v3/inception%d/avg_1x1/conv0"%(block)
    print("no name found",name)
    assert 0

def translate_inception_v3_from_keras_app():
    print("translate inception_v3's pretrained weights to tf format")
    model = inception_v3.InceptionV3()
    newweight = {}
    for layer in model.layers:
        print(layer.name)
        #continue
        if layer.name.startswith('conv2d'):
            name = layer.name.split('_')[1]
            name = inception_v3_name_map(name)
            w = layer.get_weights()[0]
            b = np.zeros(w.shape[-1])
            newweight['%s/weights:0'%name] = w
            newweight['%s/bias:0'%name] = b
            print(layer.name,name)
        elif layer.name.startswith('batch_norm'):
            name = layer.name.split('_')[-1]
            name = inception_v3_name_map(name).replace("conv","batch_norm")+'/batch_normalization'
            beta,mean,variance = layer.get_weights()
            newweight['%s/beta:0'%name] = beta
            newweight['%s/gamma:0'%name] = 1
            newweight['%s/moving_mean:0'%name] = mean
            newweight['%s/moving_variance:0'%name] = variance  
            print(layer.name,name)          
        elif layer.name == 'predictions':
            w,b = layer.get_weights()
            newweight['inception_v3/fc/weights:0'] = w
            newweight['inception_v3/fc/bias:0'] = b
            print(layer.name,name)
        else:
            assert len(layer.get_weights()) == 0
    np.save("pretrain_weights/inception_v3.npy",newweight)

