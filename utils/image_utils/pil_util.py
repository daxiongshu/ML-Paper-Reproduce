from PIL import Image
import numpy as np

def read_one_mask(imgpath,size=None):
    """
        Input:
             imgpath: image path
             size: tuple (w,h) to resize
        Return:
             a numpy array [h,w]
    """
    img = Image.open(imgpath)
    if size is not None:
        img = img.resize(size)
    return np.array(img)

def ratio(a):
    assert np.unique(a).tolist() == [0,1]
    return np.sum(a)*1.0/(a.shape[0]*a.shape[1])

if __name__ == "__main__":
    a = read_one_mask("../input/train_masks/0cdf5b5d0ce1_01_mask.gif",(512,512))
    print(a.shape,ratio(a))

