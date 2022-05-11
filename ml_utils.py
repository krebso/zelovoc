import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import os
from sklearn.decomposition import PCA
from math import ceil
import pandas as pd
import seaborn as sns
import torch


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def img_to_nparray( path, list_of_filename, size=( 64, 64 ) ):
    full_mat = None

    for fn in list_of_filename:
        fp = os.path.join( path, fn )
        current_image = image.load_img(fp, target_size = size, color_mode = 'grayscale')

        img_ts = image.img_to_array(current_image)

        img_ts = [ img_ts.ravel() ]
        if full_mat is None:
            full_mat = img_ts
        else:
            full_mat = np.concatenate( ( full_mat, img_ts ) )
    return full_mat


def find_img_stat( matrix, title, shape=( 64, 64 ), f=np.mean, **f_kwargs ):
    img = f( matrix, **f_kwargs )
    img = img.reshape( shape )
    plt.imshow( img, vmin=0, vmax=255,  )
    plt.title( title )
    plt.axis( "off" )
    plt.show()
    return img


def get_image_filepaths( class_="apple" ):
    return [ fn for fn in os.listdir(f'dataset/train/{ class_ }') if fn.endswith( ".jpg" ) ]



def eigenimages( matrix, n_components = 0.3, size = (64, 64)):
    pca = PCA( n_components=n_components, whiten=True )
    pca.fit( matrix )
    return pca


def plot_pca( pca, title, size = (64, 64) ):
    n = pca.n_components_
    fig = plt.figure( figsize=( 8, 8 ) )
    
    r = int( n ** 0.5 )
    c = ceil( n / r)
    
    for i in range(n):
        ax = fig.add_subplot( r, c, i + 1, xticks=[], yticks=[] )
        ax.imshow( pca.components_[ i ].reshape( size ) )
        plt.title( f"{ title }: { i + 1 }" )
    plt.axis( "off" )
    plt.show()
    
TRAIN_INFO = [
    0.33079434167573446,
    0.5266594124047878,
    0.5502357635110627,
    0.5749002538991658,
    0.6035545883206384,
    0.6398258977149075,
    0.6459920203119333,
    0.7174464998186434,
    0.7635110627493652,
    0.7598839318099383,
    0.7783822996010156,
    0.7678636198766775,
    0.7925281102647805,
    0.7881755531374682,
    0.7856365614798694,
    0.8019586507072906,
    0.808487486398259,
    0.7986942328618063,
    0.7950671019223794,
    0.8015959376133478,
    0.8026840768951758,
    0.8026840768951758,
    0.803772216177004,
    0.7947043888284366,
    0.8034095030830612
]

TEST_INFO = [
    0.6985507246376812,
    0.7681159420289855,
    0.7739130434782608,
    0.8115942028985508,
    0.8492753623188406,
    0.8289855072463769,
    0.8492753623188406,
    0.8753623188405797,
    0.8492753623188406,
    0.8782608695652174,
    0.8666666666666667,
    0.8782608695652174,
    0.881159420289855,
    0.8782608695652174,
    0.8840579710144928,
    0.8927536231884058,
    0.8898550724637682,
    0.8782608695652174,
    0.8724637681159421,
    0.8840579710144928,
    0.8956521739130435,
    0.8985507246376812,
    0.8869565217391304,
    0.8753623188405797,
    0.881159420289855
]