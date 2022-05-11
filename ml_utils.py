import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import os
from sklearn.decomposition import PCA
from math import ceil


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