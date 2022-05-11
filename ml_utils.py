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
    (0.1961, 2.9178917013814325),
    (0.4077, 2.030804706767895),
    (0.4745, 1.776546977687609),
    (0.5120, 1.6218134841031095),
    (0.5355, 1.5421814752237373),
    (0.5579, 1.4669572538300082),
    (0.5762, 1.4143771825116098),
    (0.6462, 1.1585341457809721),
    (0.6774, 1.0605150848531417),
    (0.6909, 1.0172116396323827),
    (0.6909, 1.006478046261097),
    (0.6992, 0.9902687905718054),
    (0.7050, 0.9629414399640327),
    (0.7104, 0.936497113615705),
    (0.7133, 0.939112865570843),
    (0.7239, 0.8904697487528405),
    (0.7361, 0.8592725630030394),
    (0.7335, 0.8590179510952765),
    (0.7294, 0.8723459697721666),
    (0.7169, 0.8975718277848862),
    (0.7274, 0.8823617479577063),
    (0.7258, 0.8727944636774245),
    (0.7287, 0.8884742648366175),
    (0.7326, 0.8751610256958735),
    (0.7329, 0.8920642570522203)
]

VALIDATION_INFO = [
    (0.6467, 1.1973988986083246),
    (0.7806, 0.6770021037922965),
    (0.7835, 0.5832887752853239),
    (0.7977, 0.4881104358178073),
    (0.8519, 0.4393659701175917),
    (0.8348, 0.4329415605744437),
    (0.8547, 0.431413234286793),
    (0.8746, 0.34774536162808756),
    (0.8803, 0.317019213483111),
    (0.8860, 0.2954649884810808),
    (0.8746, 0.2945933071993317),
    (0.8689, 0.2980838447058124),
    (0.8889, 0.26522030272119285),
    (0.8889, 0.27197254510263214),
    (0.8974, 0.25894323958032894),
    (0.8889, 0.2510832149312558),
    (0.8917, 0.25481164315999927),
    (0.8946, 0.2613924349213767),
    (0.8889, 0.2701210256110819),
    (0.8775, 0.2751597932906133),
    (0.8661, 0.29110002170976945),
    (0.8860, 0.27160994789338555),
    (0.8946, 0.26427298072587485),
    (0.8889, 0.25228861966752963),
    (0.8917, 0.27517284089199806)
]