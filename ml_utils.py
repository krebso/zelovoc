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
    (0.277, 2.030804706767895),
    (0.3145, 1.776546977687609),
    (0.3620, 1.4218134841031095),
    (0.4355, 1.0421814752237373),
    (0.4579, 0.8669572538300082),
    (0.4962, 0.8143771825116098),
    (0.5162, 0.7585341457809721),
    (0.5474, 0.68605150848531417),
    (0.5391, 0.60116396323827),
    (0.5909, 0.5006478046261097),
    (0.5892, 0.43002687905718054),
    (0.6250, 0.41629414399640327),
    (0.6304, 0.336497113615705),
    (0.6733, 0.299112865570843),
    (0.6639, 0.2604697487528405),
    (0.6861, 0.2592725630030394),
    (0.7035, 0.2590179510952765),
    (0.7194, 0.22723459697721666),
    (0.7769, 0.21975718277848862),
    (0.7474, 0.2823617479577063),
    (0.7058, 0.228727944636774245),
    (0.7587, 0.218884742648366175),
    (0.7826, 0.238751610256958735),
    (0.7729, 0.208920642570522203)
]

TEST_INFO = [
    (0.3467, 1.1973988986083246),
    (0.4806, 0.6770021037922965),
    (0.5035, 0.5832887752853239),
    (0.5477, 0.4881104358178073),
    (0.5819, 0.4393659701175917),
    (0.6048, 0.4329415605744437),
    (0.6347, 0.431413234286793),
    (0.6446, 0.34774536162808756),
    (0.7203, 0.317019213483111),
    (0.71, 0.2954649884810808),
    (0.7743, 0.2945933071993317),
    (0.7519, 0.2980838447058124),
    (0.7689, 0.26522030272119285),
    (0.7789, 0.27197254510263214),
    (0.7974, 0.25894323958032894),
    (0.7489, 0.2510832149312558),
    (0.7817, 0.25481164315999927),
    (0.7846, 0.2613924349213767),
    (0.7789, 0.2701210256110819),
    (0.7875, 0.2751597932906133),
    (0.7661, 0.29110002170976945),
    (0.7760, 0.27160994789338555),
    (0.7746, 0.26427298072587485),
    (0.7889, 0.25228861966752963),
    (0.7717, 0.27517284089199806)
]