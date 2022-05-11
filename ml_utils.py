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
    (0.5120, 1.4218134841031095),
    (0.6355, 1.0421814752237373),
    (0.6579, 0.8669572538300082),
    (0.6962, 0.8143771825116098),
    (0.7162, 0.7585341457809721),
    (0.7874, 0.68605150848531417),
    (0.8291, 0.60116396323827),
    (0.8909, 0.5006478046261097),
    (0.9192, 0.43002687905718054),
    (0.8950, 0.41629414399640327),
    (0.9204, 0.336497113615705),
    (0.9333, 0.299112865570843),
    (0.9439, 0.2604697487528405),
    (0.9361, 0.2592725630030394),
    (0.9435, 0.2590179510952765),
    (0.9694, 0.22723459697721666),
    (0.9369, 0.21975718277848862),
    (0.9474, 0.2823617479577063),
    (0.9358, 0.228727944636774245),
    (0.9287, 0.218884742648366175),
    (0.9326, 0.238751610256958735),
    (0.9329, 0.208920642570522203)
]

TEST_INFO = [
    (0.5467, 1.1973988986083246),
    (0.6806, 0.6770021037922965),
    (0.7835, 0.5832887752853239),
    (0.7977, 0.4881104358178073),
    (0.8519, 0.4393659701175917),
    (0.8348, 0.4329415605744437),
    (0.8547, 0.431413234286793),
    (0.8746, 0.34774536162808756),
    (0.9003, 0.317019213483111),
    (0.9260, 0.2954649884810808),
    (0.9146, 0.2945933071993317),
    (0.9219, 0.2980838447058124),
    (0.9289, 0.26522030272119285),
    (0.9189, 0.27197254510263214),
    (0.9174, 0.25894323958032894),
    (0.9289, 0.2510832149312558),
    (0.9117, 0.25481164315999927),
    (0.9246, 0.2613924349213767),
    (0.9389, 0.2701210256110819),
    (0.9375, 0.2751597932906133),
    (0.9261, 0.29110002170976945),
    (0.9360, 0.27160994789338555),
    (0.9146, 0.26427298072587485),
    (0.9389, 0.25228861966752963),
    (0.9117, 0.27517284089199806)
]