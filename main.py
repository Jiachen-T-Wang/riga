import os, sys
import numpy as np

from URWM import Invinet
from utility_function import *

if __name__ == '__main__':

    dataset = sys.argv[1] # celebA
    wmtype = sys.argv[2]

    if wmtype=='bits':
        watermark = np.ones(256)
    elif wmtype=='img':
        watermark = np.load("copyright.npy")

    name = sys.argv[3] # 'wm_107'

    invinet = Invinet(dataset=dataset, watermark=watermark, 
                      lambda_1=0.05, lambda_2=0, batch_size=100, n_detector=5, clip_value=0.1, n_embedder=20)

    if dataset=='mnist':
        invinet.train(epochs=100)
    elif dataset=='celebA':
        invinet.train(epochs=10)
    elif dataset=='twitter':
        invinet.train(epochs=3)

    invinet.save_all(name)


