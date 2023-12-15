#Original and Tampered Image Dataset

import pandas as pd
import numpy as np
import keras 
import keras.models as M
import keras.layers as L
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from colorama import Fore as f
import tensorflow_addons as tfa
from tensorflow_addons.layers import InstanceNormalization as IN
