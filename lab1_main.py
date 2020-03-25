import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


import ImagePair


import ObjectsSynthetic as obj
obj.test()


import PhotoViewer as pv
pv.DrawCube(obj.CreateCube(10))


import Camera