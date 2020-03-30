import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


import ImagePair


import ObjectsSynthetic as obj
obj.test()


import PhotoViewer as pv
# pv.DrawCube(obj.CreateCube(10),ax)


# <<<<<<< HEAD
import Camera
# =======
from Camera import Camera


from SingleImage import SingleImage


# B

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# create synthetic data
cube = obj.CreateCube(6)
pv.DrawCube(cube,ax)

# define camera
focal_length = 35
sensor_size = 25
omega = 0
phi = 0
kappa = 0
camera1 = Camera(focal_length, np.array([0, 0]), None, None, None, sensor_size)

# define image
img1 = SingleImage(camera1)
img1.exteriorOrientationParameters = np.array([[0, 0, 100, omega, phi, kappa]])

# draw image frame in world system
img1.drawSingleImage(cube,ax)

imagePoints = img1.GroundToImage(cube)
plt.figure()
# fig2=plt.figure()
# ax2 = fig.add_subplot(122)
pv.drawImageFrame2D(img1.camera.sensorSize,img1.camera.sensorSize)
plt.scatter(imagePoints[:,0],imagePoints[:,1])
plt.show()
