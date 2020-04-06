import numpy as np
from numpy.linalg import solve, inv
from matplotlib import pyplot as plt
from Camera import Camera
import PhotoViewer as pv
from SingleImage import SingleImage
import MatrixMethods
from ObjectsSynthetic import *

# create synthetic data
edgeSize = 10   # [m]
squere = np.array([[-edgeSize/2,edgeSize/2,0],[edgeSize/2,edgeSize/2,0],[edgeSize/2,-edgeSize/2,0],[-edgeSize/2,-edgeSize/2,0]])

# define camera
focal_length = 35   # [mm]
sensor_size = 25    # [mm]
camera1 = Camera(focal_length, np.array([0, 0]), None, None, None, sensor_size)

# define image
omega = 0
phi = 0
kappa = np.radians(50)
Z = 20   # [m]
img1 = SingleImage(camera1)
img1.innerOrientationParameters = np.array([0,1,0,0,0,1])
img1.exteriorOrientationParameters = np.array([[0, 0, Z, omega, phi, kappa]])

# # draw
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
pv.drawOrientation(img1.RotationMatrix,img1.PerspectiveCenter,1,ax)
pv.drawImageFrame(sensor_size/1000,sensor_size/1000,img1.RotationMatrix,img1.PerspectiveCenter,camera1.focalLength/1000,10,ax)
# # img1.drawSingleImage(squere,100,ax)
# ax.scatter(squere[:,0],squere[:,1],squere[:,2], c='b', s=50,marker='^')
plt.show()
#
# imagePoints1 = img1.GroundToImage(squere)
# groundPoints = img1.ImageToGround_GivenZ(imagePoints1,np.zeros(imagePoints1.shape[0]))
# print(groundPoints)

