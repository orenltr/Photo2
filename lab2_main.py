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
phi = np.radians(50)
kappa = 0
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
# plt.show()
#
# imagePoints1 = img1.GroundToImage(squere)
# groundPoints = img1.ImageToGround_GivenZ(imagePoints1,np.zeros(imagePoints1.shape[0]))
# print(groundPoints)

imgS1 = SingleImage(camera1,'synthetic')
azimuth = np.radians(0)
phi = np.radians(45)
kappa = np.radians(0)
Z = 100
imgS1.exteriorOrientationParameters = np.array([[0, 0, Z, azimuth, phi, kappa]])
imgS1.exteriorOrientationParameters[0:3] = np.dot(imgS1.RotationMatrix,imgS1.PerspectiveCenter)

imgS2 = SingleImage(camera1,'synthetic')
imgS2.innerOrientationParameters = np.array([0,1,0,0,0,1])
azimuth = np.radians(90)
phi = np.radians(45)
kappa = np.radians(45)
Z = 100
imgS2.exteriorOrientationParameters = np.array([[0, 0, Z, azimuth, phi, kappa]])
imgS2.exteriorOrientationParameters[0:3] = np.dot(imgS2.RotationMatrix,imgS2.PerspectiveCenter)
print(imgS2.PerspectiveCenter)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
imgS1.drawSingleImage(squere,100,ax,'yes')
imgS2.drawSingleImage(squere,100,ax,'yes')
ax.scatter(squere[:,0],squere[:,1],squere[:,2], c='b', s=50,marker='^')

imagePoints_S1 = imgS1.GroundToImage(squere)
imagePoints_S2 = imgS2.GroundToImage(squere)
plt.figure()
pv.drawImageFrame2D(imgS1.camera.sensorSize, imgS1.camera.sensorSize)
plt.scatter(imagePoints_S1[:, 0], imagePoints_S1[:, 1])
plt.figure()
pv.drawImageFrame2D(imgS2.camera.sensorSize, imgS2.camera.sensorSize)
plt.scatter(imagePoints_S2[:, 0], imagePoints_S2[:, 1])

print(imgS2.ComputeExteriorOrientation(imagePoints_S2,squere,0.001))
plt.show()