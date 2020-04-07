import numpy as np
from numpy.linalg import solve, inv
from matplotlib import pyplot as plt
from Camera import Camera
import PhotoViewer as pv
from SingleImage import SingleImage
import MatrixMethods
from ObjectsSynthetic import *

# מעבדה 2
# שלב 1
# חלק א'
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
kappa = 0
Z = 20   # [m]
img1 = SingleImage(camera1)
img1.innerOrientationParameters = np.array([0,1,0,0,0,1])
img1.exteriorOrientationParameters = np.array([[0, 0, Z, omega, phi, kappa]])

# # draw
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# pv.drawOrientation(img1.RotationMatrix,img1.PerspectiveCenter,1,ax)
# pv.drawImageFrame(sensor_size/1000,sensor_size/1000,img1.RotationMatrix,img1.PerspectiveCenter,camera1.focalLength/1000,10,ax)
img1.drawSingleImage(squere,100,ax,'yes')
ax.scatter(squere[:,0],squere[:,1],squere[:,2], c='b', s=50,marker='^')


imagePoints1 = img1.GroundToImage(squere)
print(imagePoints1)

img1.exteriorOrientationParameters, sigma0, sigmaX =  img1.ComputeExteriorOrientation(imagePoints1,squere,0.001)
print('Adjustment test','\n',img1.exteriorOrientationParameters)

# adding noise
noiseSize = 0.01
noise = np.random.normal(0, noiseSize, imagePoints1.shape)
imagePoints1 += noise
img1.exteriorOrientationParameters, sigma0, sigmaX =  img1.ComputeExteriorOrientation(imagePoints1,squere,0.001)
print('after adding noise','\n',img1.exteriorOrientationParameters)

# חלק ב'
# cancel the noise
img1.exteriorOrientationParameters = np.array([[0, 0, 20, 0, 0, 0]])

# define synthetic image
imgS1 = SingleImage(camera1,'synthetic')
azimuth = np.radians(0)
phi = np.radians(45)
kappa = np.radians(0)
Z = 20
imgS1.exteriorOrientationParameters = np.array([[0, 0, Z, azimuth, phi, kappa]])
imgS1.exteriorOrientationParameters[0:3] = np.dot(imgS1.RotationMatrix,imgS1.PerspectiveCenter)

fig2 = plt.figure()
ax = fig2.add_subplot(111, projection='3d')
img1.drawSingleImage(squere,100,ax,'yes')
imgS1.drawSingleImage(squere,100,ax,'yes')
ax.scatter(squere[:,0],squere[:,1],squere[:,2], c='b', s=50,marker='^')

imagePoints2 = imgS1.GroundToImage(squere)
print(imagePoints2)

imgS1.innerOrientationParameters = np.array([0,1,0,0,0,1])
imgS1.exteriorOrientationParameters, sigma0, sigmaX =  imgS1.ComputeExteriorOrientation(imagePoints2,squere,0.001)
print(imgS1.exteriorOrientationParameters)

# adding noise
noiseSize = 0.01
noise = np.random.normal(0, noiseSize, imagePoints2.shape)
imagePoints2 += noise
imgS1.exteriorOrientationParameters, sigma0, sigmaX =  imgS1.ComputeExteriorOrientation(imagePoints2,squere,0.001)
print(imgS1.exteriorOrientationParameters)



# שלב 2
# define 8 synthetic image

azimuth = np.radians(np.array([0,0,0,0,0,0,15,45,90,0,0]))
phi = np.radians(np.array([0,0,0,0,0,0,5,15,30,60,80]))
kappa = np.radians(np.array([0,0,0,0,0,0,0,0,0,60,60]))
# kappa = np.radians(np.array([0,0,0,0,0,0,0,0]))
Z = np.array([50,100,500,100,100,100,50,50,50,50,50])
X = np.array([0,0,0,5,20,30,0,0,0,0,0])
Y = np.array([0,0,0,0,20,30,0,0,0,0,0])
fig3 = plt.figure()
ax = fig3.add_subplot(111, projection='3d')
ax.scatter(squere[:,0],squere[:,1],squere[:,2], c='b', s=50,marker='^')
img1.drawSingleImage(squere,100,ax,'yes')
images = []
exterior_before = []
exterior_after = []

for i in range(len(azimuth)):
    # define image
    imgS = SingleImage(camera1,'synthetic')
    imgS.innerOrientationParameters = np.array([0, 1, 0, 0, 0, 1])
    imgS.exteriorOrientationParameters = np.array([[X[i], Y[i], Z[i], kappa[i], phi[i], azimuth[i]]])
    imgS.exteriorOrientationParameters[0:3] = np.dot(imgS.RotationMatrix,imgS.PerspectiveCenter)
    # draw image in world system
    imgS.drawSingleImage(squere,100,ax,'yes')

    images.append(imgS)
    exterior_before.append(imgS.exteriorOrientationParameters)

    # draw in camera system
    imagePoints_S = imgS.GroundToImage(squere)
    plt.figure()
    pv.drawImageFrame2D(imgS.camera.sensorSize, imgS.camera.sensorSize)
    plt.scatter(imagePoints_S[:, 0], imagePoints_S[:, 1])


for i,imgS in enumerate(images):
    imagePoints3 = imgS.GroundToImage(squere)
    print('samples image: ', i, '\n', imagePoints3)
    print('Exterior Orientation before:', '\n', imgS.exteriorOrientationParameters)
    imgS.exteriorOrientationParameters, sigma0, sigmaX = imgS.ComputeExteriorOrientation(imagePoints3, squere,0.001)
    print('Exterior Orientation:', '\n', imgS.exteriorOrientationParameters)
    print("sigma X-diag:",'\n',np.diag(sigmaX))
    exterior_after.append(imgS.exteriorOrientationParameters)

plt.show()







