import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import ImagePair

import ObjectsSynthetic as obj

obj.test()

import PhotoViewer as pv
# pv.DrawCube(obj.CreateCube(10),ax)


import Camera
# =======
from Camera import Camera

from SingleImage import SingleImage

# B

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# create synthetic data
cube = obj.CreateCube(6)
pv.DrawCube(cube, ax)

# define camera
focal_length = 35
sensor_size = 25
camera1 = Camera(focal_length, np.array([0, 0]), None, None, None, sensor_size)

# define image
omega = 0
phi = 0
kappa = 0
Z = 50

img1 = SingleImage(camera1)
img1.exteriorOrientationParameters = np.array([[0, 0, Z, omega, phi, kappa]])

# scale of image frame in the plot
scale = 100

# draw image frame in world system
img1.drawSingleImage(cube, scale, ax)

image1Points = img1.GroundToImage(cube)
# image1Points = np.hstack((image1Points,np.ones((image1Points.shape[0],1))*Z))
plt.figure()
pv.drawImageFrame2D(img1.camera.sensorSize, img1.camera.sensorSize)
pv.DrawCube2D(image1Points, plt.gca())
plt.scatter(image1Points[:, 0], image1Points[:, 1])
# plt.show()


# חלק ג #

# second camera
camera2 = Camera(focal_length, np.array([0, 0]), None, None, None, sensor_size)

# second image
base = 10
img2 = SingleImage(camera2)
img2.exteriorOrientationParameters = np.array([[base, 0, Z, omega, phi, kappa]])

# draw image frame in world system
img2.drawSingleImage(cube, scale, ax)

image2Points = img2.GroundToImage(cube)
plt.figure()
plt.subplot(121)
pv.drawImageFrame2D(img2.camera.sensorSize, img2.camera.sensorSize)
plt.scatter(image2Points[:, 0], image2Points[:, 1])
# image2Points = np.hstack((image2Points,np.ones((image2Points.shape[0],1))*Z))
pv.DrawCube2D(image2Points, plt.gca())
plt.axis('equal')

plt.subplot(122)
pv.drawImageFrame2D(img1.camera.sensorSize, img1.camera.sensorSize)
pv.DrawCube2D(image1Points, plt.gca())
plt.scatter(image1Points[:, 0], image1Points[:, 1])
plt.axis('equal')

from ImagePair import ImagePair

imgPair = ImagePair(img1, img2)

# forward intersection
calcCube, sigma = imgPair.geometricIntersection(image1Points, image2Points, 'world')

# analysis

# # base size
# for i, b in enumerate(np.arange(0.1, 100, 0.01)):
#
#     img2.exteriorOrientationParameters = np.array([[b, 0, Z, omega, phi, kappa]])
#     image2Points = img2.GroundToImage(cube)
#     # check if still in the image
#     if np.max(np.abs(image2Points)) > sensor_size / 2:
#         i -= 1
#         break
#     imgPair = ImagePair(img1, img2)
#     # forward intersection
#     calcCube, sigma = imgPair.geometricIntersection(image1Points, image2Points, 'world')
#     if i == 0:
#         precision = np.linalg.norm(sigma)
#     else:
#         precision = np.vstack((precision, np.linalg.norm(sigma)))
# plt.figure()
# plt.plot(np.arange(0.1, 100, 0.01)[:i+1], precision)
#
# # rotation img2
# i=0
# for i, omega in enumerate(np.arange(1e-6, -0.25, -1e-3)):
#
#     img2.exteriorOrientationParameters = np.array([[10, 0, Z, omega, phi, kappa]])
#     image2Points = img2.GroundToImage(cube)
#     # check if still in the image
#     if np.max(np.abs(image2Points)) > sensor_size / 2:
#         i -= 1
#         break
#     imgPair = ImagePair(img1, img2)
#     # forward intersection
#     calcCube, sigma = imgPair.geometricIntersection(image1Points, image2Points, 'world')
#     if i == 0:
#         precision = np.linalg.norm(sigma)
#     else:
#         precision = np.vstack((precision, np.linalg.norm(sigma)))
# plt.figure()
# plt.subplot(121)
# plt.plot(np.arange(1e-6, -0.25, -1e-3)[:i + 1], precision)
#
# # rotation img1
# i=0
# for i, phi in enumerate(np.arange(1e-6, -0.25, -1e-3)):
#
#     img1.exteriorOrientationParameters = np.array([[0, 0, Z, omega, phi, kappa]])
#     image1Points = img1.GroundToImage(cube)
#     # check if still in the image
#     if np.max(np.abs(image1Points)) > sensor_size / 2 or np.max(np.abs(image2Points)) > sensor_size / 2:
#         i -= 1
#         print("out of image")
#         break
#     imgPair = ImagePair(img1, img2)
#     # forward intersection
#     calcCube, sigma = imgPair.geometricIntersection(image1Points, image2Points, 'world')
#     if i == 0:
#         precision = np.linalg.norm(sigma)
#     else:
#         precision = np.vstack((precision, np.linalg.norm(sigma)))
# plt.subplot(122)
# plt.plot(np.arange(1e-6, -0.25, -1e-3)[:i + 1], precision)




def runTest(coords,base=10,Z=50,omega=0,phi=0,kappa=0, noiseSizeGround=0, noiseSizeSamples=0):
    """
    Runs test by changing parameters
    """

    img1.exteriorOrientationParameters = np.array([[0, 0, Z, 0, 0, 0]])
    img2.exteriorOrientationParameters = np.array([[base, 0, Z, omega, phi, kappa]])
    noisyCoords = coords.copy()
    # generate ground points
    if noiseSizeGround > 0:
        noise = np.random.normal(0, noiseSizeGround, coords.shape)
        noisyCoords += noise
    # generate samples
    image1Points = img1.GroundToImage(noisyCoords)
    image2Points = img2.GroundToImage(noisyCoords)
    if noiseSizeSamples > 0:
        noise = np.random.normal(0, noiseSizeSamples, image1Points.shape)
        image1Points += noise
        image2Points += noise
    # check if still in the image
    if np.max(np.abs(image1Points)) > sensor_size / 2 or np.max(np.abs(image2Points)) > sensor_size / 2:
        print("out of image")
        return None
    imgPair = ImagePair(img1, img2)
    # forward intersection
    calcCoords, sigma = imgPair.geometricIntersection(image1Points, image2Points, 'world')
    diff = np.abs(calcCoords-coords)

    return sigma,diff

# add noise to ground points
rangeArray = np.arange(0, 0.15, 0.001)
for i, noiseSize in enumerate(rangeArray):
    sigma,diff = runTest(cube,noiseSizeGround=noiseSize,Z=200)
    if sigma is None:
        i -= 1
        break
    if i == 0:
        precision = np.linalg.norm(sigma)
        diffNorm = np.linalg.norm(diff)
    else:
        precision = np.vstack((precision, np.linalg.norm(sigma)))
        diffNorm = np.vstack((diffNorm, np.linalg.norm(diff)))

plt.figure()
plt.subplot(121)
plt.plot(rangeArray[:i + 1], precision)
plt.subplot(122)
plt.plot(rangeArray[:i + 1], diffNorm)

# add noise to sampled points
rangeArray = np.arange(0, 0.15, 0.001)
for i, noiseSize in enumerate(rangeArray):
    sigma,diff = runTest(cube, noiseSizeSamples=noiseSize,Z=200)
    if sigma is None:
        i -= 1
        break
    if i == 0:
        precision = np.linalg.norm(sigma)
        diffNorm = np.linalg.norm(diff)
    else:
        precision = np.vstack((precision, np.linalg.norm(sigma)))
        diffNorm = np.vstack((diffNorm, np.linalg.norm(diff)))

plt.figure()
plt.subplot(121)
plt.plot(rangeArray[:i + 1], precision)
plt.subplot(122)
plt.plot(rangeArray[:i + 1], diffNorm)
plt.show()

