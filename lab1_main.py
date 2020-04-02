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
# plt.show()

image1Points = img1.GroundToImage(cube)
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
plt.title('image 2')
plt.axis('equal')

plt.subplot(122)
pv.drawImageFrame2D(img1.camera.sensorSize, img1.camera.sensorSize)
pv.DrawCube2D(image1Points, plt.gca())
plt.scatter(image1Points[:, 0], image1Points[:, 1])
plt.title('image 1')
plt.axis('equal')

from ImagePair import ImagePair

imgPair = ImagePair(img1, img2)

# forward intersection
calcCube, sigma = imgPair.geometricIntersection(image1Points, image2Points, 'world')

# analysis

def runTest(coords,base=10,Z=50,omega1=0,phi1=0,kappa1=0,
            omega2=0,phi2=0,kappa2=0, noiseSizeGround=0, noiseSizeSamples=0):
    """
    Runs test by changing parameters
    """

    img1.exteriorOrientationParameters = np.array([[0, 0, Z, omega1, phi1, kappa1]])
    img2.exteriorOrientationParameters = np.array([[base, 0, Z, omega2, phi2, kappa2]])
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
        return None, None
    imgPair = ImagePair(img1, img2)
    # forward intersection
    calcCoords, sigma = imgPair.geometricIntersection(image1Points, image2Points, 'world')
    diff = np.abs(calcCoords-coords)

    return sigma,diff

# change base size
rangeArray = np.arange(0.01, 50, 0.01)
for i, base in enumerate(rangeArray):
    sigma,diff = runTest(cube,base=base)
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
plt.subplot(111)
plt.xlabel('size of base [m]')
plt.ylabel('precision [m]')
plt.plot(rangeArray[:i + 1], precision)

# change img1 phi angle
rangeArray = np.radians(np.arange(0, 70, 2))
for i, phi in enumerate(rangeArray):
    sigma,diff = runTest(cube,phi1=phi,Z=200)
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
plt.subplot(111)
plt.xlabel('angle of rotation [degrees]')
plt.ylabel('precision [m]')
plt.plot(np.degrees(rangeArray[:i + 1]), precision)

# change img2 phi angle
rangeArray =np.radians(np.arange(0, 90, 0.1))
for i, phi in enumerate(rangeArray):
    sigma,diff = runTest(cube,phi2=phi)
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
plt.subplot(111)
plt.xlabel('angle of rotation [degrees]')
plt.ylabel('precision [m]')
plt.plot(np.degrees(rangeArray[:i + 1]), precision)

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
plt.title('precision')
plt.xlabel('sigma noise')
plt.ylabel('error [m]')
plt.plot(rangeArray[:i + 1], precision)
plt.subplot(122)
plt.title('difference')
plt.xlabel('sigma noise')
plt.ylabel('precision [m]')
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
plt.title('precision')
plt.xlabel('sigma noise')
plt.ylabel('error [m]')
plt.plot(rangeArray[:i + 1], precision)
plt.subplot(122)
plt.title('difference')
plt.xlabel('sigma noise')
plt.ylabel('precision [m]')
plt.plot(rangeArray[:i + 1], diffNorm)
