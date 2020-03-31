import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


import ImagePair


import ObjectsSynthetic as obj


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

# plt.figure()
# pv.drawImageFrame2D(img1.camera.sensorSize,img1.camera.sensorSize)
#
# # calculate the projection of the cube in the camera system
# imagePoints = img1.GroundToImage(cube)
# x_imgPoints = imagePoints[:,0]
# y_imgPoints = imagePoints[:,1]
# # draw the top of the cube projection
# pv.connect2Dpoints(x_imgPoints,y_imgPoints,1,4,'b-o')
# pv.connect2Dpoints(x_imgPoints,y_imgPoints,4,5,'b-o')
# pv.connect2Dpoints(x_imgPoints,y_imgPoints,5,8,'b-o')
# pv.connect2Dpoints(x_imgPoints,y_imgPoints,8,1,'b-o')
# # draw the base of the cube projection
# pv.connect2Dpoints(x_imgPoints,y_imgPoints,2,3,'k-o')
# pv.connect2Dpoints(x_imgPoints,y_imgPoints,3,6,'k-o')
# pv.connect2Dpoints(x_imgPoints,y_imgPoints,6,7,'k-o')
# pv.connect2Dpoints(x_imgPoints,y_imgPoints,7,2,'k-o')

# # Q1
# focalLength_array = np.array([18,28,35,55,85,100,150,200,300])
#
# for f in focalLength_array:
#     # define camera
#     camera2 = Camera(f, np.array([0, 0]), None, None, None, sensor_size)
#     # define image
#     img2 = SingleImage(camera2)
#     img2.exteriorOrientationParameters = np.array([[0, 0, 100, omega, phi, kappa]])
#
#     plt.figure()
#     pv.drawImageFrame2D(img2.camera.sensorSize, img2.camera.sensorSize)
#     # calculate the projection of the cube in the camera system
#     imagePoints = img2.GroundToImage(cube)
#     x_imgPoints = imagePoints[:, 0]
#     y_imgPoints = imagePoints[:, 1]
#     # draw the top of the cube projection
#     pv.connect2Dpoints(x_imgPoints, y_imgPoints, 1, 4, 'b-o')
#     pv.connect2Dpoints(x_imgPoints, y_imgPoints, 4, 5, 'b-o')
#     pv.connect2Dpoints(x_imgPoints, y_imgPoints, 5, 8, 'b-o')
#     pv.connect2Dpoints(x_imgPoints, y_imgPoints, 8, 1, 'b-o')
#     # draw the base of the cube projection
#     pv.connect2Dpoints(x_imgPoints, y_imgPoints, 2, 3, 'k-o')
#     pv.connect2Dpoints(x_imgPoints, y_imgPoints, 3, 6, 'k-o')
#     pv.connect2Dpoints(x_imgPoints, y_imgPoints, 6, 7, 'k-o')
#     pv.connect2Dpoints(x_imgPoints, y_imgPoints, 7, 2, 'k-o')
#
# # Q2
# focal_length2 = 35
# Z0_array = np.array([15,30,50,75,100,200])
# for Z0 in Z0_array:
#     # define camera
#     camera3 = Camera(focal_length2, np.array([0, 0]), None, None, None, sensor_size)
#     # define image
#     img3 = SingleImage(camera3)
#     img3.exteriorOrientationParameters = np.array([[0, 0, Z0, omega, phi, kappa]])
#
#     plt.figure()
#     pv.drawImageFrame2D(img3.camera.sensorSize, img3.camera.sensorSize)
#     # calculate the projection of the cube in the camera system
#     imagePoints = img3.GroundToImage(cube)
#     x_imgPoints = imagePoints[:, 0]
#     y_imgPoints = imagePoints[:, 1]
#     # draw the top of the cube projection
#     pv.connect2Dpoints(x_imgPoints, y_imgPoints, 1, 4, 'b-o')
#     pv.connect2Dpoints(x_imgPoints, y_imgPoints, 4, 5, 'b-o')
#     pv.connect2Dpoints(x_imgPoints, y_imgPoints, 5, 8, 'b-o')
#     pv.connect2Dpoints(x_imgPoints, y_imgPoints, 8, 1, 'b-o')
#     # draw the base of the cube projection
#     pv.connect2Dpoints(x_imgPoints, y_imgPoints, 2, 3, 'k-o')
#     pv.connect2Dpoints(x_imgPoints, y_imgPoints, 3, 6, 'k-o')
#     pv.connect2Dpoints(x_imgPoints, y_imgPoints, 6, 7, 'k-o')
#     pv.connect2Dpoints(x_imgPoints, y_imgPoints, 7, 2, 'k-o')
#
# # Q3
# focal_length3 = 300
# ZO_3 = 200
# # define camera
# camera4 = Camera(focal_length3, np.array([0, 0]), None, None, None, sensor_size)
# # define image
# img4 = SingleImage(camera4)
# img4.exteriorOrientationParameters = np.array([[0, 0, ZO_3, omega, phi, kappa]])
#
# plt.figure()
# pv.drawImageFrame2D(img4.camera.sensorSize, img4.camera.sensorSize)
# # calculate the projection of the cube in the camera system
# imagePoints = img4.GroundToImage(cube)
# x_imgPoints = imagePoints[:, 0]
# y_imgPoints = imagePoints[:, 1]
# # draw the top of the cube projection
# pv.connect2Dpoints(x_imgPoints, y_imgPoints, 1, 4, 'b-o')
# pv.connect2Dpoints(x_imgPoints, y_imgPoints, 4, 5, 'b-o')
# pv.connect2Dpoints(x_imgPoints, y_imgPoints, 5, 8, 'b-o')
# pv.connect2Dpoints(x_imgPoints, y_imgPoints, 8, 1, 'b-o')
# # draw the base of the cube projection
# pv.connect2Dpoints(x_imgPoints, y_imgPoints, 2, 3, 'k-o')
# pv.connect2Dpoints(x_imgPoints, y_imgPoints, 3, 6, 'k-o')
# pv.connect2Dpoints(x_imgPoints, y_imgPoints, 6, 7, 'k-o')
# pv.connect2Dpoints(x_imgPoints, y_imgPoints, 7, 2, 'k-o')
#

# # Q4
# kappa_array = np.array([0,30,45,60,90])
# for k in kappa_array:
#     # define camera
#     camera5 = Camera(35, np.array([0, 0]), None, None, None, sensor_size)
#     # define image
#     img5 = SingleImage(camera5)
#     img5.exteriorOrientationParameters = np.array([[0, 0, 100, omega, phi, k]])
#
#     plt.figure()
#     pv.drawImageFrame2D(img5.camera.sensorSize, img5.camera.sensorSize)
#     # calculate the projection of the cube in the camera system
#     imagePoints = img5.GroundToImage(cube)
#     x_imgPoints = imagePoints[:, 0]
#     y_imgPoints = imagePoints[:, 1]
#     # draw the top of the cube projection
#     pv.connect2Dpoints(x_imgPoints, y_imgPoints, 1, 4, 'b-o')
#     pv.connect2Dpoints(x_imgPoints, y_imgPoints, 4, 5, 'b-o')
#     pv.connect2Dpoints(x_imgPoints, y_imgPoints, 5, 8, 'b-o')
#     pv.connect2Dpoints(x_imgPoints, y_imgPoints, 8, 1, 'b-o')
#     # draw the base of the cube projection
#     pv.connect2Dpoints(x_imgPoints, y_imgPoints, 2, 3, 'k-o')
#     pv.connect2Dpoints(x_imgPoints, y_imgPoints, 3, 6, 'k-o')
#     pv.connect2Dpoints(x_imgPoints, y_imgPoints, 6, 7, 'k-o')
#     pv.connect2Dpoints(x_imgPoints, y_imgPoints, 7, 2, 'k-o')

# # Q5
# phi_array = np.array([0,30,45,60,75])
# for p in phi_array:
#     # define camera
#     camera6 = Camera(35, np.array([0, 0]), None, None, None, sensor_size)
#     # define image
#     img6 = SingleImage(camera6)
#     img6.exteriorOrientationParameters = np.array([[0, 0, 100, omega, p, kappa]])
#
#     plt.figure()
#     pv.drawImageFrame2D(img6.camera.sensorSize, img6.camera.sensorSize)
#     # calculate the projection of the cube in the camera system
#     imagePoints = img6.GroundToImage(cube)
#     x_imgPoints = imagePoints[:, 0]
#     y_imgPoints = imagePoints[:, 1]
#     # draw the top of the cube projection
#     pv.connect2Dpoints(x_imgPoints, y_imgPoints, 1, 4, 'b-o')
#     pv.connect2Dpoints(x_imgPoints, y_imgPoints, 4, 5, 'b-o')
#     pv.connect2Dpoints(x_imgPoints, y_imgPoints, 5, 8, 'b-o')
#     pv.connect2Dpoints(x_imgPoints, y_imgPoints, 8, 1, 'b-o')
#     # draw the base of the cube projection
#     pv.connect2Dpoints(x_imgPoints, y_imgPoints, 2, 3, 'k-o')
#     pv.connect2Dpoints(x_imgPoints, y_imgPoints, 3, 6, 'k-o')
#     pv.connect2Dpoints(x_imgPoints, y_imgPoints, 6, 7, 'k-o')
#     pv.connect2Dpoints(x_imgPoints, y_imgPoints, 7, 2, 'k-o')

# Q6
omega_array = np.array([0,30,45,60,75])
phi_array = np.array([0,30,45,60,75])
for i,p in enumerate(phi_array):
    # define camera
    camera7 = Camera(35, np.array([0, 0]), None, None, None, sensor_size)
    # define image
    img7 = SingleImage(camera7)
    img7.exteriorOrientationParameters = np.array([[0, 0, 100, omega_array[i], p, kappa]])

    plt.figure()
    pv.drawImageFrame2D(img7.camera.sensorSize, img7.camera.sensorSize)
    # calculate the projection of the cube in the camera system
    imagePoints = img7.GroundToImage(cube)
    x_imgPoints = imagePoints[:, 0]
    y_imgPoints = imagePoints[:, 1]
    # draw the top of the cube projection
    pv.connect2Dpoints(x_imgPoints, y_imgPoints, 1, 4, 'b-o')
    pv.connect2Dpoints(x_imgPoints, y_imgPoints, 4, 5, 'b-o')
    pv.connect2Dpoints(x_imgPoints, y_imgPoints, 5, 8, 'b-o')
    pv.connect2Dpoints(x_imgPoints, y_imgPoints, 8, 1, 'b-o')
    # draw the base of the cube projection
    pv.connect2Dpoints(x_imgPoints, y_imgPoints, 2, 3, 'k-o')
    pv.connect2Dpoints(x_imgPoints, y_imgPoints, 3, 6, 'k-o')
    pv.connect2Dpoints(x_imgPoints, y_imgPoints, 6, 7, 'k-o')
    pv.connect2Dpoints(x_imgPoints, y_imgPoints, 7, 2, 'k-o')

plt.show()
