import numpy as np
from numpy.linalg import solve, inv
from matplotlib import pyplot as plt
from Camera import Camera
import PhotoViewer as pv
from SingleImage import SingleImage
import MatrixMethods
from ObjectsSynthetic import *


if __name__ == '__main__':

    # part A
    # define camera
    focal_length = 35  # [mm]
    # sensor_size = 3500  # [micron]
    sensor_size = 35  # [mm]
    xp=100  # [micron]
    yp=200  # [micron]

    camera1 = Camera(focal_length, np.array([xp, yp]), None, None, None, sensor_size)
    img1 = SingleImage(camera1)

    omega = 0
    phi = 0
    kappa = 0
    Z = 50  # [m]
    img1.exteriorOrientationParameters = np.array([[0, 0, Z, omega, phi, kappa]])

    camera_points = np.array([[2000, 2000],
                             [1500, 400],
                              [1000, 800],
                              [2300, 2100],
                              [1300, 1300],
                              [700, 700]])  # [micron]
    print(camera_points)

    shifted_points = camera1.ShiftedPrincipalPoint(camera_points)
    print('~~~~~','\n',shifted_points)

    correted_points = camera1.CorrectionToPrincipalPoint(shifted_points)
    print('~~~~~','\n',correted_points)

    # part B
    # create ground points
    gcp = np.array([[20,20,0],
                    [20,-20,0],
                    [0,-20,0],
                    [-20,-20,0],
                    [-20,20,0],
                    [0,20,0]])
    # sampeling points in camera system
    imagePoints1 = img1.GroundToImage(gcp)

    # draw
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scale = 100
    img1.drawSingleImage(gcp, scale, ax, 'yes')
    ax.scatter(gcp[:, 0], gcp[:, 1], gcp[:, 2], c='b', s=50, marker='^')

    plt.figure()
    pv.drawImageFrame2D(img1.camera.sensorSize, img1.camera.sensorSize)
    plt.scatter(imagePoints1[:, 0], imagePoints1[:, 1])

    plt.show()