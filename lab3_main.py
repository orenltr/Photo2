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
    sensor_size = 35  # [mm]
    xp = 0.1  # [mm]
    yp = 0.2  # [mm]
    # sensor_size = 35000  # [micron]
    # xp=100  # [micron]
    # yp=200  # [micron]


    camera1 = Camera(focal_length, np.array([xp, yp]),None, None, None, sensor_size)
    img1 = SingleImage(camera1)

    omega = 0
    phi = 0
    kappa = 0
    Z = 50  # [m]
    img1.exteriorOrientationParameters = np.array([[0, 0, Z, omega, phi, kappa]])

    camera_points = np.array([[3, 3],
                              [7, 4],
                              [10, 8],
                              [12, 15],
                              [13, 13],
                              [16, 6]])  # [mm]
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
    cameraPoints1 = img1.GroundToImage(gcp)
    print('camera Points=','\n',cameraPoints1)

    # # draw
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # scale = 100
    # img1.drawSingleImage(gcp, scale, ax, 'yes')
    # ax.scatter(gcp[:, 0], gcp[:, 1], gcp[:, 2], c='b', s=50, marker='^')
    #
    # plt.figure()
    # pv.drawImageFrame2D(img1.camera.sensorSize, img1.camera.sensorSize)
    # plt.scatter(imagePoints1[:, 0], imagePoints1[:, 1])
    #
    # plt.show()

    # calculate k1, k2
    # assume maximal radial distortions in the edges
    r = np.sqrt((camera1.sensorSize/2 - camera1.principalPoint[0]) ** 2 +
                (camera1.sensorSize/2 - camera1.principalPoint[1]) ** 2)
    # dr = k1*r^3+k2*r^5....
    k2 = 0.01/(r**5)
    k1 = (0.05-k2*r**5)/(r**3)
    print('k1=','\n',k1)
    print('k2=','\n',k2)

    camera1.radial_distortions = np.array([k1,k2])

    # create array of points in camera space
    x = np.linspace(0, camera1.sensorSize/2, 20)
    y = np.linspace(0, camera1.sensorSize/2, 20)
    X = np.zeros((len(x),2))
    X[:,0] = x
    X[:,1] = y
    # X = np.array([[0,2,4,6,8,10,12,14,16,17.5],
    #               [0,2,4,6,8,10,12,14,16,17.5]]).T

    # define the distances from the principal point
    r = np.sqrt((X[:, 0] - camera1.principalPoint[0]) ** 2 +
                (X[:, 1] - camera1.principalPoint[1]) ** 2)
    # define the radial distortions (dr)
    dx, dy, dr = camera1.ComputeRadialDistortions(X)
    fig = plt.figure()
    ax = plt.plot(r, dr, 'b-o')
    plt.title('Radial distortion depends on distance from the principal point ')
    plt.xlabel('r [mm]')
    plt.ylabel('delta_r [mm]')

    # # correction for principal point
    # corrected_points1 = camera1.CorrectionToPrincipalPoint(cameraPoints1)
    # print('corrected points for principal point=', '\n', corrected_points1)

    # correction for radial distortions
    corrected_points2 = camera1.CorrectionToRadialDistortions(cameraPoints1)
    print('corrected points for radial distortion=', '\n', corrected_points2)

    # create array of principal points
    xp = np.linspace(0, 1, 10)
    yp = np.linspace(0, 1, 10)
    principal_points = np.zeros((len(xp), 2))
    principal_points[:, 0] = xp
    principal_points[:, 1] = yp
    # check radial distortion for different xp,yp
    check_point = np.array([[10,10]])
    dr2 = np.zeros([len(xp)])
    for i,p in enumerate(principal_points):
        camera1.principalPoint = p
        dx,dy,dr2[i] = camera1.ComputeRadialDistortions(check_point)
    print(dr2)

    plt.figure()
    plt.plot(principal_points, dr2, 'r-o')
    # plt.title('Radial distortion depends on principal point ')
    plt.xlabel('r [mm]')
    plt.ylabel('delta_r [mm]')
    plt.show()

    # # plot the radial distortion
    # X = np.arange(-10, 10, 1)
    # Y = np.arange(-10, 10, 1)
    # U, V = np.meshgrid(X, Y)
    #
    # fig, ax = plt.subplots()
    # q = ax.quiver(X, Y, U, V)
    # ax.quiverkey(q, X=0.3, Y=1.1, U=10,
    #              label='Quiver key, length = 10', labelpos='E')
    #
    # plt.show()

