import numpy as np
from numpy.linalg import solve, inv
from matplotlib import pyplot as plt
from Camera import Camera
import PhotoViewer as pv
from SingleImage import SingleImage
import MatrixMethods
from ObjectsSynthetic import *


if __name__ == '__main__':
    # Part A
    focal_length = 35  # [mm]
    sensor_size = 35  # [mm]
    xp = 0.02  # [mm]
    yp = 0.01  # [mm]
    k1_factor = 1e-5
    k2_factor = 1e-10
    k1 = 5*k1_factor
    k2 = 5*k2_factor
    camera1 = Camera(focal_length, np.array([xp, yp]), np.array([k1, k2]), None, None, sensor_size)

    img1 = SingleImage(camera1)
    omega = np.radians(90)
    phi = 0
    kappa = 0
    X1 = 0  # [m]
    Y1 = 0
    Z1 = 0
    img1.exteriorOrientationParameters = np.array([[X1, Y1, Z1, omega, phi, kappa]])

    # Clibration field
    calibration_field = np.array([[-1, 5, -2],
                                  [-1, 5, 2],
                                  [1, 5, 2],
                                  [1, 5, -2],
                                  [3, 7, -2],
                                  [3, 7, 3],
                                  [0, 7, 3],
                                  [-3, 7, 3],
                                  [-3, 7, -2],
                                  [0, 5, 0]])

    # sampeling points in synthetic system
    # points in camera space
    Ideal_camera_points = img1.GroundToImage(calibration_field)  # synthetic system
    camera_points = camera1.IdealCameraToCamera(Ideal_camera_points)

    # points in image space
    # img1.innerOrientationParameters = np.array([img1.camera.sensorSize / 2, 1, 0, img1.camera.sensorSize / 2, 0, -1])
    # image_points1 = img1.CameraToImage(camera_points)

    print('Ideal camera Points=', '\n', Ideal_camera_points)
    print('camera Points=', '\n', camera_points)

    # draw
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scale = 100
    img1.drawSingleImage(calibration_field, scale, ax, 'yes')
    ax.scatter(calibration_field[:, 0], calibration_field[:, 1], calibration_field[:, 2], c='g', s=50, marker='*')

    plt.figure()
    pv.drawImageFrame2D(img1.camera.sensorSize, img1.camera.sensorSize)
    plt.scatter(camera_points[:, 0], camera_points[:, 1])

    # Part B
    # calibration
    # Approximate values
    f = 35  # [mm]
    xp2 = 0  # in camera space [mm]
    yp2 = 0  # in camera space [mm]
    k1 = 0.
    k2 = 0.
    X0 = 0
    Y0 = 0
    Z0 = 0
    omega2 = np.radians(90)
    phi2 = 0
    kappa2 = 0
    approx_vals = np.array([f, xp2, yp2, k1, k2, X0, Y0, Z0, omega2, phi2, kappa2]).T

    calibration_params, sigma0, sigmaX, itr = camera1.Calibration(camera_points, calibration_field, approx_vals, img1, 0.001)
    calibration_params[8:] = np.degrees(calibration_params[8:])
    accuracy = np.diag(sigmaX)
    print('iteration','\n',itr)
    print('Accuracy norm','\n',np.linalg.norm(accuracy))
    MatrixMethods.PrintMatrix(calibration_params, 'calibration parameters', 10)
    MatrixMethods.PrintMatrix(accuracy,'Accuracy',10)

    plt.show()
    # # flat Clibration field
    #
    # focal_length = 35  # [mm]
    # sensor_size = 35  # [mm]
    # xp = 0.02  # [mm]
    # yp = 0.01  # [mm]
    # k1_factor = 1e-5
    # k2_factor = 1e-10
    # k1 = 5
    # k2 = 5
    # camera2 = Camera(focal_length, np.array([xp, yp]), np.array([k1_factor * k1, k2_factor * k2]), None, None,
    #                  sensor_size)
    #
    # img2 = SingleImage(camera2)
    # omega = np.radians(90)
    # phi = 0
    # kappa = 0
    # X1 = 0  # [m]
    # Y1 = 0
    # Z1 = 0
    # img2.exteriorOrientationParameters = np.array([[X1, Y1, Z1, omega, phi, kappa]])
    #
    #
    # calibration_field2 = np.array([[-1, 7, -2],
    #                                [-1, 7, 2],
    #                                [1, 7, 2],
    #                                [1, 7, -2],
    #                                [3, 7, -2],
    #                                [3, 7, 3],
    #                                [0, 7, 3],
    #                                [-3, 7, 3],
    #                                [-3, 7, -2],
    #                                [0, 7, 0]])
    #
    # # draw
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(calibration_field2[:, 0], calibration_field2[:, 1], calibration_field2[:, 2], c='g', s=50, marker='*')
    #
    # # points in camera space
    # Ideal_camera_points2 = img2.GroundToImage(calibration_field2)  # synthetic system
    # camera_points2 = camera2.IdealCameraToCamera(Ideal_camera_points2)
    #
    # print('Ideal camera Points=', '\n', Ideal_camera_points2)
    # print('camera Points=', '\n', camera_points2)
    #
    # # drawing
    # plt.figure()
    # pv.drawImageFrame2D(img2.camera.sensorSize, img2.camera.sensorSize)
    # plt.scatter(camera_points2[:, 0], camera_points2[:, 1])
    #
    # calibration_params, sigma0, sigmaX, itr = camera1.Calibration(camera_points2, calibration_field2, approx_vals, img2, 0.001)
    # print('calibration parameters', '\n', calibration_params)
    # print('iteration', '\n', itr)
    #
    # MatrixMethods.PrintMatrix(np.diag(sigmaX), 'Accuracy', 10)
    #
    #
    # plt.show()

