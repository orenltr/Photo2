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
    xp = 0.05  # [mm]
    yp = 0.05  # [mm]
    K1 = 0.5e-8  # [mm]
    K2 = 0.5e-12  # [mm]
    camera1 = Camera(focal_length, np.array([xp, yp]), np.array([K1, K2]), None, 'no fiducials', sensor_size)
    img1 = SingleImage(camera1)
    omega = np.radians(90)
    phi = 0
    kappa = 0
    Z = 0  # [m]
    img1.exteriorOrientationParameters = np.array([[0, 0, Z, omega, phi, kappa]])

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

    # sampeling points in camera system
    cameraPoints1 = img1.GroundToImage(calibration_field)
    corrected_points1 = camera1.CorrectionToPrincipalPoint(cameraPoints1)
    corrected_points2 = camera1.CorrectionToRadialDistortions(corrected_points1)
    print('camera Points=', '\n', cameraPoints1)
    print('correted points to Principal Point=', '\n', corrected_points1)
    print('corrected points for radial distortion=', '\n', corrected_points2)

    #image points
    img1.innerOrientationParameters = np.array([img1.camera.sensorSize/2,1,0,img1.camera.sensorSize/2,0,-1])
    print('image Inner Orientation','\n',img1.innerOrientationParameters)
    image_points1 = img1.CameraToImage(corrected_points2)
    # image_points1 = img1.CameraToImage(cameraPoints1)
    print('image points=', '\n', image_points1)

    # draw
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scale = 100
    img1.drawSingleImage(calibration_field, scale, ax, 'yes')
    ax.scatter(calibration_field[:, 0], calibration_field[:, 1], calibration_field[:, 2], c='g', s=50, marker='*')

    plt.figure()
    pv.drawImageFrame2D(img1.camera.sensorSize, img1.camera.sensorSize)
    plt.scatter(corrected_points2[:, 0], corrected_points2[:, 1])



    # Part B
    # calibration
    # Approximate values
    f = 35  # [mm]
    # xp2 = camera1.sensorSize / 2  # in image space [mm]
    # yp2 = camera1.sensorSize / 2  # in image space [mm]
    xp2 = 0  # in camera space [mm]
    yp2 = 0  # in camera space [mm]
    k1 = 0
    k2 = 0
    X0 = 0
    Y0 = 0
    Z0 = 0
    omega2 = np.radians(90)
    phi2 = 0
    kappa2 = 0
    approx_vals = np.array([f, xp2, yp2, k1, k2, X0, Y0, Z0, omega2, phi2, kappa2]).T

    # l0 = camera1.ComputeObservationVectorForCalibration(calibration_field,img1)
    # print(l0)


    # # check ExteriorOrientation
    # img1.ComputeExteriorOrientation(image_points1, calibration_field, 0.001)

    calibration_params, sigma0, sigmaX, itr = camera1.Calibration(image_points1, calibration_field, approx_vals, img1, 0.001)
    print('calibration parameters','\n',calibration_params)
    print('iteration','\n',itr)
    # print('sigma0','\n',sigma0)
    # print('sigmaX','\n',sigmaX)


    plt.show()