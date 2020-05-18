import numpy as np
from SingleImage import *
from scipy.linalg import rq
import ObjectsSynthetic as obj

if __name__ == '__main__':
    # create synthetic data
    edgeSize = 10  # [m]
    cube = obj.CreateCube(edgeSize)

    # define camera
    focal_length = 153  # [mm]
    # sensor_size = 25  # [mm]
    camera1 = Camera(focal_length, np.array([0.2, 0.2]), None, None, None, None)

    # define image
    omega = -30
    phi = 45
    kappa = 60
    Z = 50  # [m]
    img1 = SingleImage(camera1)
    img1.innerOrientationParameters = np.array([0, 1, 0, 0, 0, 1])
    img1.exteriorOrientationParameters = np.array([[15, 10, Z, np.radians(omega), np.radians(phi), np.radians(kappa)]])

    imagePoints1 = img1.GroundToImage(cube)
    print(imagePoints1)

    cube = np.hstack([cube,np.ones((len(cube), 1))])
    # imagePoints1 = np.hstack([imagePoints1,np.ones((len(imagePoints1), 1))])
    p1 = img1.PerspectiveMatrix
    imagePoints = np.dot(p1,cube.T).T
    imagePoints = imagePoints/imagePoints[:,2,np.newaxis]
    img1.DLT(imagePoints[:, :2],cube[:,:3])
    k = img1.CalibrationMatrix
    img1.ComputeDLTDesignMatrix(imagePoints1, cube)
