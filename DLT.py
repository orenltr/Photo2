import numpy as np
from SingleImage import *
from scipy.linalg import rq, norm
import ObjectsSynthetic as obj


def test1(focalLength, sensorSize, X, Y, Z, kappa=0., phi=0., azimuth=0., xp=0., yp=0.,  system='middle', noiseSize=0.):
    """ testing DLT model """


    # create synthetic data
    edgeSize = 10  # [m]
    groundPoints = obj.CreateCube(edgeSize)
    # define camera
    camera1 = Camera(focalLength, np.array([xp, yp]), None, None, None, sensorSize)
    # define image
    img = SingleImage(camera1,'synthetic')
    img.innerOrientationParameters = np.array([0, 1, 0, 0, 0, 1])
    img.exteriorOrientationParameters = np.array([[X, Y, Z, np.radians(kappa), np.radians(phi), np.radians(azimuth)]])

    # creating synthetic samples
    realCalibrationMatrix = camera1.CalibrationMatrix
    realPercpectiveCenter = np.copy(img.PerspectiveCenter)
    realRotationMatrix = img.RotationMatrix

    groundPoints = np.hstack([groundPoints, np.ones((len(groundPoints), 1))])  # change to homogeneous representation
    imagePoints = np.dot(img.PerspectiveMatrix, groundPoints.T).T  # creating samples by using perspective matrix
    imagePoints = imagePoints / imagePoints[:, 2, np.newaxis]  # normalize

    # changing image system
    if system == 'top left':  # image axis begin from top left corner
        t = np.array([[1, 0, img.camera.sensorSize / 2],
                      [0, -1, img.camera.sensorSize / 2],
                      [0, 0, 1]])
        imagePoints = np.dot(t, imagePoints.T).T
    if system == 'bottom left':  # image axis begin from bottom left corner
        t = np.array([[1, 0, img.camera.sensorSize / 2],
                      [0, 1, img.camera.sensorSize / 2],
                      [0, 0, 1]])
        imagePoints = np.dot(t, imagePoints.T).T
    # adding noise
    if noiseSize > 0:
        noise = np.random.normal(0, noiseSize, imagePoints.shape)
        imagePoints += noise
    # applying DLT model
    img.DLT(imagePoints[:, :2], groundPoints[:, :3])
    printResults(img,realCalibrationMatrix,realPercpectiveCenter,realRotationMatrix)
    # errors
    errorPerspective = norm(realPercpectiveCenter-img.PerspectiveCenter)
    errorRotation = norm(abs(realRotationMatrix)-abs(img.RotationMatrix))
    errorCalibration = norm(realCalibrationMatrix-img.camera.CalibrationMatrix)
    return np.array([errorPerspective, errorRotation, errorCalibration])

def test2(focalLength, sensorSize, X, Y, Z, kappa=0., phi=0., azimuth=0., xp=0., yp=0.,  system='middle', noiseSize=0.):
    """ testing normal model """


    # create synthetic data
    edgeSize = 10  # [m]
    groundPoints = obj.CreateCube(edgeSize)
    # define camera
    camera1 = Camera(focalLength, np.array([xp, yp]), None, None, None, sensorSize)
    # define image
    img = SingleImage(camera1,'synthetic')
    img.innerOrientationParameters = np.array([0, 1, 0, 0, 0, 1])
    img.exteriorOrientationParameters = np.array([[X, Y, Z, np.radians(kappa), np.radians(phi), np.radians(azimuth)]])

    # creating synthetic samples
    realCalibrationMatrix = camera1.CalibrationMatrix
    realPercpectiveCenter = np.copy(img.PerspectiveCenter)
    realRotationMatrix = img.RotationMatrix

    groundPoints = np.hstack([groundPoints, np.ones((len(groundPoints), 1))])  # change to homogeneous representation
    imagePoints = np.dot(img.PerspectiveMatrix, groundPoints.T).T  # creating samples by using perspective matrix
    imagePoints = imagePoints / imagePoints[:, 2, np.newaxis]  # normalize

    # changing image system
    if system == 'top left':  # image axis begin from top left corner
        t = np.array([[1, 0, img.camera.sensorSize / 2],
                      [0, -1, img.camera.sensorSize / 2],
                      [0, 0, 1]])
        imagePoints = np.dot(t, imagePoints.T).T
        img.innerOrientationParameters = np.array([img.camera.sensorSize / 2, 1, 0, img.camera.sensorSize / 2, 0, -1])
    if system == 'bottom left':  # image axis begin from bottom left corner
        t = np.array([[1, 0, img.camera.sensorSize / 2],
                      [0, 1, img.camera.sensorSize / 2],
                      [0, 0, 1]])
        imagePoints = np.dot(t, imagePoints.T).T
        img.innerOrientationParameters = np.array([img.camera.sensorSize / 2, 1, 0, img.camera.sensorSize / 2, 0, 1])
    # adding noise
    if noiseSize > 0:
        noise = np.random.normal(0, noiseSize, imagePoints.shape)
        imagePoints += noise
    # applying co-linear model
    img.ComputeExteriorOrientation(imagePoints[:, :2], groundPoints[:, :3],0.0001)
    # img.ComputeExteriorOrientation(imagePoints[:5, :2], groundPoints[:5, :3],0.0001) # reducing observations number for compare with DLT
    printResults(img,realCalibrationMatrix,realPercpectiveCenter,realRotationMatrix)
    # errors
    errorPerspective = norm(realPercpectiveCenter-img.PerspectiveCenter[:,np.newaxis])
    errorRotation = norm(abs(realRotationMatrix)-abs(img.RotationMatrix))
    errorCalibration = norm(realCalibrationMatrix-img.camera.CalibrationMatrix)
    return np.array([errorPerspective[0], errorRotation, errorCalibration])

def printResults(img,realCalibrationMatrix,realPercpectiveCenter,realRotationMatrix):
    print('real rotation matrix')
    print(realRotationMatrix)
    print('compute rotation matrix: ')
    print(img.RotationMatrix)
    print('real perspective center: ')
    print(realPercpectiveCenter)
    print('compute perspective center: ')
    print(img.PerspectiveCenter)
    print('real calibration matrix: ')
    print(realCalibrationMatrix)
    print('compute calibration matrix: ')
    print(img.camera.CalibrationMatrix)




if __name__ == '__main__':

    focalLength = 35  # [mm]
    sensorSize = 25  # [mm]

    # model test1
    sumOfErrors=np.zeros((1,3))
    for i in range(20):
        # sumOfErrors += test1(focalLength, sensorSize, X=0, Y=0, Z=50)
        sumOfErrors += test1(focalLength, sensorSize, X=0, Y=0, Z=100, kappa=40, phi=30, azimuth=15)

        # changing image system
        # sumOfErrors += test1(focalLength, sensorSize, X=0, Y=0, Z=50, system='top left')
        # sumOfErrors +=test1(focalLength, sensorSize, X=0, Y=0, Z=50, system='bottom left')
        # sumOfErrors +=test1(focalLength, sensorSize, X=0, Y=0, Z=50,kappa=30, phi=50, azimuth=30, system='top left')
        # sumOfErrors +=test1(focalLength, sensorSize, X=15, Y=10, Z=50,omega=10,phi=15,kappa=10,  system='bottom left')

        # noise and distance
        # sumOfErrors +=test1(focalLength, sensorSize, X=0, Y=0, Z=50, noiseSize=0.1)
        # sumOfErrors +=test1(focalLength, sensorSize, X=0, Y=0, Z=100, noiseSize=0.1)
        # sumOfErrors +=test1(focalLength, sensorSize, X=0, Y=0, Z=50, kappa=60, phi=15, azimuth=45, noiseSize=0.1)

        # sumOfErrors +=test1(focalLength, sensorSize, X=15, Y=10, Z=50,omega=10,phi=15,kappa=10, system='bottom left')

        # solving single image by non linear transformation
        # sumOfErrors += test2(focalLength, sensorSize, X=0, Y=0, Z=50)
        # sumOfErrors += test2(focalLength, sensorSize, X=0, Y=0, Z=50, system='top left')
        # sumOfErrors += test2(focalLength, sensorSize, X=0, Y=0, Z=50,kappa=30, phi=50, azimuth=30, system='top left')
        # sumOfErrors += test2(focalLength, sensorSize, X=0, Y=0, Z=100,kappa=40, phi=30, azimuth=15)
    print('average errors: perspective center, rotation , calibration')
    print(sumOfErrors/20)