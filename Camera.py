import numpy as np
import math
from MatrixMethods import *
# import PhotoViewer as pv
# import SingleImage
import matplotlib as plt
import cv2
from glob import glob


class Camera(object):

    def __init__(self, focal_length, principal_point, radial_distortions, decentering_distortions, fiducial_marks, sensorSize):
        """

        Initialize the Camera object

        :param focal_length: focal length of the camera(mm)
        :param principal_point: principle point
        :param radial_distortions: the radial distortion parameters K0, K1, K2
        :param decentering_distortions: decentering distortion parameters P0, P1, P2
        :param fiducial_marks: fiducial marks in camera space

        :type focal_length: double
        :type principal_point: np.array
        :type radial_distortions: np.array
        :type decentering_distortions: np.array
        :type fiducial_marks: np.array

        """
        # private parameters
        self.__focal_length = focal_length
        self.__principal_point = principal_point
        self.__radial_distortions = radial_distortions
        self.__decentering_distortions = decentering_distortions
        self.__fiducial_marks = fiducial_marks
        self.__CalibrationParam = None
        self.__sensorSize = sensorSize

    @property
    def focalLength(self):
        """
        Focal length of the camera

        :return: focal length

        :rtype: float

        """
        return self.__focal_length

    @property
    def sensorSize(self):
        """
        sensor size of the camera

        :return: sensor size

        :rtype: float

        """
        return self.__sensorSize

    @focalLength.setter
    def focalLength(self, val):
        """
        Set the focal length value

        :param val: value for setting

        :type: float

        """

        self.__focal_length = val

    @property
    def fiducialMarks(self):
        """
        Fiducial marks of the camera, by order

        :return: fiducial marks of the camera

        :rtype: np.array nx2

        """

        return self.__fiducial_marks

    @property
    def principalPoint(self):
        """
        Principal point of the camera

        :return: principal point coordinates

        :rtype: np.ndarray

        """

        return self.__principal_point

    @principalPoint.setter
    def principalPoint(self, val):
        """
        Set the principal Point

        :param val: value for setting

        :type: np.ndarray

        """

        self.__principal_point = val

    @property
    def radial_distortions(self):
        """
        radial distortions params

        :return: radial distortions params- k1, k2

        :rtype: np.ndarray

        """

        return self.__radial_distortions

    @radial_distortions.setter
    def radial_distortions(self, val):
        """
        Set the radial distortions params

        :param val: value for setting

        :type: np.ndarray

        """

        self.__radial_distortions = val

    @property
    def CalibrationMatrix(self):

        return np.array([[-self.focalLength, 0, self.principalPoint[0]],
                         [0,-self.focalLength,self.principalPoint[1]],
                         [0,0,1]])

    @property
    def K(self):
        f = self.focalLength
        xp = self.principalPoint[0]
        yp = self.principalPoint[1]
        K = np.array([[-f, 0, xp], [0, -f, yp], [0, 0, 1]])
        self.__k = K
        return self.__k

    def aoutomatic_calibration(self, path):
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((7 * 9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:7].T.reshape(-1, 2)
        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.
        images = glob(path + '\*.jpg')
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (9, 7), None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners)
                # # Draw and display the corners
                # cv.drawChessboardCorners(img, (7, 9), corners2, ret)
                # cv.imshow('img', img)
                # cv.waitKey(500)
        # get calibration params
        gray = cv2.cvtColor(cv2.imread(images[0]), cv2.COLOR_BGR2GRAY)
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        # cv.destroyAllWindows()
        K[0, 0] = -K[0, 0]
        K[1, 1] = -K[1, 1]
        self.focalLength = -K[0, 0]
        self.principalPoint[0] = K[0, 2]
        self.principalPoint[1] = K[1, 2]
        return K


    def CameraToIdealCamera(self, camera_points):
        """
        Transform camera coordinates to an ideal system.

        :param camera_points: set of points in camera space

        :type camera_points: np.array nx2

        :return: fixed point set

        :rtype: np.array nx2

        .. warning::

            This function is empty, need implementation
        """
        # if camera_points.shape[1] ==
        camera_points = self.CorrectionToPrincipalPoint(camera_points)
        camera_points = self.CorrectionToRadialDistortions(camera_points)
        return camera_points


    def IdealCameraToCamera(self, camera_points):
        r"""
        Transform from ideal camera to camera with distortions

        :param camera_points: points in ideal camera space

        :type camera_points: np.array nx2

        :return: corresponding points in image space

        :rtype: np.array nx2

        .. warning::

            This function is empty, need implementation
        """
        camera_points = self.ShiftedPrincipalPoint(camera_points)
        camera_points = self.add_RadialDistortions(camera_points)
        return camera_points


    def ComputeDecenteringDistortions(self, camera_points):
        """
        Compute decentering distortions for given points

        :param camera_points: points in camera space

        :type camera_points: np.array nx2

        :return: decentering distortions: d_x, d_y

        :rtype: tuple of np.array

        .. warning::

            This function is empty, need implementation
        """
        pass  # delete for implementation

    def ComputeRadialDistortions(self, camera_points):
        """
        Compute radial distortions for given points

        :param camera_points: points in camera space

        :type camera_points: np.array nx2

        :return: radial distortions: delta_x, delta_y , delta_r

        :rtype: np.array

        """
        r = np.sqrt((camera_points[:, 0] - self.principalPoint[0])**2 +
                    (camera_points[:, 1] - self.principalPoint[1])**2)
        dx = (camera_points[:, 0] - self.principalPoint[0])*(self.radial_distortions[0]*r**2 + self.radial_distortions[1]*r**4)
        dy = (camera_points[:, 1] - self.principalPoint[1])*(self.radial_distortions[0]*r**2 + self.radial_distortions[1]*r**4)
        dr = np.sqrt(dx**2+dy**2)
        return dx, dy, dr

    def CorrectionToRadialDistortions(self, camera_points):
        """
        Correction to radial distortions for given points
        :param camera_points: points in camera space
        :type camera_points: np.array nx2
        :return: correct points in camera space
        :rtype: np.array
        """
        r = np.sqrt((camera_points[:, 0] - self.principalPoint[0])**2 +
                    (camera_points[:, 1] - self.principalPoint[1])**2)
        dx = (camera_points[:, 0] - self.principalPoint[0])*(self.radial_distortions[0]*r**2 + self.radial_distortions[1]*r**4)
        dy = (camera_points[:, 1] - self.principalPoint[1])*(self.radial_distortions[0]*r**2 + self.radial_distortions[1]*r**4)
        correct_camera_points = np.zeros([len(camera_points), 2])
        correct_camera_points[:,0] = np.array([[camera_points[:, 0] + dx]])
        correct_camera_points[:,1] = np.array([[camera_points[:, 1] + dy]])

        return correct_camera_points

    def add_RadialDistortions(self,camera_points):
        """
        Add radial distortions for given points in synthetic system
         :param camera_points: points in camera space
        :type camera_points: np.array nx2
        :return: points with radial distortions in camera space
        :rtype: np.array
        """
        r = np.sqrt((camera_points[:, 0] - self.principalPoint[0]) ** 2 +
                    (camera_points[:, 1] - self.principalPoint[1]) ** 2)
        dx = (camera_points[:, 0] - self.principalPoint[0]) * (
                    self.radial_distortions[0] * r ** 2 + self.radial_distortions[1] * r ** 4)
        dy = (camera_points[:, 1] - self.principalPoint[1]) * (
                    self.radial_distortions[0] * r ** 2 + self.radial_distortions[1] * r ** 4)

        camera_points[:, 0] -= dx
        camera_points[:, 1] -= dy
        return camera_points


    def CorrectionToPrincipalPoint(self, camera_points):
        """
        Correction to principal point
        :param camera_points: sampled image points
        :type: np.array nx2
        :return: corrected image points
        :rtype: np.array nx2
        .. warning::
            This function is empty, need implementation
        .. note::
            The principal point is an attribute of the camera object, i.e., ``self.principalPoint``
        """
        corrected_camera_points = np.zeros([len(camera_points), 2])
        corrected_camera_points[:, 0] = camera_points[:, 0] - self.principalPoint[0]
        corrected_camera_points[:, 1] = camera_points[:, 1] - self.principalPoint[1]

        return corrected_camera_points



    def ShiftedPrincipalPoint(self, camera_points):
        """
        Points in camera space when principal point is shifted- use in synthetic system
        :param camera_points: sampled image points
        :param t: shifting from principal point

        :type camera_points: np.array nx2
        :type t: np.array 1X2

        :return: shifted image points
        """
        shifted_camera_points = np.zeros([len(camera_points), 2])
        shifted_camera_points[:, 0] = camera_points[:, 0] + self.principalPoint[0]
        shifted_camera_points[:, 1] = camera_points[:, 1] + self.principalPoint[1]

        return shifted_camera_points

    def ComputeObservationVectorForCalibration(self, groundPoints, camera_points, image):
        """
        Compute observation vector for solving the camera calibration

        :param groundPoints: Ground coordinates of the control points

        :type groundPoints: np.array nx3

        :return: Vector l0

        :rtype: np.array nx1
        """
        # create an instance of SingleImage
        # image = SingleImage(self)
        # the points in camera space
        # camera_points = image.GroundToImage(groundPoints)

        n = groundPoints.shape[0]  # number of points

        # Coordinates subtraction
        dX = groundPoints[:, 0] - image.exteriorOrientationParameters[0]
        dY = groundPoints[:, 1] - image.exteriorOrientationParameters[1]
        dZ = groundPoints[:, 2] - image.exteriorOrientationParameters[2]
        dXYZ = np.vstack([dX, dY, dZ])
        rotated_XYZ = np.dot(image.RotationMatrix.T, dXYZ).T

        r = np.sqrt((camera_points[:, 0] - self.principalPoint[0]) ** 2 +
                    (camera_points[:, 1] - self.principalPoint[1]) ** 2)
        dx = (camera_points[:, 0] - self.principalPoint[0]) *\
             (1e-5*self.radial_distortions[0]*r**2 + 1e-10*self.radial_distortions[1]*r**4)
        dy = (camera_points[:, 1] - self.principalPoint[1]) *\
             (1e-5*self.radial_distortions[0]*r**2 + 1e-10*self.radial_distortions[1]*r**4)

        # dx = (camera_points[:, 0] - self.principalPoint[0]) * \
        #      (self.radial_distortions[0] * r ** 2 + self.radial_distortions[1] * r ** 4)
        # dy = (camera_points[:, 1] - self.principalPoint[1]) * \
        #      (self.radial_distortions[0] * r ** 2 + self.radial_distortions[1] * r ** 4)

        l0 = np.empty(n * 2)

        # Computation of the observation vector based on approximate exterior orientation parameters:
        l0[::2] = self.principalPoint[0] - self.focalLength * (rotated_XYZ[:, 0] / rotated_XYZ[:, 2])-dx
        l0[1::2] = self.principalPoint[1] - self.focalLength * (rotated_XYZ[:, 1] / rotated_XYZ[:, 2])-dy

        # l0[::2] = -self.focalLength * (rotated_XYZ[:, 0] / rotated_XYZ[:, 2])
        # l0[1::2] = -self.focalLength * (rotated_XYZ[:, 1] / rotated_XYZ[:, 2])

        return l0

    def ComputeDesignMatrixForCalibration(self, groundPoints, camera_points,image):
        """
            Compute the derivatives of the collinear law (design matrix)

            :param groundPoints: Ground coordinates of the control points

            :type groundPoints: np.array nx3

            :return: The design matrix

            :rtype: np.array nx6

        """
        # initialization for readability
        omega = image.exteriorOrientationParameters[3]
        phi = image.exteriorOrientationParameters[4]
        kappa = image.exteriorOrientationParameters[5]

        # Coordinates subtraction
        dX = groundPoints[:, 0] - image.exteriorOrientationParameters[0]
        dY = groundPoints[:, 1] - image.exteriorOrientationParameters[1]
        dZ = groundPoints[:, 2] - image.exteriorOrientationParameters[2]
        dXYZ = np.vstack([dX, dY, dZ])

        rotationMatrixT = image.RotationMatrix.T
        rotatedG = rotationMatrixT.dot(dXYZ)
        rT1g = rotatedG[0, :]
        rT2g = rotatedG[1, :]
        rT3g = rotatedG[2, :]

        focalBySqauredRT3g = self.focalLength / rT3g ** 2

        dxdg = rotationMatrixT[0, :][None, :] * rT3g[:, None] - rT1g[:, None] * rotationMatrixT[2, :][None, :]
        dydg = rotationMatrixT[1, :][None, :] * rT3g[:, None] - rT2g[:, None] * rotationMatrixT[2, :][None, :]

        dgdX0 = np.array([-1, 0, 0], 'f')
        dgdY0 = np.array([0, -1, 0], 'f')
        dgdZ0 = np.array([0, 0, -1], 'f')

        # Derivatives with respect to X0
        dxdX0 = -focalBySqauredRT3g * np.dot(dxdg, dgdX0)
        dydX0 = -focalBySqauredRT3g * np.dot(dydg, dgdX0)

        # Derivatives with respect to Y0
        dxdY0 = -focalBySqauredRT3g * np.dot(dxdg, dgdY0)
        dydY0 = -focalBySqauredRT3g * np.dot(dydg, dgdY0)

        # Derivatives with respect to Z0
        dxdZ0 = -focalBySqauredRT3g * np.dot(dxdg, dgdZ0)
        dydZ0 = -focalBySqauredRT3g * np.dot(dydg, dgdZ0)

        if image.type == 'real':
            dRTdOmega = Compute3DRotationDerivativeMatrix(omega, phi, kappa, 'omega').T
            dRTdPhi = Compute3DRotationDerivativeMatrix(omega, phi, kappa, 'phi').T
            dRTdKappa = Compute3DRotationDerivativeMatrix(omega, phi, kappa, 'kappa').T

        else:
            dRTdOmega = Compute3DRotationDerivativeMatrix_RzRyRz(omega, phi, kappa, 'azimuth').T
            dRTdPhi = Compute3DRotationDerivativeMatrix_RzRyRz(omega, phi, kappa, 'phi').T
            dRTdKappa = Compute3DRotationDerivativeMatrix_RzRyRz(omega, phi, kappa, 'kappa').T

        gRT3g = dXYZ * rT3g

        # Derivatives with respect to Omega
        dxdOmega = -focalBySqauredRT3g * (dRTdOmega[0, :][None, :].dot(gRT3g) -
                                          rT1g * (dRTdOmega[2, :][None, :].dot(dXYZ)))[0]

        dydOmega = -focalBySqauredRT3g * (dRTdOmega[1, :][None, :].dot(gRT3g) -
                                          rT2g * (dRTdOmega[2, :][None, :].dot(dXYZ)))[0]

        # Derivatives with respect to Phi
        dxdPhi = -focalBySqauredRT3g * (dRTdPhi[0, :][None, :].dot(gRT3g) -
                                        rT1g * (dRTdPhi[2, :][None, :].dot(dXYZ)))[0]

        dydPhi = -focalBySqauredRT3g * (dRTdPhi[1, :][None, :].dot(gRT3g) -
                                        rT2g * (dRTdPhi[2, :][None, :].dot(dXYZ)))[0]

        # Derivatives with respect to Kappa
        dxdKappa = -focalBySqauredRT3g * (dRTdKappa[0, :][None, :].dot(gRT3g) -
                                          rT1g * (dRTdKappa[2, :][None, :].dot(dXYZ)))[0]

        dydKappa = -focalBySqauredRT3g * (dRTdKappa[1, :][None, :].dot(gRT3g) -
                                          rT2g * (dRTdKappa[2, :][None, :].dot(dXYZ)))[0]

        # Derivatives with respect to calibration params
        dxdf = -rT1g / rT3g
        dydf = -rT2g / rT3g

        dxdxp = np.ones(len(groundPoints[:,0]))
        dydxp = np.zeros(len(groundPoints[:,0]))

        dxdyp = np.zeros(len(groundPoints[:,0]))
        dydyp = np.ones(len(groundPoints[:,0]))

        # camera_points = image.GroundToImage(groundPoints)
        r = np.sqrt((camera_points[:, 0] - self.principalPoint[0]) ** 2 +
                    (camera_points[:, 1] - self.principalPoint[1]) ** 2)

        dxdk1 = (-1e-5)*(camera_points[:, 0] - self.principalPoint[0])*r**2
        dxdk2 = (-1e-10)*(camera_points[:, 0] - self.principalPoint[0])*r**4

        dydk1 = (-1e-5)*(camera_points[:, 1] - self.principalPoint[1])*r**2
        dydk2 = (-1e-10)*(camera_points[:, 1] - self.principalPoint[1])*r**4

        # dxdk1 = (camera_points[:, 0] - self.principalPoint[0]) * r ** 2
        # dxdk2 = (camera_points[:, 0] - self.principalPoint[0]) * r ** 4
        #
        # dydk1 = (camera_points[:, 1] - self.principalPoint[1]) * r ** 2
        # dydk2 = (camera_points[:, 1] - self.principalPoint[1]) * r ** 4

        # all derivatives of x and y
        dd = np.array([np.vstack([dxdf, dxdxp, dxdyp, dxdk1, dxdk2, dxdX0, dxdY0, dxdZ0, dxdOmega, dxdPhi, dxdKappa]).T,
                       np.vstack([dydf, dydxp, dydyp, dydk1, dydk2, dydX0, dydY0, dydZ0, dydOmega, dydPhi, dydKappa]).T])

        a = np.zeros((2 * dd[0].shape[0], 11))
        a[0::2] = dd[0]
        a[1::2] = dd[1]

        return a

    def Calibration(self,  camera_points, groundPoints, approx_vals, image,epsilon):
        """
        Compute the calibration parameters
        :param camera_points: points in camera space
        :param groundPoints: corresponding ground points
        :param approx_vals: approximate values
        :param image: SingleImage instance
        :param epsilon: tolerance for the loop

        :type camera_points: points in camera space
        :type groundPoints:
        :type approx_vals:
        :type image: SingleImage instance
        :type epsilon: float
        :return: calibration parameters and accuracies
        """
        # compute control points in camera system using the inner orientation
        # camera_points = image.ImageToCamera(imagePoints)
        lb = camera_points.flatten().T
        X = approx_vals
        dx = np.ones([11, 1]) * 100000
        itr = 0
        # adjustment
        while np.linalg.norm(dx) > epsilon and itr < 100:
            itr += 1
            # update parameters according to approximate values
            self.focalLength = X[0]
            self.principalPoint = np.array([X[1], X[2]])
            self.radial_distortions = np.array([X[3], X[4]])
            image.exteriorOrientationParameters = X[5:]
            l0 = self.ComputeObservationVectorForCalibration(groundPoints, camera_points ,image).T
            L = lb - l0
            A = self.ComputeDesignMatrixForCalibration(groundPoints, camera_points, image)
            N = np.dot(A.T, A)
            U = np.dot(A.T, L)
            dx = np.dot(np.linalg.inv(N), U)
            X = X + dx
        # update  k1, k2
        X[3] = X[3]*1e-5
        X[4] = X[4]*1e-10

        v = A.dot(dx) - L
        # sigma posteriory
        u = 11
        r = len(L) - u
        if r != 0:
            sigma0 = ((v.T).dot(v)) / r
            sigmaX = sigma0 * (np.linalg.inv(N))
        else:
            sigma0 = None
            sigmaX = None

        return X, sigma0, sigmaX, itr

    def cameraToIdealCameraByMatrix(self, cameraPoints):
        """
        transform points from camera system to ideal camera using calibration matrix
        :type cameraPoints: np.array nX2 (x,y)
        :return: idealCameraPoints nX3
        """
        cameraPoints = np.hstack([cameraPoints, np.ones((len(cameraPoints), 1))])
        return np.dot(self.CalibrationMatrix,cameraPoints.T).T



if __name__ == '__main__':

    f0 = 4360.
    xp0 = 2144.5
    yp0 = 1424.5
    K1 = 0
    K2 = 0
    P1 = 0
    P2 = 0

    # define the initial values vector
    cam = Camera(f0, np.array([xp0, yp0]), np.array([K1, K2]),np.array([P1, P2]), None)

