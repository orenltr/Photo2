import numpy as np
import math
from Camera import Camera
from MatrixMethods import Compute3DRotationMatrix, Compute3DRotationDerivativeMatrix
import PhotoViewer as pv

class SingleImage(object):

    def __init__(self, camera):
        """
        Initialize the SingleImage object

        :param camera: instance of the Camera class
        :param points: points in image space

        :type camera: Camera
        :type points: np.array

        """
        self.__camera = camera
        self.__innerOrientationParameters = None
        self.__isSolved = False
        self.__exteriorOrientationParameters = np.array([[0, 0, 0, 0, 0, 0]]).T
        # self.__exteriorOrientationParameters = np.array([0, 0, 0, 0, 0, 0], 'f')
        self.__rotationMatrix = None

    @property
    def innerOrientationParameters(self):
        """
        Inner orientation parameters


        .. warning::

            Can be held either as dictionary or array. For your implementation and decision.

        .. note::

            Do not forget to decide how it is held and document your decision

        :return: inner orinetation parameters

        :rtype: dictionary
        """
        return self.__innerOrientationParameters

    @property
    def camera(self):
        """
        The camera that took the image

        :rtype: Camera

        """
        return self.__camera

    @property
    def exteriorOrientationParameters(self):
        r"""
        Property for the exterior orientation parameters

        :return: exterior orientation parameters in the following order, **however you can decide how to hold them (dictionary or array)**

        .. math::
            exteriorOrientationParameters = \begin{bmatrix} X_0 \\ Y_0 \\ Z_0 \\ \omega \\ \varphi \\ \kappa \end{bmatrix}

        :rtype: np.ndarray or dict
        """
        return self.__exteriorOrientationParameters

    @exteriorOrientationParameters.setter
    def exteriorOrientationParameters(self, parametersArray):
        r"""

        :param parametersArray: the parameters to update the ``self.__exteriorOrientationParameters``

        **Usage example**

        .. code-block:: py

            self.exteriorOrintationParameters = parametersArray

        """
        self.__exteriorOrientationParameters = parametersArray.T

    @property
    def rotationMatrix(self):
        """
        The rotation matrix of the image

        Relates to the exterior orientation
        :return: rotation matrix

        :rtype: np.ndarray (3x3)
        """

        R = Compute3DRotationMatrix(self.exteriorOrientationParameters[3], self.exteriorOrientationParameters[4],
                                    self.exteriorOrientationParameters[5])

        return R

    @property
    def isSolved(self):
        """
        True if the exterior orientation is solved

        :return True or False

        :rtype: boolean
        """
        return self.__isSolved

    @property
    def PerspectiveCenter(self):
        """
        return the perspective center of the first image

        :return: perspective center

        :rtype: np.array (3, )
        """
        return self.exteriorOrientationParameters[0:3]

    def ComputeInnerOrientation(self, imagePoints):
        r"""
        Compute inner orientation parameters

        :param imagePoints: coordinates in image space

        :type imagePoints: np.array nx2

        :return: a dictionary of inner orientation parameters, their accuracies, and the residuals vector

        :rtype: dict

        .. warning::

            This function is empty, need implementation

        .. note::

            - Don't forget to update the ``self.__innerOrinetationParameters`` member. You decide the type
            - The fiducial marks are held within the camera attribute of the object, i.e., ``self.camera.fiducialMarks``
            - return values can be a tuple of dictionaries and arrays.

        **Usage example**

        .. code-block:: py

            fMarks = np.array([[113.010, 113.011],
                       [-112.984, -113.004],
                       [-112.984, 113.004],
                       [113.024, -112.999]])
            img_fmarks = np.array([[-7208.01, 7379.35],
                        [7290.91, -7289.28],
                        [-7291.19, -7208.22],
                        [7375.09, 7293.59]])
            cam = Camera(153.42, np.array([0.015, -0.020]), None, None, fMarks)
            img = SingleImage(camera = cam, points = None)
            inner_parameters, accuracies, residuals = img.ComputeInnerOrientation(img_fmarks)
        """
        if self.camera.fiducialMarks == 'no fiducials':  # case of digital camera
            pixel_size = 0.0024  # [mm]
            a1 = 1/pixel_size
            b2 = -1/pixel_size
            a2 = 0
            b1 = 0
            a0 = self.camera.principalPoint[0]/pixel_size
            b0 = self.camera.principalPoint[1]/pixel_size
            self.__innerOrientationParameters = {'a0': a0, 'a1': a1, 'a2': a2, 'b0': b0, 'b1': b1, 'b2': b2,
                    'V': 0, 'sigma0': 0, 'sigmaX': 0}
            return {'a0': a0, 'a1': a1, 'a2': a2, 'b0': b0, 'b1': b1, 'b2': b2,
                    'V': 0, 'sigma0': 0, 'sigmaX': 0}
        else:


            # observation vector
            l = np.matrix(imagePoints).flatten('F').T

            # fiducial marks - camera system
            fc = self.camera.fiducialMarks

            # A matrix (16X6)
            j = len(imagePoints[:, 0])
            A = np.zeros((len(l), 6))
            for i in range(j):
                A[i, 0:3] = np.array([1, fc[i, 0], fc[i, 1]])
                A[i + j, 3:] = np.array([1, fc[i, 0], fc[i, 1]])

            # N matrix
            N = (A.T).dot(A)
            # U vector
            U = (A.T).dot(l)
            # adjusted variables
            X = (np.linalg.inv(N)).dot(U)
            # v remainders vector
            v = A.dot(X)-l

            # sigma posteriory
            u = 6
            r = len(l)-u
            sigma0 = ((v.T).dot(v)) / r
            sigmaX = sigma0[0, 0] * (np.linalg.inv(N))
            # update field
            self.__innerOrientationParameters = {'a0': X[0, 0], 'a1': X[1, 0], 'a2': X[2, 0], 'b0': X[3, 0], 'b1': X[4, 0],
                                                 'b2': X[5, 0],
                                                 'V': v, 'sigma0': sigma0[0, 0], 'sigmaX': sigmaX}

            return {'a0': X[0, 0], 'a1': X[1, 0], 'a2': X[2, 0], 'b0': X[3, 0], 'b1': X[4, 0], 'b2': X[5, 0],
                    'V': v, 'sigma0': sigma0[0, 0], 'sigmaX': sigmaX}



    def ComputeGeometricParameters(self):
        """
        Computes the geometric inner orientation parameters

        :return: geometric inner orientation parameters

        :rtype: dict

        .. warning::

           This function is empty, need implementation

        .. note::

            The algebraic inner orinetation paramters are held in ``self.innerOrientatioParameters`` and their type
            is according to what you decided when initialized them

        """
        # algebraic inner orinetation paramters
        x = self.__innerOrientationParameters
        tx = x['a0']
        ty = x['b0']
        tetha = np.arctan((x['b1']/x['b2']))
        gamma = np.arctan((x['a1']*np.sin(tetha)+x['a2']*np.cos(tetha))
                          /(x['b1']*np.sin(tetha)+x['b2']*np.cos(tetha)))
        sx = x['a1']*np.cos(tetha)-x['a2']*np.sin(tetha)
        sy = (x['a1']*np.sin(tetha)+x['a2']*np.cos(tetha))/(np.sin(gamma))

        return {'translationX': tx, 'translationY': ty, 'rotationAngle': tetha,
                'scaleFactorX': sx, 'scaleFactorY': sy, 'shearAngle': gamma}





    def ComputeInverseInnerOrientation(self):
        """
        Computes the parameters of the inverse inner orientation transformation

        :return: parameters of the inverse transformation

        :rtype: dict

        .. warning::

            This function is empty, need implementation

        .. note::

            The inner orientation algebraic parameters are held in ``self.innerOrientationParameters``
            their type is as you decided when implementing
        """
        inner = self.__innerOrientationParameters
        matrix = np.array([[inner['a1'],inner['a2']],[inner['b1'],inner['b2']]])
        # inverse matrix
        inv_matrix = np.linalg.inv(matrix)
        return {'a0*': -inner['a0'], 'a1*': inv_matrix[0, 0], 'a2*': inv_matrix[0, 1],
                'b0*': -inner['b0'], 'b1*': inv_matrix[1, 0], 'b2*': inv_matrix[1, 1]}


    def CameraToImage(self, cameraPoints):
        """
        Transforms camera points to image points

        :param cameraPoints: camera points

        :type cameraPoints: np.array nx2

        :return: corresponding Image points

        :rtype: np.array nx2


        .. warning::

            This function is empty, need implementation

        .. note::

            The inner orientation parameters required for this function are held in ``self.innerOrientationParameters``

        **Usage example**

        .. code-block:: py

            fMarks = np.array([[113.010, 113.011],
                       [-112.984, -113.004],
                       [-112.984, 113.004],
                       [113.024, -112.999]])
            img_fmarks = np.array([[-7208.01, 7379.35],
                        [7290.91, -7289.28],
                        [-7291.19, -7208.22],
                        [7375.09, 7293.59]])
            cam = Camera(153.42, np.array([0.015, -0.020]), None, None, fMarks)
            img = SingleImage(camera = cam, points = None)
            img.ComputeInnerOrientation(img_fmarks)
            pts_image = img.Camera2Image(fMarks)

        """
        # get algebric parameters
        inner = self.__innerOrientationParameters

        imgPoints = np.zeros((len(cameraPoints[:,0]),2))
        for i in range(len(cameraPoints[:,0])):
            imgPoints[i,0] = inner['a0']+inner['a1']*cameraPoints[i,0]+inner['a2']*cameraPoints[i,1]
            imgPoints[i,1] = inner['b0']+inner['b1']*cameraPoints[i,0]+inner['b2']*cameraPoints[i,1]

        return imgPoints


    def ImageToCamera(self, imagePoints):
        """

        Transforms image points to ideal camera points

        :param imagePoints: image points

        :type imagePoints: np.array nx2

        :return: corresponding camera points

        :rtype: np.array nx2

        .. warning::

            This function is empty, need implementation

        .. note::

            The inner orientation parameters required for this function are held in ``self.innerOrientationParameters``


        **Usage example**

        .. code-block:: py

            fMarks = np.array([[113.010, 113.011],
                       [-112.984, -113.004],
                       [-112.984, 113.004],
                       [113.024, -112.999]])
            img_fmarks = np.array([[-7208.01, 7379.35],
                        [7290.91, -7289.28],
                        [-7291.19, -7208.22],
                        [7375.09, 7293.59]])
            cam = Camera(153.42, np.array([0.015, -0.020]), None, None, fMarks)
            img = SingleImage(camera = cam, points = None)
            img.ComputeInnerOrientation(img_fmarks)
            pts_camera = img.Image2Camera(img_fmarks)

        """
        # get the inverse inner orientation param
        inv_param = self.ComputeInverseInnerOrientation()

        camPoints = np.zeros((len(imagePoints[:, 0]), 2))
        for i in range(len(imagePoints[:, 0])):
            camPoints[i, 0] = inv_param['a1*']*(imagePoints[i,0]+inv_param['a0*']) + inv_param['a2*']*(imagePoints[i,1]+inv_param['b0*'])
            camPoints[i, 1] = inv_param['b1*']*(imagePoints[i,0]+inv_param['a0*']) + inv_param['b2*']*(imagePoints[i,1]+inv_param['b0*'])

        return camPoints


    def ComputeExteriorOrientation(self, imagePoints, groundPoints, epsilon):
        """
        Compute exterior orientation parameters.

        This function can be used in conjecture with ``self.__ComputeDesignMatrix(groundPoints)`` and ``self__ComputeObservationVector(imagePoints)``

        :param imagePoints: image points
        :param groundPoints: corresponding ground points

            .. note::

                Angles are given in radians

        :param epsilon: threshold for convergence criteria

        :type imagePoints: np.array nx2
        :type groundPoints: np.array nx3
        :type epsilon: float

        :return: Exterior orientation parameters: (X0, Y0, Z0, omega, phi, kappa), their accuracies, and residuals vector. *The orientation parameters can be either dictionary or array -- to your decision*

        :rtype: dict


        .. warning::

           - This function is empty, need implementation
           - Decide how the parameters are held, don't forget to update documentation

        .. note::

            - Don't forget to update the ``self.exteriorOrientationParameters`` member (every iteration and at the end).
            - Don't forget to call ``cameraPoints = self.ImageToCamera(imagePoints)`` to correct the coordinates              that are sent to ``self.__ComputeApproximateVals(cameraPoints, groundPoints)``
            - return values can be a tuple of dictionaries and arrays.

        **Usage Example**

        .. code-block:: py

            img = SingleImage(camera = cam)
            grdPnts = np.array([[201058.062, 743515.351, 243.987],
                        [201113.400, 743566.374, 252.489],
                        [201112.276, 743599.838, 247.401],
                        [201166.862, 743608.707, 248.259],
                        [201196.752, 743575.451, 247.377]])
            imgPnts3 = np.array([[-98.574, 10.892],
                         [-99.563, -5.458],
                         [-93.286, -10.081],
                         [-99.904, -20.212],
                         [-109.488, -20.183]])
            img.ComputeExteriorOrientation(imgPnts3, grdPnts, 0.3)


        """
        # compute control points in camera system using the inner orientation
        camera_points = self.ImageToCamera(imagePoints)

        # compute approximate values for exteriror orientation using conformic transformation
        self.ComputeApproximateVals(camera_points, groundPoints)
        lb = camera_points.flatten().T

        dx = np.ones([6, 1]) * 100000
        while np.linalg.norm(dx) > epsilon:
            X = self.exteriorOrientationParameters.T
            l0 = self.ComputeObservationVector(groundPoints).T
            L = lb - l0
            A = self.ComputeDesignMatrix(groundPoints)
            N = np.dot(A.T, A)
            U = np.dot(A.T, L)
            dx = np.dot(np.linalg.inv(N), U)
            X = X + dx
            self.exteriorOrientationParameters = X.T

        v = A.dot(dx) - L

        # sigma posteriory
        u = 6
        r = len(L) - u
        if r != 0:
            sigma0 = ((v.T).dot(v)) / r
            sigmaX = sigma0 * (np.linalg.inv(N))
        else:
            sigma0 = None
            sigmaX = None

        return self.exteriorOrientationParameters, sigma0, sigmaX



    def GroundToImage(self, groundPoints):
        """
        Transforming ground points to image points

        :param groundPoints: ground points [m]

        :type groundPoints: np.array nx3

        :return: corresponding Image points

        :rtype: np.array nx2

        """
        X0_1 = self.exteriorOrientationParameters[0]
        Y0_1 = self.exteriorOrientationParameters[1]
        Z0_1 = self.exteriorOrientationParameters[2]
        O1 = np.array([X0_1, Y0_1, Z0_1]).T
        R1 = self.rotationMatrix
        x1 = np.zeros((len(groundPoints), 1))
        y1 = np.zeros((len(groundPoints), 1))
        f = self.camera.focalLength

        for i in range(len(groundPoints)):
            lamda1 = -f / (np.dot(R1.T[2], (groundPoints[i] - O1).T))  # scale first image
            x1[i] = lamda1 * np.dot(R1.T[0], (groundPoints[i] - O1).T)
            y1[i] = lamda1 * np.dot(R1.T[1], (groundPoints[i] - O1).T)
            camera_points1 = np.vstack([x1.T, y1.T]).T
            # img_points1 = self.CameraToImage(camera_points1)
            img_points1 = camera_points1
        return img_points1

    def ImageToRay(self, imagePoints):
        """
        Transforms Image point to a Ray in world system

        :param imagePoints: coordinates of an image point

        :type imagePoints: np.array nx2

        :return: Ray direction in world system

        :rtype: np.array nx3

        .. warning::

           This function is empty, need implementation

        .. note::

            The exterior orientation parameters needed here are called by ``self.exteriorOrientationParameters``
        """
        pass  # delete after implementations

    def ImageToGround_GivenZ(self, imagePoints, Z_values):
        """
        Compute corresponding ground point given the height in world system

        :param imagePoints: points in image space
        :param Z_values: height of the ground points


        :type Z_values: np.array nx1
        :type imagePoints: np.array nx2
        :type eop: np.ndarray 6x1

        :return: corresponding ground points

        :rtype: np.ndarray

        .. warning::

             This function is empty, need implementation

        .. note::

            - The exterior orientation parameters needed here are called by ``self.exteriorOrientationParameters``
            - The focal length can be called by ``self.camera.focalLength``

        **Usage Example**

        .. code-block:: py


            imgPnt = np.array([-50., -33.])
            img.ImageToGround_GivenZ(imgPnt, 115.)

        """
        camera_points = self.ImageToCamera(imagePoints)

        # exterior orientation parameters

        omega = self.exteriorOrientationParameters[3]
        phi = self.exteriorOrientationParameters[4]
        kapa = self.exteriorOrientationParameters[5]
        X0 = self.exteriorOrientationParameters[0]
        Y0 = self.exteriorOrientationParameters[1]
        Z0 = self.exteriorOrientationParameters[2]

        Z = Z_values
        R = Compute3DRotationMatrix(omega, phi, kapa)
        X = np.zeros(len(Z))
        Y = np.zeros(len(Z))

        for i in range(len(Z)):
            xyf = np.array([camera_points[i, 0] - self.camera.principalPoint[0],
                            camera_points[i, 1] - self.camera.principalPoint[1],
                            -self.camera.focalLength])  # camera point vector
            lamda = (Z[i] - Z0) / (np.dot(R[2], xyf))  # scale
            X[i] = X0 + lamda * np.dot(R[0], xyf)
            Y[i] = Y0 + lamda * np.dot(R[1], xyf)

        return np.vstack([X, Y, Z]).T






    # ---------------------- Private methods ----------------------

    def ComputeApproximateVals(self, cameraPoints, groundPoints):
        """
        Compute exterior orientation approximate values via 2-D conform transformation

        :param cameraPoints: points in image space (x y)
        :param groundPoints: corresponding points in world system (X, Y, Z)

        :type cameraPoints: np.ndarray [nx2]
        :type groundPoints: np.ndarray [nx3]

        :return: Approximate values of exterior orientation parameters
        :rtype: np.ndarray or dict

        .. note::

            - ImagePoints should be transformed to ideal camera using ``self.ImageToCamera(imagePoints)``. See code below
            - The focal length is stored in ``self.camera.focalLength``
            - Don't forget to update ``self.exteriorOrientationParameters`` in the order defined within the property
            - return values can be a tuple of dictionaries and arrays.

        .. warning::

           - This function is empty, need implementation
           - Decide how the exterior parameters are held, don't forget to update documentation



        """
        # Find approximate values

        # partial derevative matrix
        # order: a b c d
        A = np.array([[1, 0, cameraPoints[0, 0], cameraPoints[0, 1]],
                      [0, 1, cameraPoints[0, 1], -1 * (cameraPoints[0, 0])],
                      [1, 0, cameraPoints[1, 0], cameraPoints[1, 1]],
                      [0, 1, cameraPoints[1, 1], -1 * (cameraPoints[1, 0])]])

        b = np.array([[groundPoints[0, 0]],
                      [groundPoints[0, 1]],
                      [groundPoints[1, 0]],
                      [groundPoints[1, 1]]])
        X = np.dot(np.linalg.inv(A), b)
        X0 = X[0]
        Y0 = X[1]
        # kapa = np.arctan(-(X[3] / X[2]))
        kapa = np.arctan2(-X[3], X[2])
        # kapa = 1.73
        lamda = np.sqrt(X[2] ** 2, X[3] ** 2)
        # Z0 = groundPoints[0, 2] + lamda * self.camera.focalLength
        Z0 = groundPoints[0, 2] + lamda * self.camera.focalLength
        omega = 0
        phi = 0
        self.exteriorOrientationParameters = np.array([X0, Y0, Z0, omega, phi, kapa])

        # self.exteriorOrientationParameters = {'X0': X0, 'Y0': Y0, 'Z0': Z0, 'lamda': lamda,
        #         'kapa': kapa, 'omega': omega, 'phi': phi}
        # return {'X0': X0, 'Y0': Y0, 'Z0': Z0, 'lamda': lamda,
        #         'kapa': kapa, 'omega': omega, 'phi': phi}

    def ComputeObservationVector(self, groundPoints):
        """
        Compute observation vector for solving the exterior orientation parameters of a single image
        based on their approximate values

        :param groundPoints: Ground coordinates of the control points

        :type groundPoints: np.array nx3

        :return: Vector l0

        :rtype: np.array nx1
        """

        n = groundPoints.shape[0]  # number of points

        # Coordinates subtraction
        dX = groundPoints[:,0] - self.exteriorOrientationParameters[0]
        dY = groundPoints[:,1] - self.exteriorOrientationParameters[1]
        dZ = groundPoints[:,2] - self.exteriorOrientationParameters[2]
        dXYZ = np.vstack([dX, dY, dZ])
        rotated_XYZ = np.dot(self.rotationMatrix.T, dXYZ).T

        l0 = np.empty(n * 2)

        # Computation of the observation vector based on approximate exterior orientation parameters:
        l0[::2] = -self.camera.focalLength * rotated_XYZ[:, 0] / rotated_XYZ[:, 2]
        l0[1::2] = -self.camera.focalLength * rotated_XYZ[:, 1] / rotated_XYZ[:, 2]

        return l0

    def ComputeDesignMatrix(self, groundPoints):
        """
            Compute the derivatives of the collinear law (design matrix)

            :param groundPoints: Ground coordinates of the control points

            :type groundPoints: np.array nx3

            :return: The design matrix

            :rtype: np.array nx6

        """
        # initialization for readability
        omega = self.exteriorOrientationParameters[3]
        phi = self.exteriorOrientationParameters[4]
        kappa = self.exteriorOrientationParameters[5]

        # Coordinates subtraction
        dX = groundPoints[:, 0] - self.exteriorOrientationParameters[0]
        dY = groundPoints[:, 1] - self.exteriorOrientationParameters[1]
        dZ = groundPoints[:, 2] - self.exteriorOrientationParameters[2]
        dXYZ = np.vstack([dX, dY, dZ])

        rotationMatrixT = self.rotationMatrix.T
        rotatedG = rotationMatrixT.dot(dXYZ)
        rT1g = rotatedG[0, :]
        rT2g = rotatedG[1, :]
        rT3g = rotatedG[2, :]

        focalBySqauredRT3g = self.camera.focalLength / rT3g ** 2

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

        dRTdOmega = Compute3DRotationDerivativeMatrix(omega, phi, kappa, 'omega').T
        dRTdPhi = Compute3DRotationDerivativeMatrix(omega, phi, kappa, 'phi').T
        dRTdKappa = Compute3DRotationDerivativeMatrix(omega, phi, kappa, 'kappa').T

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

        # all derivatives of x and y
        dd = np.array([np.vstack([dxdX0, dxdY0, dxdZ0, dxdOmega, dxdPhi, dxdKappa]).T,
                       np.vstack([dydX0, dydY0, dydZ0, dydOmega, dydPhi, dydKappa]).T])

        a = np.zeros((2 * dd[0].shape[0], 6))
        a[0::2] = dd[0]
        a[1::2] = dd[1]

        return a

    def drawSingleImage(self, modelPoints,scale,ax):
        """
        draws the rays to the modelpoints from the perspective center of the two images

        :param modelPoints: points in the model system [ model units]
        :param ax: axes of the plot
        :type modelPoints: np.array nx3
        :type ax: plot axes

        :return: none

        """
        pixel_size = 0.0000024 # [m]

        # images coordinate systems
        pv.drawOrientation(self.rotationMatrix, self.PerspectiveCenter, 1,ax)

        # images frames
        pv.drawImageFrame(self.camera.sensorSize/1000*scale, self.camera.sensorSize/1000*scale,
        self.rotationMatrix, self.PerspectiveCenter,self.camera.focalLength/1000,1,ax)

        # draw rays from perspective centers to model points
        pv.drawRays(modelPoints,self.PerspectiveCenter,ax)


if __name__ == '__main__':
    fMarks = np.array([[113.010, 113.011],
                       [-112.984, -113.004],
                       [-112.984, 113.004],
                       [113.024, -112.999]])
    img_fmarks = np.array([[-7208.01, 7379.35],
                           [7290.91, -7289.28],
                           [-7291.19, -7208.22],
                           [7375.09, 7293.59]])
    cam = Camera(153.42, np.array([0.015, -0.020]), None, None, fMarks)
    img = SingleImage(camera = cam)
    print(img.ComputeInnerOrientation(img_fmarks))

    print(img.ImageToCamera(img_fmarks))

    print(img.CameraToImage(fMarks))

    GrdPnts = np.array([[5100.00, 9800.00, 100.00]])
    print(img.GroundToImage(GrdPnts))

    imgPnt = np.array([23.00, 25.00])
    print(img.ImageToRay(imgPnt))

    imgPnt2 = np.array([-50., -33.])
    print(img.ImageToGround_GivenZ(imgPnt2, 115.))

    # grdPnts = np.array([[201058.062, 743515.351, 243.987],
    #                     [201113.400, 743566.374, 252.489],
    #                     [201112.276, 743599.838, 247.401],
    #                     [201166.862, 743608.707, 248.259],
    #                     [201196.752, 743575.451, 247.377]])
    #
    # imgPnts3 = np.array([[-98.574, 10.892],
    #                      [-99.563, -5.458],
    #                      [-93.286, -10.081],
    #                      [-99.904, -20.212],
    #                      [-109.488, -20.183]])
    #
    # intVal = np.array([200786.686, 743884.889, 954.787, 0, 0, 133 * np.pi / 180])
    #
    # print img.ComputeExteriorOrientation(imgPnts3, grdPnts, intVal)
