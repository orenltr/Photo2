from Camera import Camera
from SingleImage import SingleImage
from MatrixMethods import Compute3DRotationMatrix, Compute3DRotationDerivativeMatrix, ComputeSkewMatrixFromVector
import numpy as np
import PhotoViewer as pv
from matplotlib import pyplot as plt
from numpy import linalg as la
class ImagePair(object):

    def __init__(self, image1, image2):
        """
        Initialize the ImagePair class
        :param image1: First image
        :param image2: Second image
        """
        self.__image1 = image1
        self.__image2 = image2
        self.__relativeOrientationImage1 = np.array([[0, 0, 0, 0, 0, 0]]).T  # The relative orientation of the first image
        self.__relativeOrientationImage2 = None  # The relative orientation of the second image
        self.__absoluteOrientation = None
        self.__isSolved = False  # Flag for the relative orientation

    @property
    def isSolved(self):
        """
        Flag for the relative orientation
        returns True if the relative orientation is solved, otherwise it returns False

        :return: boolean, True or False values
        """
        return self.__isSolved

    @property
    def RotationMatrix_Image1(self):
        """
        return the rotation matrix of the first image

        :return: rotation matrix

        :rtype: np.array 3x3
        """
        return Compute3DRotationMatrix(self.__relativeOrientationImage1[3], self.__relativeOrientationImage1[4],
                                       self.__relativeOrientationImage1[5])

    @property
    def RotationMatrix_Image2(self):
        """
        return the rotation matrix of the second image

        :return: rotation matrix

        :rtype: np.array 3x3
        """
        return Compute3DRotationMatrix(self.__relativeOrientationImage2[3], self.__relativeOrientationImage2[4],
                                       self.__relativeOrientationImage2[5])

    @property
    def PerspectiveCenter_Image1(self):
        """
        return the perspective center of the first image

        :return: perspective center

        :rtype: np.array (3, )
        """
        return self.__relativeOrientationImage1[0:3]

    @property
    def PerspectiveCenter_Image2(self):
        """
        return the perspective center of the second image

        :return: perspective center

        :rtype: np.array (3, )
        """
        return self.__relativeOrientationImage2[0:3]


    def ImagesToGround(self, imagePoints1, imagePoints2, Method):
        """
        Computes ground coordinates of homological points

        :param imagePoints1: points in image 1
        :param imagePoints2: corresponding points in image 2
        :param Method: method to use for the ray intersection, three options exist: geometric, vector, Collinearity

        :type imagePoints1: np.array nx2
        :type imagePoints2: np.array nx2
        :type Method: string

        :return: ground points, their accuracies.

        :rtype: dict

        .. warning::

            This function is empty, need implementation


        **Usage example**

        .. code-block:: py

            camera = Camera(152, None, None, None, None)
            image1 = SingleImage(camera)
            image2 = SingleImage(camera)

            imagePoints1 = np.array([[-4.83,7.80],
                                [-4.64, 134.86],
                                [5.39,-100.80],
                                [4.58,55.13],
                                [98.73,9.59],
                                [62.39,128.00],
                                [67.90,143.92],
                                    [56.54,-85.76]])
            imagePoints2 = np.array([[-83.17,6.53],
                                 [-102.32,146.36],
                                 [-62.84,-102.87],
                                 [-97.33,56.40],
                                 [-3.51,14.86],
                                 [-27.44,136.08],
                                 [-23.70,152.90],
                                 [-8.08,-78.07]])

            new = ImagePair(image1, image2)

            new.ImagesToGround(imagePoints1, imagePoints2, 'geometric'))

        """
        image1 = self.__image1
        image2 = self.__image2
        camera_points1 = image1.ImageToCamera(imagePoints1)
        camera_points2 = image2.ImageToCamera(imagePoints2)

        X0_1 = image1.exteriorOrientationParameters[0]
        Y0_1 = image1.exteriorOrientationParameters[1]
        Z0_1 = image1.exteriorOrientationParameters[2]
        X0_2 = image2.exteriorOrientationParameters[0]
        Y0_2 = image2.exteriorOrientationParameters[1]
        Z0_2 = image2.exteriorOrientationParameters[2]
        O1 = np.array([X0_1, Y0_1, Z0_1]).T
        O2 = np.array([X0_2, Y0_2, Z0_2]).T
        R1 = image1.rotationMatrix
        R2 = image2.rotationMatrix

        dO = O2 - O1

        X = np.zeros([len(imagePoints1), 3])  # optimal point
        # V = np.zeros([6,1])  # error
        sigma = np.zeros([len(imagePoints1), 3])
        d = np.zeros([len(imagePoints1), 3])

        if Method == 'vector':
            for i in range(len(imagePoints1)):
                # compute rays
                v1 = np.dot(R1, np.array([camera_points1[i, 0] - self.PerspectiveCenter_Image1[0],
                                          camera_points1[i, 1] - self.PerspectiveCenter_Image1[1],
                                          -image1.camera.focalLength]))
                v2 = np.dot(R2, np.array([camera_points2[i, 0] - self.PerspectiveCenter_Image2[0],
                                          camera_points2[i, 1] - self.PerspectiveCenter_Image2[1],
                                          -image2.camera.focalLength]))

                mat1 = np.array([[v1.dot(v1), -v1.dot(v2)], [v1.dot(v2), -v2.dot(v2)]])
                mat2 = np.array([dO.dot(v1), dO.dot(v2)])
                lamda = np.linalg.inv(mat1).dot(mat2)

                # closest points on the rays
                f = O1 + lamda[0] * v1
                g = O2 + lamda[1] * v2

                X[i] = (f + g) / 2
                d[i] = np.abs(f - g)
            return X, d
        elif Method == 'geometric':
            for i in range(len(imagePoints1)):
                # compute rays
                v1 = np.dot(R1, np.array([camera_points1[i, 0] - image1.camera.principalPoint[0],
                                          camera_points1[i, 1] - image1.camera.principalPoint[1],
                                          -image1.camera.focalLength]))
                v2 = np.dot(R2, np.array([camera_points2[i, 0] - image2.camera.principalPoint[0],
                                          camera_points2[i, 1] - image2.camera.principalPoint[1],
                                          -image2.camera.focalLength]))
                v1_normelized = v1 / np.linalg.norm(v1)
                v2_normelized = v2 / np.linalg.norm(v2)
                # adjustment
                A = np.vstack((np.eye(3) - np.outer(v1_normelized, v1_normelized.T),
                               np.eye(3) - np.outer(v2_normelized, v2_normelized.T)))
                l = np.vstack((A[0:3].dot(O1.T), A[3:].dot(O2.T)))
                X[i] = (np.linalg.inv(np.dot(A.T, A)).dot(A.T).dot(l)).T
                V = np.dot(A, X[i].T) - l[:, 0]
                # sigma[i] = (V.T*V)/(6-3)
            return X, V

    def RotationLevelModel(self, constrain1, constrain2):
        """
        Compute rotation matrix from the current model coordinate system to the other coordinate system

        :param constrain1: constrain of the first axis
        :param constrain2: constrain of the second axis

        :type constrain1: tuple
        :type constrain2: tuple

        :return: rotation matrix

        :rtype: np.array 3x3

        .. note::

            The vector data included in the two constrains must be normalized

            The two constrains should be given to two different axises, if not return identity matrix

        """
        if constrain1[0] == constrain2[0]:
            return np.eye(3)
        if constrain1[0] == 'x':
            x = constrain1[1]/la.norm(constrain1[1])
            if constrain2[0] == 'y':
                y = constrain2[1] / la.norm(constrain2[1])
                z = np.cross(x,y)
        if constrain1[0] == 'x':
            x = constrain1[1]/la.norm(constrain1[1])
            if constrain2[0] == 'z':
                z = constrain2[1] / la.norm(constrain2[1])
                y = np.cross(x,z)
        if constrain1[0] == 'y':
            y = constrain1[1]/la.norm(constrain1[1])
            if constrain2[0] == 'x':
                x = constrain2[1] / la.norm(constrain2[1])
                z = np.cross(x,y)
        if constrain1[0] == 'y':
            y = constrain1[1]/la.norm(constrain1[1])
            if constrain2[0] == 'z':
                z = constrain2[1] / la.norm(constrain2[1])
                x = np.cross(z,y)
        if constrain1[0] == 'z':
            z = constrain1[1]/la.norm(constrain1[1])
            if constrain2[0] == 'x':
                x = constrain2[1] / la.norm(constrain2[1])
                y = np.cross(x,z)
        if constrain1[0] == 'z':
            z = constrain1[1]/la.norm(constrain1[1])
            if constrain2[0] == 'y':
                y = constrain2[1] / la.norm(constrain2[1])
                x = np.cross(z,y)
        self.__absoluteOrientation = np.vstack((x,y,z)).T
        return np.vstack((x,y,z)).T


    def ModelTransformation(self, modelPoints, scale):
        """
        Transform model from the current coordinate system to other coordinate system

        :param modelPoints: coordinates in current model space
        :param scale: scale between the two coordinate systems

        :type modelPoints: np.array nx3
        :type scale: float

        :return: corresponding coordinates in the other coordinate system

        :rtype: np.array nx3

        .. warning::

            This function is empty, needs implementation

        """
        ground_points = np.dot(self.__absoluteOrientation,modelPoints.T) * scale
        return ground_points.T
    def ComputeDependentRelativeOrientation(self, imagePoints1, imagePoints2, initialValues):
        """
         Compute relative orientation parameters

        :param imagePoints1: points in the first image [m"m]
        :param imagePoints2: corresponding points in image 2(homology points) nx2 [m"m]
        :param initialValues: approximate values of relative orientation parameters

        :type imagePoints1: np.array nx2
        :type imagePoints2: np.array nx2
        :type initialValues: np.array (6L,)

        :return: relative orientation parameters.

        :rtype: np.array 6x1 [by,bz,omega,phi,kapa]

        .. warning::

            Can be held either as dictionary or array. For your implementation and decision.

        .. note::

            Do not forget to decide how it is held and document your decision


        **Usage example**

        .. code-block:: py

            camera = Camera(152, None, None, None, None)
            image1 = SingleImage(camera)
            image2 = SingleImage(camera)

            imagePoints1 = np.array([[-4.83,7.80],
                                [-4.64, 134.86],
                                [5.39,-100.80],
                                [4.58,55.13],
                                [98.73,9.59],
                                [62.39,128.00],
                                [67.90,143.92],
                                [56.54,-85.76]])
            imagePoints2 = np.array([[-83.17,6.53],
                                 [-102.32,146.36],
                                 [-62.84,-102.87],
                                 [-97.33,56.40],
                                 [-3.51,14.86],
                                 [-27.44,136.08],
                                 [-23.70,152.90],
                                 [-8.08,-78.07]])
            new = ImagePair(image1, image2)

            new.ComputeDependentRelativeOrientation(imagePoints1, imagePoints2, np.array([1, 0, 0, 0, 0, 0])))

        """

        # transfer points to camera system
        cameraPoints1 = self.__image1.ImageToCamera(imagePoints1)
        cameraPoints2 = self.__image2.ImageToCamera(imagePoints2)
        temp_f1 = np.zeros([cameraPoints1.shape[0]]) - self.__image1.camera.focalLength
        temp_f2 = np.zeros([cameraPoints2.shape[0]]) - self.__image2.camera.focalLength
        cameraPoints1 = np.c_[cameraPoints1, temp_f1]
        cameraPoints2 = np.c_[cameraPoints2, temp_f2]


        # initial values
        x0 = initialValues.T[1:]

        r = cameraPoints1.shape[0]  # number of conditions equations
        u = 5  # number of variables

        i = 0
        dx = 100
        while np.linalg.norm(dx) > 0.001 and i < 100:
            i = i + 1
            A, B, W = self.Build_A_B_W(cameraPoints1, cameraPoints2, x0)
            M = B.dot(B.T)
            N = (A.T).dot((np.linalg.inv(M)).dot(A))
            U = A.T.dot((np.linalg.inv(M)).dot(W))
            dx = np.array([-np.linalg.inv(N).dot(U)])
            x0 = x0 + dx.T
            v = np.array([-B.T.dot(np.linalg.inv(M)).dot(W)]).T
        # lb = np.zeros([cameraPoints1.shape[0] * 4, 1])
        # lb[0:len(lb):4, 0] = cameraPoints1[:, 0]
        # lb[1:len(lb):4, 0] = cameraPoints1[:, 1]
        # lb[2:len(lb):4, 0] = cameraPoints2[:, 0]
        # lb[3:len(lb) + 1:4, 0] = cameraPoints2[:, 1]
        # la = lb + v
        sigma_squere = (v.T.dot(v)) / (r - u)
        sigmaX = sigma_squere * np.linalg.inv(N)

        x0 = np.vstack((np.array([1]), x0))
        self.__relativeOrientationImage2 = x0

        return x0, np.diag(sigmaX)

    def Build_A_B_W(self, cameraPoints1, cameraPoints2, x):
        """
        Function for computing the A and B matrices and vector w.
        :param cameraPoints1: points in the first camera system
        :param ImagePoints2: corresponding homology points in the second camera system
        :param x: initialValues vector by, bz, omega, phi, kappa ( bx=1)

        :type cameraPoints1: np.array nx3
        :type cameraPoints2: np.array nx3
        :type x: np.array (5,1)

        :return: A ,B matrices, w vector

        :rtype: tuple
        """
        numPnts = cameraPoints1.shape[0]  # Number of points

        dbdy = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
        dbdz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])

        dXdx = np.array([1, 0, 0])
        dXdy = np.array([0, 1, 0])

        # Compute rotation matrix and it's derivatives
        rotationMatrix2 = Compute3DRotationMatrix(x[2, 0], x[3, 0], x[4, 0])
        dRdOmega = Compute3DRotationDerivativeMatrix(x[2, 0], x[3, 0], x[4, 0], 'omega')
        dRdPhi = Compute3DRotationDerivativeMatrix(x[2, 0], x[3, 0], x[4, 0], 'phi')
        dRdKappa = Compute3DRotationDerivativeMatrix(x[2, 0], x[3, 0], x[4, 0], 'kappa')

        # Create the skew matrix from the vector [bx, by, bz]
        bMatrix = ComputeSkewMatrixFromVector(np.array([1, x[0, 0], x[1, 0]]))

        # Compute A matrix; the coplanar derivatives with respect to the unknowns by, bz, omega, phi, kappa
        A = np.zeros((numPnts, 5))
        A[:, 0] = np.diag(
            np.dot(cameraPoints1,
                   np.dot(dbdy, np.dot(rotationMatrix2, cameraPoints2.T))))  # derivative in respect to by
        A[:, 1] = np.diag(
            np.dot(cameraPoints1,
                   np.dot(dbdz, np.dot(rotationMatrix2, cameraPoints2.T))))  # derivative in respect to bz
        A[:, 2] = np.diag(
            np.dot(cameraPoints1, np.dot(bMatrix, np.dot(dRdOmega, cameraPoints2.T))))  # derivative in respect to omega
        A[:, 3] = np.diag(
            np.dot(cameraPoints1, np.dot(bMatrix, np.dot(dRdPhi, cameraPoints2.T))))  # derivative in respect to phi
        A[:, 4] = np.diag(
            np.dot(cameraPoints1, np.dot(bMatrix, np.dot(dRdKappa, cameraPoints2.T))))  # derivative in respect to kappa

        # Compute B matrix; the coplanar derivatives in respect to the observations, x', y', x'', y''.
        B = np.zeros((numPnts, 4 * numPnts))
        k = 0
        for i in range(numPnts):
            p1vec = cameraPoints1[i, :]
            p2vec = cameraPoints2[i, :]
            B[i, k] = np.dot(dXdx, np.dot(bMatrix, np.dot(rotationMatrix2, p2vec)))
            B[i, k + 1] = np.dot(dXdy, np.dot(bMatrix, np.dot(rotationMatrix2, p2vec)))
            B[i, k + 2] = np.dot(np.dot(p1vec, np.dot(bMatrix, rotationMatrix2)), dXdx)
            B[i, k + 3] = np.dot(np.dot(p1vec, np.dot(bMatrix, rotationMatrix2)), dXdy)
            k += 4

        # w vector
        w = np.diag(np.dot(cameraPoints1, np.dot(bMatrix, np.dot(rotationMatrix2, cameraPoints2.T))))

        return A, B, w

    def ImagesToModel(self, imagePoints1, imagePoints2, Method):
        """
        Mapping points from image space to model space

        :param imagePoints1: points from the first image
        :param imagePoints2: points from the second image
        :param Method: method for intersection

        :type imagePoints1: np.array nx2
        :type imagePoints2: np.array nx2
        :type Method: string

        :return: corresponding model points
        :rtype: np.array nx3


        .. warning::

            This function is empty, need implementation

        .. note::

            One of the images is a reference, orientation of this image must be set.


        """
        camera_points1 = self.__image1.ImageToCamera(imagePoints1)
        camera_points2 = self.__image2.ImageToCamera(imagePoints2)

        if Method == 'vector':
            X, v = self.vectorIntersction(camera_points1, camera_points2)
        elif Method == 'geometric':
            X, v = self.geometricIntersection(camera_points1, camera_points2)
        else:
            print('no such method')

        return X, v

    def GroundToImage(self, groundPoints):
        """
        Transforming ground points to image points

        :param groundPoints: ground points [m]

        :type groundPoints: np.array nx3

        :return: corresponding Image points

        :rtype: np.array nx2

        """
        X0_1 = self.__image1.exteriorOrientationParameters[0]
        Y0_1 = self.__image1.exteriorOrientationParameters[1]
        Z0_1 = self.__image1.exteriorOrientationParameters[2]
        X0_2 = self.__image2.exteriorOrientationParameters[0]
        Y0_2 = self.__image2.exteriorOrientationParameters[1]
        Z0_2 = self.__image2.exteriorOrientationParameters[2]
        O1 = np.array([X0_1, Y0_1, Z0_1]).T
        O2 = np.array([X0_2, Y0_2, Z0_2]).T
        R1 = self.__image1.rotationMatrix
        R2 = self.__image2.rotationMatrix
        x1 = np.zeros((len(groundPoints), 1))
        x2 = np.zeros((len(groundPoints), 1))
        y1 = np.zeros((len(groundPoints), 1))
        y2 = np.zeros((len(groundPoints), 1))
        f = self.__image1.camera.focalLength

        for i in range(len(groundPoints)):
            lamda1 = -f / (np.dot(R1.T[2], (groundPoints[i] - O1).T))  # scale first image
            lamda2 = -f / (np.dot(R2.T[2], (groundPoints[i] - O2).T))  # scale second image
            x1[i] = lamda1 * np.dot(R1.T[0], (groundPoints[i] - O1).T)
            y1[i] = lamda1 * np.dot(R1.T[1], (groundPoints[i] - O1).T)
            x2[i] = lamda2 * np.dot(R2.T[0], (groundPoints[i] - O2).T)
            y2[i] = lamda2 * np.dot(R2.T[1], (groundPoints[i] - O2).T)
            camera_points1 = np.vstack([x1.T, y1.T]).T
            camera_points2 = np.vstack([x2.T, y2.T]).T
            img_points1 = self.__image1.CameraToImage(camera_points1)
            img_points2 = self.__image2.CameraToImage(camera_points2)
        return img_points1, img_points2

    def geometricIntersection(self, cameraPoints1, cameraPoints2, system='relative'):
        """
        Ray Intersection based on geometric calculations.

        :param cameraPoints1: points in the first image
        :param cameraPoints2: corresponding points in the second image
        :param system: world or relative

        :type cameraPoints1: np.array nx3
        :type cameraPoints2: np.array nx3
        :type system: str 'world' or 'relative'

        :return: lambda1, lambda2 scalars

        :rtype: np.array nx2

        .. warning::

            This function is empty, need implementation

        """
        image1 = self.__image1
        image2 = self.__image2

        if system == 'relative':
            X0_1 = self.PerspectiveCenter_Image1[0]
            Y0_1 = self.PerspectiveCenter_Image1[1]
            Z0_1 = self.PerspectiveCenter_Image1[2]
            X0_2 = self.PerspectiveCenter_Image2[0]
            Y0_2 = self.PerspectiveCenter_Image2[1]
            Z0_2 = self.PerspectiveCenter_Image2[2]
            O1 = np.array([X0_1, Y0_1, Z0_1]).T
            O2 = np.array([X0_2, Y0_2, Z0_2]).T
            R1 = self.RotationMatrix_Image1
            R2 = self.RotationMatrix_Image2
        elif system == 'world':
            X0_1 = image1.PerspectiveCenter[0]
            Y0_1 = image1.PerspectiveCenter[1]
            Z0_1 = image1.PerspectiveCenter[2]
            X0_2 = image2.PerspectiveCenter[0]
            Y0_2 = image2.PerspectiveCenter[1]
            Z0_2 = image2.PerspectiveCenter[2]
            O1 = np.array([X0_1, Y0_1, Z0_1]).T
            O2 = np.array([X0_2, Y0_2, Z0_2]).T
            R1 = image1.RotationMatrix
            R2 = image2.RotationMatrix
        else:
            print('system need to be "relative" or "world"')
            return

        X = np.zeros([len(cameraPoints1), 3])  # optimal point
        Sigma = np.zeros([len(cameraPoints1), 3])   # variance matrix


        for i in range(len(cameraPoints1)):

            # compute rays
            v1 = np.dot(R1, np.array([[cameraPoints1[i, 0],
                                       cameraPoints1[i, 1],
                                       -image1.camera.focalLength]]).T)
            v2 = np.dot(R2, np.array([[cameraPoints2[i, 0],
                                       cameraPoints2[i, 1],
                                       -image2.camera.focalLength]]).T)
            v1_normelized = v1 / np.linalg.norm(v1)
            v2_normelized = v2 / np.linalg.norm(v2)

            # adjustment
            A = np.vstack((np.eye(3) - np.outer(v1_normelized, v1_normelized.T),
                           np.eye(3) - np.outer(v2_normelized, v2_normelized.T)))
            l = np.vstack((A[0:3].dot(O1.T), A[3:].dot(O2.T)))
            N = np.dot(A.T, A)
            N_inv = np.linalg.inv(N)
            U = np.dot(A.T,l)
            X[i] = np.dot(N_inv,U).T

            # precision
            V = np.dot(A, X[i].T) - l[:, 0]
            sigma0 = np.dot(V.T,V)/(6-3)
            Sigma[i] = np.diag(sigma0*N_inv)
        Sigma = np.sqrt(Sigma)
        return X, Sigma

    def vectorIntersction(self, cameraPoints1, cameraPoints2):
        """
        Ray Intersection based on vector calculations.

        :param cameraPoints1: points in image space
        :param cameraPoints2: corresponding image points

        :type cameraPoints1: np.array nx
        :type cameraPoints2: np.array nx


        :return: lambda1, lambda2 scalars

        :rtype: np.array nx2

        .. warning::

            This function is empty, need implementation

        """
        image1 = self.__image1
        image2 = self.__image2


        O1 = self.PerspectiveCenter_Image1
        O2 = self.PerspectiveCenter_Image2
        R1 = self.RotationMatrix_Image1
        R2 = self.RotationMatrix_Image2

        dO = O2 - O1

        X = np.zeros([len(cameraPoints1), 3])  # optimal point
        # V = np.zeros([6,1])  # error
        sigma = np.zeros([len(cameraPoints1), 3])
        d = np.zeros([len(cameraPoints1), 3])

        for i in range(len(cameraPoints1)):
            # compute rays

            v1 = np.dot(R1, np.array([[cameraPoints1[i, 0],
                                       cameraPoints1[i, 1],
                                       -image1.camera.focalLength]]).T)
            v2 = np.dot(R2, np.array([[cameraPoints2[i, 0],
                                       cameraPoints2[i, 1],
                                       -image2.camera.focalLength]]).T)

            mat1 = np.array([[v1.T.dot(v1)[0,0], -v1.T.dot(v2)[0,0]], [v1.T.dot(v2)[0,0], -v2.T.dot(v2)[0,0]]])
            mat2 = np.array([[dO.T.dot(v1)[0,0], dO.T.dot(v2)[0,0]]])
            lamda = np.linalg.inv(mat1).dot(mat2.T)

            # closest points on the rays
            f = O1 + lamda[0,0] * v1
            g = O2 + lamda[1,0] * v2

            X[i] = ((f + g) / 2).T
            d[i] = np.abs(f - g).T
        return X, d / 2

    def CollinearityIntersection(self, cameraPoints1, cameraPoints2):
        """
        Ray intersection based on the collinearity principle

        :param cameraPoints1: points in image space
        :param cameraPoints2: corresponding image points

        :type cameraPoints1: np.array nx2
        :type cameraPoints2: np.array nx2

        :return: corresponding ground points

        :rtype: np.array nx3

        .. warning::

            This function is empty, need implementation

        """

    def drawImagePair(self, modelPoints,ax):
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
        pv.drawOrientation(self.RotationMatrix_Image1, self.PerspectiveCenter_Image1, 1,ax)
        pv.drawOrientation(self.RotationMatrix_Image2, self.PerspectiveCenter_Image2, 1,ax)

        # images frames
        pv.drawImageFrame((self.__image1.camera.principalPoint[0]*2), self.__image1.camera.principalPoint[1]*2,
        self.RotationMatrix_Image1, self.PerspectiveCenter_Image1,self.__image1.camera.focalLength,0.1,ax)
        pv.drawImageFrame(self.__image2.camera.principalPoint[0] * 2,
                          self.__image2.camera.principalPoint[1] * 2,
                          self.RotationMatrix_Image2, self.PerspectiveCenter_Image2,
                          self.__image2.camera.focalLength, 0.1,ax)

        # draw rays from perspective centers to model points
        pv.drawRays(modelPoints,self.PerspectiveCenter_Image1,ax)
        pv.drawRays(modelPoints,self.PerspectiveCenter_Image2,ax)

    def drawModel(self, modelPoints,ax):
        """
        draws the rays to the modelpoints from the perspective center of the two images

        :param modelPoints: points in the model system [ model units]
        :param ax: axes of the plot
        :type modelPoints: np.array nx3
        :type ax: plot axes

        :return: none

        """

        ax.scatter(modelPoints[:11, 0], modelPoints[:11, 1], modelPoints[:11, 2], c='black', marker='^')
        ax.scatter(modelPoints[11:, 0], modelPoints[11:, 1], modelPoints[11:, 2], c='red', marker='o')

        x = modelPoints[:, 0]
        y = modelPoints[:, 1]
        z = modelPoints[:, 2]
        connectpoints(x, y, z, 12, 13)
        connectpoints(x, y, z, 13, 24)
        connectpoints(x, y, z, 12, 14)
        connectpoints(x, y, z, 13, 15)
        connectpoints(x, y, z, 15, 14)
        connectpoints(x, y, z, 15, 16)
        connectpoints(x, y, z, 14, 17)
        connectpoints(x, y, z, 16, 17)
        connectpoints(x, y, z, 15, 23)
        connectpoints(x, y, z, 24, 23)
        connectpoints(x, y, z, 16, 19)
        connectpoints(x, y, z, 17, 18)
        connectpoints(x, y, z, 22, 20)
        connectpoints(x, y, z, 20, 21)
        connectpoints(x, y, z, 20, 19)
        connectpoints(x, y, z, 19, 18)
        connectpoints(x, y, z, 18, 21)
        set_axes_equal(ax)


def connectpoints(x, y, z, p1, p2):

    x1, x2 = x[p1-1], x[p2-1]
    y1, y2 = y[p1-1], y[p2-1]
    z1, z2 = z[p1-1], z[p2-1]
    plt.plot([x1, x2], [y1, y2],[z1, z2], 'k-')




def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


if __name__ == '__main__':
    camera = Camera(152, None, None, None, None)
    image1 = SingleImage(camera)
    image2 = SingleImage(camera)
    leftCamPnts = np.array([[-4.83, 7.80],
                            [-4.64, 134.86],
                            [5.39, -100.80],
                            [4.58, 55.13],
                            [98.73, 9.59],
                            [62.39, 128.00],
                            [67.90, 143.92],
                            [56.54, -85.76]])
    rightCamPnts = np.array([[-83.17, 6.53],
                             [-102.32, 146.36],
                             [-62.84, -102.87],
                             [-97.33, 56.40],
                             [-3.51, 14.86],
                             [-27.44, 136.08],
                             [-23.70, 152.90],
                             [-8.08, -78.07]])
    new = ImagePair(image1, image2)

    print(new.ComputeDependentRelativeOrientation(leftCamPnts, rightCamPnts, np.array([1, 0, 0, 0, 0, 0])))
