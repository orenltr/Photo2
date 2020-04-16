
import numpy as np

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
        pass  # delete for implementation

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
        pass  # delete for implementation

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
        Points in camera space when principal point is shifted
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

