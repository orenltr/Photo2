from numpy import cos, sin, array, dot, atleast_2d, abs, trunc, any, finfo, get_printoptions, set_printoptions, ndarray, sign
import numpy.core.arrayprint as arrayprint
from json import dumps, dump, load, loads
import contextlib
import math
import  numpy as np

def Compute3DRotationMatrix_RzRyRz(azimuth, phi, kappa):
    """
    Computes a 3x3 rotation matrix defined by the 3 given angles

    :param azimuth: Rotation angle around the optic axis (radians)
    :param phi: Rotation angle around the y-axis (radians)
    :param kappa: Rotation angle around the z-axis (radians)

    :type azimuth: float
    :type phi: float
    :type kappa: float


    :return: The corresponding 3D rotation matrix
    :rtype: array  (3x3)

    **Examples**

    1. Rotation matrix with rotation around the optic-axis only:

        .. code-block:: py

            from numpy import pi
            Compute3DRotationMatrix(30 * pi / 180.0, 0.0, 0.0)

    2. Rotation matrix with rotation around the y-axis only:

        .. code-block:: py

            from numpy import pi
            Compute3DRotationMatrix(0.0, 30 * pi / 180.0, 0.0)

    3. Rotation matrix with rotation around the z-axis only:

        .. code-block:: py

            from numpy import pi
            Compute3DRotationMatrix(0.0, 0.0, 30 * pi / 180.0)

    4. Rotation matrix with rotation of 5, 2, and 30 around the optic, y, and z axes respectively:

        .. code-block:: py

            from numpy import pi
            Compute3DRotationMatrix(5 * pi / 180.0, 2 * pi / 180.0, 30 * pi / 180.0)

    """
    # Rotation matrix around the z-axis
    rAzimuth = array([[cos(azimuth), -sin(azimuth), 0],
                    [sin(azimuth), cos(azimuth), 0],
                    [0, 0, 1]], 'f')

    # Rotation matrix around the y-axis
    rPhi = array([[cos(phi), 0, sin(phi)],
                  [0, 1, 0],
                  [-sin(phi), 0, cos(phi)]], 'f')

    # Rotation matrix around the z-axis
    rKappa = array([[cos(kappa), -sin(kappa), 0],
                    [sin(kappa), cos(kappa), 0],
                    [0, 0, 1]], 'f')

    return dot(dot(rAzimuth, rPhi), rKappa)


def Compute3DRotationDerivativeMatrix_RzRyRz(azimuth, phi, kappa, var):
    """
    Computing the derivative of the 3D rotaion matrix defined by the angles according to one of the angles

    :param azimuth: Rotation angle around the optic axis (radians)
    :param phi: Rotation angle around the y-axis (radians)
    :param kappa: Rotation angle around the z-axis (radians)
    :param var: Name of the angle to compute the derivative by (azimuth/phi/kappa) (string)

    :type azimuth: float
    :type phi: float
    :type kappa: float
    :type var: str

    :return: The corresponding derivative matrix (3x3, ndarray). If var is not one of euler angles, the method will return None


    **Examples**

    1. Derivative matrix with respect to :math:\azimuth:

        .. code-block:: py

            from numpy import pi
            Compute3DRotationDerivativeMatrix(5 * pi / 180.0, 2 * pi / 180.0, 30 * pi / 180.0, 'omega')

    2. Derivative matrix with respect to :math:\phi:

        .. code-block:: py

            from numpy import pi
            Compute3DRotationDerivativeMatrix(5 * pi / 180.0, 2 * pi / 180.0, 30 * pi / 180.0, 'phi')

    3. Derivative matrix with respect to :math:\kappa:

        .. code-block:: py

            from numpy import pi
            Compute3DRotationDerivativeMatrix(5 * pi / 180.0, 2 * pi / 180.0, 30 * pi / 180.0, 'kappa')

    """
    # Rotation matrix around the z-axis
    rAzimut = array([[cos(azimuth), -sin(azimuth), 0],
                     [sin(azimuth), cos(azimuth), 0],
                     [0, 0, 1]], 'f')

    # Rotation matrix around the y-axis
    rPhi = array([[cos(phi), 0, sin(phi)],
                  [0, 1, 0],
                  [-sin(phi), 0, cos(phi)]], 'f')

    # Rotation matrix around the z-axis
    rKappa = array([[cos(kappa), -sin(kappa), 0],
                    [sin(kappa), cos(kappa), 0],
                    [0, 0, 1]], 'f')

    # Computing the derivative matrix based on requested parameter
    if var == 'azimuth':
        d = array([[0, -1, 0], [1, 0, 0], [0, 0, 0]], 'f')
        res = dot(d, dot(rAzimut, dot(rPhi, rKappa)))
    elif var == 'phi':
        d = array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]], 'f')
        res = dot(rAzimut, dot(d, dot(rPhi, rKappa)))
    elif var == 'kappa':
        d = array([[0, -1, 0], [1, 0, 0], [0, 0, 0]], 'f')
        res = dot(rAzimut, dot(rPhi, dot(d, rKappa)))
    else:
        res = None

    return res


def Compute3DRotationMatrix(omega, phi, kappa):
    """
    Computes a 3x3 rotation matrix defined by euler angles given in radians

    :param omega: Rotation angle around the x-axis (radians)
    :param phi: Rotation angle around the y-axis (radians)
    :param kappa: Rotation angle around the z-axis (radians)

    :type omega: float
    :type phi: float
    :type kappa: float


    :return: The corresponding 3D rotation matrix
    :rtype: array  (3x3)

    **Examples**

    1. Rotation matrix with rotation around the x-axis only:

        .. code-block:: py

            from numpy import pi
            Compute3DRotationMatrix(30 * pi / 180.0, 0.0, 0.0)

    2. Rotation matrix with rotation around the y-axis only:

        .. code-block:: py

            from numpy import pi
            Compute3DRotationMatrix(0.0, 30 * pi / 180.0, 0.0)

    3. Rotation matrix with rotation around the z-axis only:

        .. code-block:: py

            from numpy import pi
            Compute3DRotationMatrix(0.0, 0.0, 30 * pi / 180.0)

    4. Rotation matrix with rotation of 5, 2, and 30 around the x, y, and z axes respectively:

        .. code-block:: py

            from numpy import pi
            Compute3DRotationMatrix(5 * pi / 180.0, 2 * pi / 180.0, 30 * pi / 180.0)

    """
    # Rotation matrix around the x-axis
    rOmega = array([[1, 0, 0],
                    [0, cos(omega), -sin(omega)],
                    [0, sin(omega), cos(omega)]], 'f')

    # Rotation matrix around the y-axis
    rPhi = array([[cos(phi), 0, sin(phi)],
                  [0, 1, 0],
                  [-sin(phi), 0, cos(phi)]], 'f')

    # Rotation matrix around the z-axis
    rKappa = array([[cos(kappa), -sin(kappa), 0],
                    [sin(kappa), cos(kappa), 0],
                    [0, 0, 1]], 'f')

    return dot(dot(rOmega, rPhi), rKappa)


def Compute3DRotationDerivativeMatrix(omega, phi, kappa, var):
    r"""
    Computing the derivative of the 3D rotaion matrix defined by the euler angles according to one of the angles

    :param omega: Rotation angle around the x-axis (radians)
    :param phi: Rotation angle around the y-axis (radians)
    :param kappa: Rotation angle around the z-axis (radians)
    :param var: Name of the angle to compute the derivative by (omega/phi/kappa) (string)

    :type omega: float
    :type phi: float
    :type kappa: float
    :type var: str

    :return: The corresponding derivative matrix (3x3, ndarray). If var is not one of euler angles, the method will return None


    **Examples**

    1. Derivative matrix with respect to :math:\omega:

        .. code-block:: py

            from numpy import pi
            Compute3DRotationDerivativeMatrix(5 * pi / 180.0, 2 * pi / 180.0, 30 * pi / 180.0, 'omega')

    2. Derivative matrix with respect to :math:\varphi:

        .. code-block:: py

            from numpy import pi
            Compute3DRotationDerivativeMatrix(5 * pi / 180.0, 2 * pi / 180.0, 30 * pi / 180.0, 'phi')

    3. Derivative matrix with respect to :math:\kappa:

        .. code-block:: py

            from numpy import pi
            Compute3DRotationDerivativeMatrix(5 * pi / 180.0, 2 * pi / 180.0, 30 * pi / 180.0, 'kappa')

    """
    # Rotation matrix around the x-axis
    rOmega = array([[1, 0, 0],
                    [0, cos(omega), -sin(omega)],
                    [0, sin(omega), cos(omega)]], 'f')

    # Rotation matrix around the y-axis
    rPhi = array([[cos(phi), 0, sin(phi)],
                  [0, 1, 0],
                  [-sin(phi), 0, cos(phi)]], 'f')

    # Rotation matrix around the z-axis
    rKappa = array([[cos(kappa), -sin(kappa), 0],
                    [sin(kappa), cos(kappa), 0],
                    [0, 0, 1]], 'f')

    # Computing the derivative matrix based on requested parameter
    if (var == 'omega'):
        d = array([[0, 0, 0], [0, 0, -1], [0, 1, 0]], 'f')
        res = dot(d, dot(rOmega, dot(rPhi, rKappa)))
    elif (var == 'phi'):
        d = array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]], 'f')
        res = dot(rOmega, dot(d, dot(rPhi, rKappa)))
    elif (var == 'kappa'):
        d = array([[0, -1, 0], [1, 0, 0], [0, 0, 0]], 'f')
        res = dot(rOmega, dot(rPhi, dot(d, rKappa)))
    else:
        res = None

    return res


def ComputeSkewMatrixFromVector(vec):
    """
    Computing the 3x3 skew matrix from a given vector

    :param vec: A 3D vector to convert

    :type vec: array

    :return: The corresponding skew matrix (3x3 ndarray). If the given vector is not a 3D one, the method will return None

    **Example**

    .. code-block:: py

        ComputeSkewMatrixFromVector(array([1, 2, 3]))


    """
    # Checking if the size of the vector is three
    if (len(vec) != 3):
        return None

    vec = vec.reshape((-1,))  # reshaping the vector to a 1-d array

    return array([[0, -vec[2], vec[1]],
                  [vec[2], 0, -vec[0]],
                  [-vec[1], vec[0], 0]], 'f')


def PrintMatrix(matrixToPrint, title=None, decimalPrecision=2, integralPrecision=6):
    """
    Method for Printing a matrix with defined preciesion

    :param matrixToPrint: The matrix to be printed (2d array)
    :param title: A heading to print before printing the matrix (string)
    :param decimalPrecision: The number of digits after the decimal points
    :param integralPrecision: The number of digits before the decimal points

    :type matrixToPrint: np.array
    :type title: str
    :type decimalPrecision: int
    :type integralPrecision: int

    **Example**

    .. code-block:: py

            PrintMatrix(ComputeSkewMatrixFromVector(array([1, 2, 3])), 'Skew Matrix')

    """
    if matrixToPrint is None:
        return

    if matrixToPrint.ndim == 1:
        matrixToPrint = matrixToPrint.reshape((-1, 1))

    if (title != None):
        print(title)

    print('   dimensions: ', matrixToPrint.shape[0], 'by', matrixToPrint.shape[1])
    for i in range(0, matrixToPrint.shape[0]):
        for j in range(0, matrixToPrint.shape[1]):
            frmt = '%{}.{}f\t'.format(integralPrecision, decimalPrecision)
            print(frmt % matrixToPrint[i, j], end='')
        print('')
    print('')


@contextlib.contextmanager
def __printoptions(strip_zeros=True, **kwargs):
    """
    http://stackoverflow.com/questions/2891790/pretty-printing-of-numpy-array

    :param strip_zeros: Boolean variable indicating whether to strip zeros at the end of float numbers
    :param kwargs: Additional parameters for printing

    :type strip_zeros: bool

    :return:
    """
    origcall = arrayprint.FloatFormat.__call__

    # noinspection PyUnresolvedReferences
    def __call__(self, x, strip_zeros=strip_zeros):
        return origcall.__call__(self, x, strip_zeros)

    arrayprint.FloatFormat.__call__ = __call__
    original = get_printoptions()
    set_printoptions(**kwargs)
    yield
    set_printoptions(**original)
    arrayprint.FloatFormat.__call__ = origcall


def MatrixToLatex(a, name, units='-', decimals=4, tab='  '):
    """
    Create a latex string of a given 2D array

    :param a: matrix to print
    :param name: The name of the matrix
    :param units: The units of the matrix (optional, default: no-unit)
    :param decimals: The number of digits after the decimal point (optional, default: 4 digits)
    :param tab: The delimiter to break between columns (optional, default: blank space)

    :type a: np.array
    :type name: str
    :type units: str
    :type decimals: int
    :type tab: str

    :return: A string containing the matrix in latex format

    **Example**

    .. code-block:: py

        print(MatrixToLatex(ComputeSkewMatrixFromVector(array([1, 2, 3])), 'B', '[-]'))

    """

    array = atleast_2d(a)

    # Determine the number of digits left of the decimal.
    # Grab integral part, convert to string, and get maximum length.
    # This does not handle negative signs appropriately since -0.5 goes to 0.
    # So we make sure it never counts a negative sign by taking the abs().
    integral = abs(trunc(array.flat).astype(int))
    left_digits = max(map(len, map(str, integral)))

    # Adjust for negative signs.
    if any(array < 0):
        left_digits += 1

    # Set decimal digits to 0 if data are not floats.
    try:
        finfo(array.dtype)
    except ValueError:
        decimals = 0

    # Align the elements on the decimal, making room for largest element.
    matrixName = r"{{\bf {name}}}_{{{size}}}".format(name=name, size=a.shape)

    # Specify that we want all columns to have the same column type.

    # Build the lines in the array.
    #
    # In general, we could just use the rounded array and map(str, row),
    # but NumPy strips trailing zeros on floats (undesirably). So we make
    # use of the context manager to prevent that.
    options = {
        'precision': decimals,
        'suppress': True,
        'strip_zeros': False,
    }
    with __printoptions(**options):
        lines = []
        for row in array:
            # Strip [ and ], remove newlines, and split on whitespace
            elements = row.__str__()[1:-1].replace('\n', '').split()
            line = [tab, ' & '.join(elements), r' \\']
            lines.append(''.join(line))

    # Remove the \\ on the last line.
    lines[-1] = lines[-1][:-3]

    # Create the LaTeX code
    subs = {'matrixName': matrixName, 'lines': '\n'.join(lines), 'units': units}
    template = r"""\begin{{equation*}} {matrixName} =

    \end{{equation*}} \begin{{bmatrix}}
    {lines}
    \end{{bmatrix}} _{{[{units}]}}
    """

    return template.format(**subs)

def isclose(x, y, rtol=1.e-5, atol=1.e-8):
    return abs(x - y) <= atol + rtol * abs(y)

def euler_angles_from_rotation_matrix(R):
    '''
    From a paper by Gregory G. Slabaugh (undated),
    "Computing Euler angles from a rotation matrix
    '''
    phi = 0.0
    if isclose(R[2, 0], -1.0):
        theta = math.pi / 2.0
        psi = math.atan2(R[0, 1], R[0, 2])
    elif isclose(R[2, 0], 1.0):
        theta = -math.pi / 2.0
        psi = math.atan2(-R[0, 1], -R[0, 2])
    else:
        theta = -math.asin(R[2, 0])
        cos_theta = math.cos(theta)
        psi = math.atan2(R[2, 1] / cos_theta, R[2, 2] / cos_theta)
        phi = math.atan2(R[1, 0] / cos_theta, R[0, 0] / cos_theta)
    return psi, theta, phi

def findSignMat(calibrationMatrix):
    signMat = np.eye(3)
    if calibrationMatrix[0,0] > 0:
        signMat[0,0] = signMat[0,0]*-1
    if calibrationMatrix[1,1] > 0:
        signMat[1,1] = signMat[1,1]*-1
    if calibrationMatrix[2,2] < 0:
        signMat[2,2] = signMat[2,2]*-1
    return signMat

if __name__ == "__main__":
    from numpy import pi

    # Examples for creating rotation matrices
    print('Rotation matrix with rotation around the x-axis only:')
    print(Compute3DRotationMatrix_RzRyRz(30 * pi / 180.0, 0.0, 0.0))
    print('Rotation matrix with rotation around the y-axis only:')
    print(Compute3DRotationMatrix_RzRyRz(0.0, 30 * pi / 180.0, 0.0))
    print('Rotation matrix with rotation around the z-axis only:')
    print(Compute3DRotationMatrix_RzRyRz(0.0, 0.0, 30 * pi / 180.0))
    print('Rotation matrix with rotation of 5, 2, and 30 around the x, y, and z axes respectively:')
    print(Compute3DRotationMatrix_RzRyRz(5 * pi / 180.0, 2 * pi / 180.0, 30 * pi / 180.0))

    # Examples for creating derivative matrices
    print('The derivative matrices:')
    print('    With respect to omega:')
    print(Compute3DRotationDerivativeMatrix_RzRyRz(5 * pi / 180.0, 2 * pi / 180.0, 30 * pi / 180.0, 'omega'))
    print('    With respect to phi:')
    print(Compute3DRotationDerivativeMatrix_RzRyRz(5 * pi / 180.0, 2 * pi / 180.0, 30 * pi / 180.0, 'phi'))
    print('    With respect to kappa:')
    print(Compute3DRotationDerivativeMatrix_RzRyRz(5 * pi / 180.0, 2 * pi / 180.0, 30 * pi / 180.0, 'kappa'))
    print('    Example of sending a wrong argument:')
    print(Compute3DRotationDerivativeMatrix_RzRyRz(5 * pi / 180.0, 2 * pi / 180.0, 30 * pi / 180.0, 'Omega'))

    # Example for creating a skew matrix
    print('Skew matrix of the vector (1, 2, 3):')
    print(ComputeSkewMatrixFromVector(array([1, 2, 3])))

    # Example for using PrintMatrix
    PrintMatrix(ComputeSkewMatrixFromVector(array([1, 2, 3])), 'Skew Matrix')

    # Example for Converting a 2D array to latex
    print(MatrixToLatex(ComputeSkewMatrixFromVector(array([1, 2, 3])), 'B', '[-]'))

