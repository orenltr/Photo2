from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from MatrixMethods import Compute3DRotationMatrix


def drawRays(listOfPoints, x0,ax):
    """
    Draw 3d lines representing the rays coming out from the perspective center to the ground points

    :param listOfPoints: 3d coordinates of the points in model/world space
    :param x0: perspective center of the camera

    :type listOfPoints: np.array nx2
    :type: x0: np.array 3x1

    :return: None
    """

    for p in listOfPoints:
        p = np.reshape(p, (3, 1))
        x, y, z = [x0[0, 0], p[0, 0]], [x0[1, 0], p[1, 0]], [x0[2, 0], p[2, 0]]
        ax.plot(x, y, zs=z, color='black')

def drawImageFrame(imageWidth, imageHeight, R, x0, f, scale,ax):
    """
    Draw image frame in the 3d coordinate system

    :param imageWidth: width of the image [m]
    :param imageHeight: height of the image [m]
    :param R: rotation matrix
    :param x0: perspective center 3d coordinates
    :param f: focal length [m]
    :param scale: scale

    :type imageWidth: float
    :type imageHeight: float
    :type R: np.array 3x3
    :type x0: np.array 3x1
    :type f: float
    :type scale: float

    :return: None
    """

    tl, tr, bl, br = calcFrameEdgesIn3d(R, x0, f, scale, imageWidth, imageHeight)
    x = [tl[0, 0], tr[0, 0], br[0, 0], bl[0, 0], tl[0,0]]
    y = [tl[1, 0], tr[1, 0], br[1, 0], bl[1, 0], tl[1,0]]
    z = [tl[2, 0], tr[2, 0], br[2, 0], bl[2, 0], tl[2,0]]

    ax.scatter(x, y, z, c='r', s=50)
    ax.plot(x, y, z, color='r')




def calcFrameEdgesIn3d(R, x0, f, scale, imageWidth, imageHeight):
    """
    Find the image corners in 3d system, using a simple version of the co-linear role

    :param R: rotation matrix
    :param x0: perspective center of the camera in the 3d coordinate system
    :param f: focal length [m]
    :param scale: scale
    :param imageWidth: image frame width [m]
    :param imageHeight: image frame height[m]

    :type R: np.array 3x3
    :type x0: np.array 3x1
    :type f: float
    :type scale: float
    :type imageWidth: float
    :type imageHeight: float

    :return: None
    """

    # this section defines each point
    tl = np.array([[-imageWidth / 2], [imageHeight / 2], [-f]])  # top left point
    tr = np.array([[imageWidth / 2], [imageHeight / 2], [-f]])  # top right point
    bl = np.array([[-imageWidth / 2], [-imageHeight / 2], [-f]])  # bot left point
    br = np.array([[imageWidth / 2], [-imageHeight / 2], [-f]])  # bot right point
    # calc the value in the 3d system, lambda = 1
    tl = x0 + scale * R.dot(tl)
    tr = x0 + scale * R.dot(tr)
    bl = x0 + scale * R.dot(bl)
    br = x0 + scale * R.dot(br)
    return tl, tr, bl, br


def drawOrientation(R, x0, scale, ax):
    """
    Draw a 3d axis system representing the orientation of the camera

    :param R: rotation matrix
    :param x0: perspective center of the camera in the model/world space
    :param scale: scale for defining the axis length

    :type R: np.array 3x3
    :type x0: np.array 3x1
    :type scale: float

    :return: None
    """

    xAxis = x0 + np.reshape(scale * R[:, 0], x0.shape)
    yAxis = x0 + np.reshape(scale * R[:, 1], x0.shape)
    zAxis = x0 + np.reshape(scale * R[:, 2], x0.shape)

    # in the section draw the lines -> from x0 to xAxis ( for example )
    # plot x axis - red
    xs, ys, zs = [x0[0, 0], xAxis[0, 0]], [x0[1, 0], xAxis[1, 0]], [x0[2, 0], xAxis[2, 0]]
    ax.plot(xs, ys, zs, c='r')
    # plot y axis - green
    xs, ys, zs = [x0[0, 0], yAxis[0, 0]], [x0[1, 0], yAxis[1, 0]], [x0[2, 0], yAxis[2, 0]]
    ax.plot(xs, ys, zs, c='g')
    # plot z axis - blue
    xs, ys, zs = [x0[0, 0], zAxis[0, 0]], [x0[1, 0], zAxis[1, 0]], [x0[2, 0], zAxis[2, 0]]
    ax.plot(xs, ys, zs, c='b')





if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #chek if the DrawRays function works
    grdPnts = np.array([[201.062, 741.351, 241.987]])
    drawRays(grdPnts, np.array([[50], [50], [50]]))


    # check if drawimageframe function works
    f = 0.153
    R = Compute3DRotationMatrix(np.pi/3, 0, 0)
    scale = 50
    drawImageFrame(0.5, 0.5, R, np.array([[50], [50], [50]]), f, scale)


    # check if drawOrientation function works
    R = Compute3DRotationMatrix(np.pi/3, 0, 0)
    x0 = np.array([[50], [50], [50]])
    drawOrientation(R, x0, scale)

    plt.show()
