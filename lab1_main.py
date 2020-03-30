import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


import ImagePair


import ObjectsSynthetic as obj
obj.test()


import PhotoViewer as pv
# pv.DrawCube(obj.CreateCube(10),ax)


# <<<<<<< HEAD
import Camera
# =======
from Camera import Camera


from SingleImage import SingleImage


# B

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

cube = obj.CreateCube(6)
pv.DrawCube(cube,ax)

# plt.show()

focal_length = 35
sensor_size = 25
camera1 = Camera(focal_length, np.array([0, 0]), None, None, None, sensor_size)
img1 = SingleImage(camera1)
img1.exteriorOrientationParameters = np.array([[0, 0, 20, 0, 0, 0]])

# draw image and cube
x0 = np.array([[0], [0], [10]])
# pv.drawOrientation(np.identity(3), x0, 2, ax)

# image_border = img1.imageBorders(0)
# # draw image border in world system
# x = image_border[:, 0]
# y = image_border[:, 1]
# z = np.zeros(len(x))

# draw image frame in world system
img1.drawSingleImage(cube,ax)
# pv.drawImageFrame(camera1.sensorSize/10, camera1.sensorSize/10, np.identity(3), x0, camera1.focalLength/1000, 1, ax)

plt.show()
# >>>>>>> 38cbc8d... some changes 2
