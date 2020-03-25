import numpy as np

def DrawCube(EdgeSize):

    return (np.array([[EdgeSize/2,EdgeSize/2,EdgeSize/2],[EdgeSize/2,EdgeSize/2,-EdgeSize/2],
              [EdgeSize/2,-EdgeSize/2,-EdgeSize/2],[EdgeSize/2,-EdgeSize/2,EdgeSize/2],
              [-EdgeSize/2,-EdgeSize/2,EdgeSize/2],[-EdgeSize/2,-EdgeSize/2,-EdgeSize/2],
              [-EdgeSize/2,EdgeSize/2,-EdgeSize/2],[-EdgeSize/2,-EdgeSize/2,EdgeSize/2]]))

def test():
    print('hello')


if __name__ == '__main__':
    print(DrawCube(10))