import numpy as np

def CreateCube(EdgeSize):

    return (np.array([[EdgeSize/2,EdgeSize/2,EdgeSize],
                      [EdgeSize/2,EdgeSize/2,0],
                      [EdgeSize/2,-EdgeSize/2,0],
                      [EdgeSize/2,-EdgeSize/2,EdgeSize],
                      [-EdgeSize/2,-EdgeSize/2,EdgeSize],
                      [-EdgeSize/2,-EdgeSize/2,0],
                      [-EdgeSize/2,EdgeSize/2,0],
                      [-EdgeSize/2,EdgeSize/2,EdgeSize]]))

def test():
    print('hello')


if __name__ == '__main__':
    print(CreateCube(10))