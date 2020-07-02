import cv2
from Camera import Camera
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from scipy.linalg import null_space
import SingleImage
import ImagePair
from mpl_toolkits.mplot3d import Axes3D
import numpy.linalg as la

def find_match_points(gray1,gray2):
    """
    Find homological points between 2 images
    :param gray1: image
    :param gray2: image
    :return: match points
    """

    # keypoints detection
    # sift = cv2.SIFT()
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # keypoints matching
    indexParams = dict(algorithm=0, trees=5)
    searchParams = dict(checks=50)

    flann = cv2.FlannBasedMatcher(indexParams, searchParams)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    pts1 = []
    pts2 = []

    # filter out “far” homological points
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.70 * n.distance:  # change 0.85 to the threshold needed
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    print('Number of matches:', len(good))
    return np.asarray(pts1), np.asarray(pts2)

def create_F(homolog1, homolog2):
    """
    create Fundamental matrix by set of homological points in 2 images
    :param homolog1: homological points in first image
    :param homolog2: homological points in second image
    :return: F
    """
    x1 , y1 = homolog1[0], homolog1[1]
    x2 , y2 = homolog2[0], homolog2[1]
    A = np.zeros((len(x1),9))
    for i in range(len(x1)):
        A[i] = np.array([x1[i]*x2[i], y1[i]*x2[i], x2[i], x1[i]*y2[i], y1[i]*y2[i], y2[i], x1[i], y1[i], 1])
    N = A.T.dot(A)
    # find Eigenvalues and Eigenvectors
    eigvals, eigvecs = la.eig(N)
    # find the eigvec that corresponding to the minimal eigval
    min_eigval_index = eigvals.argmin()
    F = eigvecs[:, min_eigval_index]
    F = np.reshape(F, (3, 3))
    return F

def find_epipolarLine(F, point):
    """
    Find the epipolar line on the second image that corresponding to the point in the first image
    :param F: Fundamental matrix
    :param point: homological point in the first image
    :return: line coefficient
    """
    l_tag = F.dot(point)
    a = l_tag[0]
    b = l_tag[1]
    c = l_tag[2]
    return a,b,c

def runSac(hom1, hom2, p, w , treshold):

    n = hom1.shape[0]
    k = np.log(1 - p) / np.log(1 - w ** 8)
    hom1 = np.hstack((hom1, np.ones((n, 1))))
    hom2 = np.hstack((hom2, np.ones((n, 1))))
    i = 0
    while i < k:
        rand_index = np.random.choice(n, 8)
        points1 = hom1[rand_index]
        points2 = hom2[rand_index]
        F = create_F(points1, points2)
        D = []
        # calc the distance of the homological points from the epipolar lines
        for j in range(n):
            a, b, c = find_epipolarLine(F, hom1[j])
            x0, y0 = hom2[j, 0], hom2[j, 1]
            d = abs(a * x0 + b * y0 + c) / np.sqrt(a ** 2 + b ** 2)
            D.append(d)

        D = np.asarray(D)
        average_d = np.average(D)
        std_d = np.std(D)
        indexes = np.where(D < treshold)
        count = np.size(indexes)
        if (count / len(D)) < w:
            if i < k - 1:
                i = i + 1
            else:
                print('the model failed')
                i = i + 1
        else:
            print('number of homologic points', count)
            return indexes


if __name__ == '__main__':
    # cam1 = Camera(None, np.array([None, None]), np.array([None, None]), np.array([None, None]), None, None)
    # K = cam1.aoutomatic_calibration('D:\HWC\python\photo2\Photo2\lab8\images\chessbord')
    # print(cam1.K)
    cam2 = Camera(3581,np.array([1555,1908]),np.array([None, None]), np.array([None, None]), None, None)
    # import images
    images = glob(r'images\*.jpg')
    img1 = cv2.imread(images[0])
    img2 = cv2.imread(images[1])
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # plt.imshow(gray1)
    # plt.show()
    # plt.imshow(gray2)
    # plt.show()

    # find match points
    match_points1, match_points2 = find_match_points(gray1,gray2)
    # print('match points in first image','\n',match_rand_points1)
    # print('match points in second image','\n',match_rand_points2)

    # poltting the match points on the images
    plt.imshow(img1)
    plt.scatter(match_points1[:, 0], match_points1[:, 1],c='r', s=50)
    plt.show()
    plt.imshow(img2)
    plt.scatter(match_points2[:, 0], match_points2[:, 1],c='r', s=50)
    plt.show()

    # probability for model success
    p = 0.90
    # model fitting
    w = 0.6
    threshold = 0.5

    good_indexees = runSac(match_points1,match_points2,p,w,threshold)
    print(good_indexees)

    # n = np.size(good_points)
    # hom1 = np.hstack((matches1[good_points], np.ones((n, 1))))
    # hom2 = np.hstack((matches2[good_points], np.ones((n, 1))))





