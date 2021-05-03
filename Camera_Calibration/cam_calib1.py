import numpy as np
import cv2 as cv
import glob


# FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS

chessboardSize = (12, 9)
ImageframeSize = (2048, 1536)


# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
# for 100mm (or 1 cm)size square, here translational matrix will be in "cm"
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

'''
# for 100mm size square, here translational matrix will be in "mm", if multiply by 100
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)*100
'''


# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.


images = glob.glob('Calibration-imgs/*.bmp')

for image in images:

    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(1000)


cv.destroyAllWindows()


# GET CALIBRATION DATA

ret, FirstCameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, ImageframeSize, None, None)

# Output of the camera calibration data
print("camera calibration is done: ", ret)
print("\nCamera matrix is: \n", FirstCameraMatrix)
print("Distrotion Paramenters are: \n", dist)
print("\n")
print("TRANSLATIONAL matrix in extrinsic camera matrix: \n", tvecs)
print("\n")
print("ROTATIONAL matrix in extrinsic camera matrix: \n", rvecs)


# UNDISTORTION, here the image "cam1_0.bmp" is used to remove distortion
# Therefore, paste the image "cam1_0.bmp" here first
img = cv.imread('cam1_0.bmp')
h,  w = img.shape[:2]
NewCameraMatrix, roi = cv.getOptimalNewCameraMatrix(FirstCameraMatrix, dist, (w, h), 1, (w, h))


# ...following codes are extracted form the Opencv camera calibration documents...

# Undistort
dst = cv.undistort(img, FirstCameraMatrix, dist, None, NewCameraMatrix)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibratedIMG1.png', dst)


# using Remapping
mapx, mapy = cv.initUndistortRectifyMap(FirstCameraMatrix, dist, None, NewCameraMatrix, (w, h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibratedIMG2.png', dst)


# Reprojection Error
mean_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], FirstCameraMatrix, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

print("\n")
print("total error: {}".format(mean_error/len(objpoints)))
