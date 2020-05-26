import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load( open( "input-images/wide_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Read in an image
img = cv2.imread('input-images/test_image2.png')
nx = 8 # the number of inside corners in x
ny = 6 # the number of inside corners in y

# MODIFY THIS FUNCTION TO GENERATE OUTPUT 
# THAT LOOKS LIKE THE IMAGE ABOVE
def corners_unwarp(img, nx, ny, mtx, dist):
    # Pass in your image into this function
    # Write code to do the following steps
    M = None
    corners = None
    warped = np.copy(img)    
    
    # 1) Undistort using mtx and dist
    img_undistort = cv2.undistort(warped, mtx, dist, None, mtx)
    
    # 2) Convert to grayscale
    img_gray = cv2.cvtColor(img_undistort, cv2.COLOR_BGR2GRAY)
    
    # 3) Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(img_gray, (nx, ny), None)
    
    # 4) If corners found: 
    if ret == True:
        
        # a) draw corners
        cv2.drawChessboardCorners(img_undistort, (nx, ny), corners, ret)
        
        
        print(np.argmin(a, axis=1))
        print(corners.shape)
        print(corners[1,0])
        # b) define 4 source points src = np.float32([[,],[,],[,],[,]])
                 #Note: you could pick any four of the detected corners 
                 # as long as those four corners define a rectangle
                 #One especially smart way to do this would be to use four 
                    # well-chosen
                 # corners that were automatically detected during the 
                #undistortion steps
                 #We recommend using the automatic detection of 
                    #corners in your code
        src_points = np.float32([[0,0],
                          [0,0],
                          [0,0],
                          [0,0]])
        # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
        destination_points = np.float32([[0,0],
                          [0,0],
                          [0,0],
                          [0,0]])
        # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
        M = cv2.getPerspectiveTransform(src,dst)
        Minv = cv2.getPerspectiveTransform(dst,src)
        
        # e) use cv2.warpPerspective() to warp your image to a top-down view
        warped = cv2.warpPerspective(img, M, image_size, flags=cv2.INTER_LINER)
        
    #delete the next two lines
    warped = img_undistort 
    return warped, M

top_down, perspective_M = corners_unwarp(img, nx, ny, mtx, dist)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(top_down)
ax2.set_title('Undistorted and Warped Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)