import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from . import image_processings

def search_matching_points(
    img_src, img_ref, corner_detector:str = 'harris', block_size:int = 2,
    ksize:int = 3, plot_corner:bool = False, matcher:str = 'bruteforce',
    k:float = 0.04, threshold:float = 0.01, min_good_matches:int = 5,
    threshold_distance:float = 0.8
    ):
    """
    A function to find two matching points in two images (img_src, img_ref)
    mainly using corner matching method + SIFT. If corner_detector is set to 'none',
    only SIFT descriptor will be run. If min_good_matches condition is pass,
    will return (src_pts, ref_pts, img_src, img_ref) otherwise None.

    Arguments:
    - corner_detector (str) : i.e: 'harris', 'shitomasi', 'none'
    - matcher (str) : matcher type i.e: 'bruteforce', 'flann'

    Returns:
    - src_pts, ref_pts, img_src, img_ref
    or
    - None

    """

    # ----- check arguments -----

    corner_detectors = ['harris', 'shitomasi', 'none']
    matchers = ['bruteforce', 'flann']

    if corner_detector not in corner_detectors:
        raise Exception(f'Invalid corner_detector argument, choose from : {corner_detectors}')

    if matcher not in matchers:
        raise Exception(f'Invalid matcher argument, choose from : {matchers}')


    # ----- detect corner -----

    # Harris corner detection
    if corner_detector == 'harris':

        # detect Harris corners
        corners1 = cv.cornerHarris(img_src, block_size, ksize, k)
        corners2 = cv.cornerHarris(img_ref, block_size, ksize, k)

        # threshold the corners to keep only strong corners
        corners1[corners1 < threshold * corners1.max()] = 0
        corners2[corners2 < threshold * corners2.max()] = 0

        # find the coordinates of the detected corners
        corner_coordinates1 = np.argwhere(corners1 > 0)
        corner_coordinates2 = np.argwhere(corners2 > 0)

        # finding descriptors & keypoints
        sift = cv.SIFT_create()
        keypoints1 = [cv.KeyPoint(float(c[1]), float(c[0]), 5) for c in corner_coordinates1]
        keypoints2 = [cv.KeyPoint(float(c[1]), float(c[0]), 5) for c in corner_coordinates2]
        _, descriptors1 = sift.compute(img_src, keypoints1)
        _, descriptors2 = sift.compute(img_ref, keypoints2)


    # Shi-Tomasi corner detection
    elif corner_detector == 'shitomasi':

        # === detect corners ===
        corners1 = cv.goodFeaturesToTrack(img_src, maxCorners=100, qualityLevel=0.01, minDistance=2)
        corners2 = cv.goodFeaturesToTrack(img_ref, maxCorners=100, qualityLevel=0.01, minDistance=2)
        # corners1 = np.int0(corners1)
        # corners2 = np.int0(corners2)
        corners1 = np.intp(corners1)
        corners2 = np.intp(corners2)

        # find the coordinates of the detected corners
        corner_coordinates1 = corners1.copy() #corners1[:, 0, :]
        corner_coordinates2 = corners2.copy() #corners2[:, 0, :]

        # finding descriptors & keypoints
        sift = cv.SIFT_create()
        keypoints1 = [cv.KeyPoint(float(x), float(y), 5) for x, y in corners1[:, 0]]
        keypoints2 = [cv.KeyPoint(float(x), float(y), 5) for x, y in corners2[:, 0]]
        _, descriptors1 = sift.compute(img_src, keypoints1)
        _, descriptors2 = sift.compute(img_ref, keypoints2)


    # if no corner detection, do sift features detection
    elif corner_detector == 'none':

        # finding descriptors & keypoints
        sift = cv.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(img_src, None)
        keypoints2, descriptors2 = sift.detectAndCompute(img_ref, None)


    # ----- visualised detected corners -----

    if plot_corner != False and corner_detector != 'none':

        # convert to BGR (3 channels)
        img_src_corners = cv.cvtColor(img_src, cv.COLOR_GRAY2BGR)
        img_ref_corners = cv.cvtColor(img_ref, cv.COLOR_GRAY2BGR)

        # mark coordinates in img_src
        for coords in corner_coordinates1:
            if corner_detector=='harris':
                cv.circle(img_src_corners, (coords[1], coords[0]), 2, (0, 0, 255), -1)
            elif corner_detector=='shitomasi':
                x1, y1 = coords.ravel()
                cv.circle(img_src_corners, (x1, y1), 2, (0, 0, 255), -1)

        # mark coordinates in img_ref
        for coords in corner_coordinates2:
            if corner_detector=='harris':
                cv.circle(img_ref_corners, (coords[1], coords[0]), 2, (0, 0, 255), -1)
            elif corner_detector=='shitomasi':
                x2, y2 = coords.ravel()
                cv.circle(img_ref_corners, (x2, y2), 2, (0, 0, 255), -1)

        # plot the source image on the left
        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(img_src_corners)
        plt.title('Source Image')
        plt.axis('off')

        # plot the ref image on the right
        plt.subplot(1, 2, 2)
        plt.imshow(img_ref_corners)
        plt.title('Reference Image')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        plt.clf()
        plt.close()


    # ----- matching the coordinates -----

    # Brute-Force Matcher
    if matcher=='bruteforce':
        bf = cv.BFMatcher()
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # FLANN-based Matcher
    elif matcher=='flann':
        index_params = dict(algorithm=0, trees=5)
        search_params = dict(checks=50)  #<- Higher checks value gives better accuracy, but is slower
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)


    # ----- filter good matches only -----

    # apply ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < threshold_distance * n.distance:
            good_matches.append(m)

    # filter good_matches to min value
    if len(good_matches) >= min_good_matches:

        # extract matching keypoints coordinates
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        ref_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        return src_pts, ref_pts, img_src, img_ref
    else:
        return None


def imgs_transformation(
        src_pts, dst_pts, img_src, img_ref, img_transformed=None,
        transformation='affine', bin_invert:bool=False,
        plot_transformation:bool=False, show_plot:bool=False, 
        border_mode=cv.BORDER_REPLICATE, 
        fname:str=None
        ):
    """
    Accept 2D (w, h) or multi-channel image (w, h, c)

    """
    multichannel_mode = False

    # ----- affine transformation -----

    if transformation=='affine':

        # get the M for warp
        M, mask = cv.estimateAffinePartial2D(src_pts, dst_pts, cv.RANSAC)
        M = M.astype(np.float32)

        # img_src = img_src.astype(np.float32) # dont uncomment for bin_invert

        # ----- 2D image transformation -----

        tgt_img_size = (img_ref.shape[0], img_ref.shape[1])

        if len(img_src.shape) == 2:
            transformed = cv.warpAffine(img_src, M, tgt_img_size)

        # ----- multi channel image transformation -----

        elif len(img_src.shape) != 2:

            multichannel_mode = True

            img_ref_size = (img_ref.shape[0], img_ref.shape[1])
            channel =  img_src.shape[2]
            transformed = np.zeros((tgt_img_size[0], tgt_img_size[1], channel), dtype=np.float32)


            for i in range(0, transformed.shape[2]):
                transformed_channel = cv.warpAffine(img_src[:,:,i], M, img_ref_size, flags=cv.INTER_NEAREST, borderMode=border_mode)
                transformed[:,:,i] = transformed_channel

            # transformed = cv2.convertScaleAbs(transformed)
            # transformed[transformed == 0] = 255


    # if no transformation not needed (just for plotting) but expect img_transformed
    elif transformation==None:
        if img_transformed != None:
            transformed = img_transformed
        else:
            print('Expect <img_transformed> not None.')

    # invalid transformation method
    else:
        print('Invalid <transformation> input.')

    # binarise and invert transformed image
    if bin_invert != False:
        if multichannel_mode:
            raise Exception('*bin_invert not supported for multichannel image.')
        # print('RUNNING INVERT')
        # print(transformed.shape, transformed.min(), transformed.max())
        transformed = image_processings.img_thres_otsu(transformed, blur_kernel=(1,1)) #<- binarise back since warp affine returns 255 values
        transformed = cv.bitwise_not(transformed)
        # print(transformed.shape, transformed.min(), transformed.max())


    # ----- visualise the transformation points -----

    if plot_transformation != False:

        # check channels first, visualisations part expect all images in 2D

        # 1. source image
        if len(img_src.shape) != 2:
            img_src_gray = cv.cvtColor(img_src, cv.COLOR_BGR2GRAY)
        else:
            img_src_gray = img_src

        # 2. referemce image
        if len(img_ref.shape) != 2:
            img_ref_gray = cv.cvtColor(img_ref, cv.COLOR_BGR2GRAY)
        else:
            img_ref_gray = img_ref

        # 3. transformed image
        if len(transformed.shape) != 2:
            transformed_gray = cv.convertScaleAbs(transformed)
            transformed_gray = cv.cvtColor(transformed_gray, cv.COLOR_BGR2GRAY)
        else:
            transformed_gray = transformed

        # stack 2 imgs
        visualization = np.hstack((img_src_gray, img_ref_gray))

        # draw lines to connect corresponding points
        plt.figure(figsize=(12, 6))
        for pt1, pt2 in zip(src_pts, dst_pts):
            pt1 = tuple(map(int, pt1[0]))
            pt2 = tuple(map(int, pt2[0]))
            plt.plot([pt1[0], pt2[0] + img_src_gray.shape[1]], [pt1[1], pt2[1]], color='red', linewidth=2)
        visualization_with_lines = np.copy(visualization)

        # stack 3 imgs
        final_visualization = np.hstack((visualization_with_lines, transformed_gray))

        # display
        plt.imshow(final_visualization)
        plt.axis('off')
        if fname != None:
            plt.savefig(fname, bbox_inches='tight', pad_inches=0)
        if show_plot != False:
            plt.show()
        plt.clf()
        plt.close()

    return transformed



def plot_sift_keypoints(img_src=None, img_ref=None,
                        img_src_path=None, img_ref_path=None
                        ):
    
    # Load your images
    if img_src!=None and img_ref!=None:
        img_src = img_src
        img_ref = img_ref
    elif img_src_path!=None and img_ref_path!=None:
        img_src = cv.imread(img_src_path)
        img_ref = cv.imread(img_ref_path)

    # Initialize SIFT detector
    sift = cv.SIFT_create()

    # Detect keypoints and compute descriptors for both images
    keypoints1, descriptors1 = sift.detectAndCompute(img_src, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img_ref, None)

    # Draw keypoints on the images
    img_keypoints1 = cv.drawKeypoints(img_src, keypoints1, img_src)
    img_keypoints2 = cv.drawKeypoints(img_ref, keypoints2, img_ref)

    # Convert images from BGR to RGB (matplotlib uses RGB)
    img_keypoints1 = cv.cvtColor(img_keypoints1, cv.COLOR_BGR2RGB)
    img_keypoints2 = cv.cvtColor(img_keypoints2, cv.COLOR_BGR2RGB)

    # Plot the images with keypoints using matplotlib
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].imshow(img_keypoints1)
    axs[0].set_title('Keypoints Image 1')
    axs[0].axis('off')

    axs[1].imshow(img_keypoints2)
    axs[1].set_title('Keypoints Image 2')
    axs[1].axis('off')

    plt.show()