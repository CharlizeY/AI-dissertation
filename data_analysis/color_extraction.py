import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import cv2
import os
from PIL import Image
from scipy.signal import find_peaks
import skimage.feature as feature


def coarseness(image, kmax):
	image = np.array(image)
	w = image.shape[0]
	h = image.shape[1]
	kmax = kmax if (np.power(2,kmax) < w) else int(np.log(w) / np.log(2))
	kmax = kmax if (np.power(2,kmax) < h) else int(np.log(h) / np.log(2))
	average_gray = np.zeros([kmax,w,h])
	horizon = np.zeros([kmax,w,h])
	vertical = np.zeros([kmax,w,h])
	Sbest = np.zeros([w,h])

	for k in range(kmax):
		window = np.power(2,k)
		for wi in range(w)[window:(w-window)]:
			for hi in range(h)[window:(h-window)]:
				average_gray[k][wi][hi] = np.sum(image[wi-window:wi+window, hi-window:hi+window])
		for wi in range(w)[window:(w-window-1)]:
			for hi in range(h)[window:(h-window-1)]:
				horizon[k][wi][hi] = average_gray[k][wi+window][hi] - average_gray[k][wi-window][hi]
				vertical[k][wi][hi] = average_gray[k][wi][hi+window] - average_gray[k][wi][hi-window]
		horizon[k] = horizon[k] * (1.0 / np.power(2, 2*(k+1)))
		vertical[k] = horizon[k] * (1.0 / np.power(2, 2*(k+1)))

	for wi in range(w):
		for hi in range(h):
			h_max = np.max(horizon[:,wi,hi])
			h_max_index = np.argmax(horizon[:,wi,hi])
			v_max = np.max(vertical[:,wi,hi])
			v_max_index = np.argmax(vertical[:,wi,hi])
			index = h_max_index if (h_max > v_max) else v_max_index
			Sbest[wi][hi] = np.power(2,index)

	fcrs = np.mean(Sbest)
	return fcrs


def contrast(image):
	image = np.array(image)
	image = np.reshape(image, (1, image.shape[0]*image.shape[1]))
	m4 = np.mean(np.power(image - np.mean(image),4))
	v = np.var(image)
	std = np.power(v, 0.5)
	alfa4 = m4 / np.power(v,2)
	fcon = std / np.power(alfa4, 0.25)

	return fcon


def directionality(image):
	image = np.array(image, dtype = 'int64')
	h = image.shape[0]
	w = image.shape[1]
	convH = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
	convV = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
	deltaH = np.zeros([h,w])
	deltaV = np.zeros([h,w])
	theta = np.zeros([h,w])

	# calculation for deltaH
	for hi in range(h)[1:h-1]:
		for wi in range(w)[1:w-1]:
			deltaH[hi][wi] = np.sum(np.multiply(image[hi-1:hi+2, wi-1:wi+2], convH))
	for wi in range(w)[1:w-1]:
		deltaH[0][wi] = image[0][wi+1] - image[0][wi]
		deltaH[h-1][wi] = image[h-1][wi+1] - image[h-1][wi]
	for hi in range(h):
		deltaH[hi][0] = image[hi][1] - image[hi][0]
		deltaH[hi][w-1] = image[hi][w-1] - image[hi][w-2]

	# calculation for deltaV
	for hi in range(h)[1:h-1]:
		for wi in range(w)[1:w-1]:
			deltaV[hi][wi] = np.sum(np.multiply(image[hi-1:hi+2, wi-1:wi+2], convV))
	for wi in range(w):
		deltaV[0][wi] = image[1][wi] - image[0][wi]
		deltaV[h-1][wi] = image[h-1][wi] - image[h-2][wi]
	for hi in range(h)[1:h-1]:
		deltaV[hi][0] = image[hi+1][0] - image[hi][0]
		deltaV[hi][w-1] = image[hi+1][w-1] - image[hi][w-1]

	deltaG = (np.absolute(deltaH) + np.absolute(deltaV)) / 2.0
	deltaG_vec = np.reshape(deltaG, (deltaG.shape[0] * deltaG.shape[1]))

	# calculation the theta
	for hi in range(h):
		for wi in range(w):
			if (deltaH[hi][wi] == 0 and deltaV[hi][wi] == 0):
				theta[hi][wi] = 0
			elif(deltaH[hi][wi] == 0):
				theta[hi][wi] = np.pi
			else:
				theta[hi][wi] = np.arctan(deltaV[hi][wi] / deltaH[hi][wi]) + np.pi / 2.0
	theta_vec = np.reshape(theta, (theta.shape[0] * theta.shape[1]))

	n = 16
	t = 12
	cnt = 0
	hd = np.zeros(n)
	dlen = deltaG_vec.shape[0]

	for ni in range(n):
		for k in range(dlen):
			if((deltaG_vec[k] >= t) and (theta_vec[k] >= (2*ni-1) * np.pi / (2 * n)) and (theta_vec[k] < (2*ni+1) * np.pi / (2 * n))):
				hd[ni] += 1
	hd = hd / np.mean(hd)
	hd_max_index = np.argmax(hd)

	fdir = 0
	for ni in range(n):
		fdir += np.power((ni - hd_max_index), 2) * hd[ni]

	return fdir


def extract_color_features(filepath):
    img = cv2.imread(filepath)
    if img is None:
        return
    # Convert BGR to HSV colorspace
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Split the channels - h,s,v
    h, s, v = cv2.split(hsv)

    # The first central moment - average
    h_mean = np.mean(h)  
    s_mean = np.mean(s)  
    v_mean = np.mean(v)  

    # The second central moment - standard deviation
    h_std = np.std(h)  
    s_std = np.std(s)  
    v_std = np.std(v)  

    color_features = [h_mean, h_std, s_mean, s_std, v_mean, v_std]

    return color_features


def extract_intensity_features(filepath):
    img = cv2.imread(filepath)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img_pixels = np.asarray(gray_img).flatten()
    # Compute the global mean and std of the grayscale intensities
    mean = gray_img_pixels.mean()
    std = gray_img_pixels.std()
    
    # # Create a histogram of grayscale intensities
    # h = plt.hist(gray_img_pixels, bins=100, density=True)
    # density = h[0]
    # norm_density = density / density.sum()
    # # Smooth the data with moving average to treat the hist as a pdf 
    # norm_density_ma = pd.Series(norm_density).rolling(7, center=True).mean().values

    # # Find the peaks (maxima) in the pdf
    # peaks = find_peaks(norm_density_ma)[0]
    # num_of_peaks = len(peaks)

    # return [mean, std, num_of_peaks]
    return [mean, std]


def extract_texture_features(filepath): 
    img = cv2.imread(filepath)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #######################
    ## Tamura's features ##
    #######################
    # coars = coarseness(gray_img, 5)
    # cont = contrast(gray_img)
    # dir = directionality(gray_img)

    ##########
    ## GLCM ##
    ##########
    # Input pixel pair distance offsets (here is 1) and the pixel pair angles in radians
    graycom = feature.greycomatrix(gray_img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256)

    # Find the GLCM properties
    glcm_contrast = feature.greycoprops(graycom, 'contrast')
    homogeneity = feature.greycoprops(graycom, 'homogeneity')
    energy = feature.greycoprops(graycom, 'energy')
    correlation = feature.greycoprops(graycom, 'correlation')
    # dissimilarity = feature.greycoprops(graycom, 'dissimilarity')
    # ASM = feature.greycoprops(graycom, 'ASM')

    ave_contrast = np.mean(glcm_contrast)
    ave_homogeneity = np.mean(homogeneity)
    ave_energy = np.mean(energy)
    ave_correlation = np.mean(correlation)

    # return [coars, cont, dir]
    return [ave_contrast, ave_correlation, ave_energy, ave_homogeneity]


def extract_composition_features(filepath):
    ####################
    ## Edge detection ##
    ####################
    img = cv2.imread(filepath)
    h, w, c = img.shape
    img_size = h*w
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply the Canny edge detector, increase the apertureSize (odd) if want more detailed edges
    edges = cv2.Canny(gray_img, 100, 200, apertureSize=5)
    # Approximate detected edges to lines using Hough transformation
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=8, maxLineGap=2)
    # Calculate the normalised number of lines divided by the image size
    if lines is not None:
        num_of_lines = len(lines)/img_size
        lines = lines[:, 0]
        # Compute the average length of detected lines
        dists = np.sqrt((lines[:,2] - lines[:,0])**2 + (lines[:,3] - lines[:,1])**2)
        mean_len = np.mean(dists)
    else:
        num_of_lines = 0
        mean_len = 0

    return [num_of_lines, mean_len]


def concatenate_features(filepath):
    col_feats = extract_color_features(filepath)
    int_feats = extract_intensity_features(filepath)
    text_feats = extract_texture_features(filepath)
    comp_feats = extract_composition_features(filepath)

    concat_feats = np.array(col_feats + int_feats + text_feats + comp_feats)    
    return concat_feats


if __name__== "__main__":
    # Read the artemis data
    artemis_preprocessed_csv = '/Users/Cherry0904/Desktop/official_data/artemis_preprocessed.csv'
    # artemis_preprocessed_csv = '/content/drive/MyDrive/Github/artemis_preprocessed.csv'
    df = pd.read_csv(artemis_preprocessed_csv)

    # Create an empty df to record the image features
    temp_df = pd.DataFrame(columns=['art_style', 'painting', 'image_feature_vector'])

    # Set the path to Wikiart images
    # folder_dir = '/Users/Cherry0904/Desktop/wikiart_test'
    folder_dir = '/content/drive/MyDrive/Github/wikiart'
   
    # Add the image feature vector for each image
    for subdir, dirs, files in os.walk(folder_dir):
        for file in files:
            if not str(os.path.join(file)).startswith('.'): # Ignore the hidden file
                filepath = os.path.join(subdir, file)
                art_style = str(os.path.join(subdir))[38:]
                painting_name = str(os.path.join(file))[:-4] # A string
                image_feature_vector = concatenate_features(filepath) # A numpy array
                new_df = pd.DataFrame({'art_style': [art_style],
                                       'painting': [painting_name],
                                       'image_feature_vector' : [image_feature_vector]})
                temp_df = pd.concat([temp_df, new_df], ignore_index = True, axis = 0)

    # Feature normalisation 
    mean_vector = temp_df['image_feature_vector'].mean()
    sd_vector = np.stack(temp_df['image_feature_vector']).std(axis=0)
    temp_df['normalised_image_feature_vector'] = temp_df.apply(lambda row: ((row['image_feature_vector']-mean_vector)/sd_vector), axis=1)
    
    # Save the temporary dataframe
    temp_df.to_csv('/Users/Cherry0904/Desktop/df.csv', sep=',', encoding='utf-8')
    # temp_df.to_csv('/content/drive/MyDrive/Github/temp_df_three.csv', sep=',', encoding='utf-8')

    # Add the columns of image feature vectors to the artemis dataframe
    merged_df = df.merge(temp_df, on=['art_style','painting'], how='left')

    # Save the completed dataframe
    merged_df.to_csv('/Users/Cherry0904/Desktop/merged_df.csv', sep=',', encoding='utf-8')


    