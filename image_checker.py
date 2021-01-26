"""Contains methods to check a directory for duplicate and similar images and move
them to a designated folder. Also contains methods to create a memory for images
between certain dates. Useful for organization of an archive of images.
(only accepts .jpg currently)"""
import os
import hashlib
import ntpath
import itertools
from datetime import datetime
import imageio
import asyncio
import scipy.spatial.distance as sci
import cv2
import numpy as np
from PIL import Image

def path_leaf(path):
	"""Returns the last identifier of a path"""

	head, tail = ntpath.split(path)
	return tail or ntpath.basename(head)
#hash duplicate checker
def get_duplicates(imageList):
	"""Returns a list with objects containing (index of duplicate in imageList,
	 index of duplicated image in imageList)"""

	duplicates = []
	hash_keys = dict()
	index = 0
	for filename in imageList:
		print('Checking ' + filename + '...')
		with open(filename, 'rb') as f:
			filehash = hashlib.md5(f.read()).hexdigest()
		if filehash not in hash_keys:
			hash_keys[filehash] = index
		else:
			print('duplicate added ' + filename)
			duplicates.append((index, hash_keys[filehash]))
		index = index + 1
	return duplicates


#
def move_duplicates(sourcePath, destPath):
	"""Find duplicates in a working directory, then move them to a different directory"""

	imageList = get_images(sourcePath)
	duplicates = get_duplicates(imageList) #get the duplicates of the file list
	print('Moving ' + str(len(duplicates)) + ' files...')
	for index in duplicates: #put each duplicate in the destination folder
		print('Moving file index ' + str(index))
		filePath = imageList[index[0]] #get the path of the duplicate file
		send_img_to_dest(filePath, destPath)

#blurry image checker
def get_images(sourcePath):
	"""Get all of the .jpg files in a given directory"""
	picturelist = []
	print('Retrieving images...')
	pictures = os.walk(sourcePath)
	for path, dirnames, img in pictures:
		for file in img:
			if '.JPG' in file or '.jpg' in file:
				print(file + 'added to image list...')
				picturelist.append(os.path.join(path, file))
	return picturelist

#get the laplacian value of the image
def get_laplacian_value(img):
	"""Calculate the laplacian value of an image"""
	return cv2.Laplacian(cv2.imread(img, cv2.IMREAD_GRAYSCALE), cv2.CV_64F).var()


#move blurry images to a different directory
def move_blurry_images(sourcePath, destPath, laplacianValue):
	"""Move blurry images that have a lower laplacian value than the given value"""

	picturelist = get_images(sourcePath)
	picturelist = filter_images(picturelist)
	j = len(picturelist)
	for img in picturelist:
		print(str(j) + ' files remaining...')
		j = j-1
		if get_laplacian_value(img) < laplacianValue:
			send_img_to_dest(img, destPath)

#similar image checker
def filter_images(images):
	"""Asserts a grayscale image as an image with 3 color channels"""

	imageList = []
	for image in images:
		try:
			assert imageio.imread(image, True)
			imageList.append(image)
		except AssertionError as e:
			print(e)
	return imageList

def img_gray(image):
	"""Converts an image to a grayscale image"""
	image = cv2.imread(image)

	return np.average(image, weights={0.299, 0.587, 0.114}, axis=2)

def resize(image, height=30, width=30):
	"""Get the "thumbprint" of an image, Returns the row array resized and column array resized"""
	rowRes = cv2.resize((image, height, width), interpolation=cv2.INTER_AREA).flatten()
	colRes = cv2.resize((image, height, width), interpolation=cv2.INTER_AREA).flatten('F')
	return rowRes, colRes

def intensity_diff(rowRes, colRes):
	"""Compares the index of each value to the index before it,
	and if that difference is less than 0, that index becomes a False;
	if greater then the index becomes True. It then smashes the difference row
	and difference column together in one long array"""
	difference_row = np.diff(rowRes)
	difference_col = np.diff(colRes)
	difference_row = difference_row > 0
	difference_col = difference_col > 0
	return np.hstack((difference_row, difference_col))

def file_hash(array):
	"""Hashes an array with MD5"""
	return hashlib.md5(array).hexdigest()

def difference_score(image, height=30, width=30):
	"""Calculates the difference score for an image"""
	gray = img_gray(image)
	rowRes, colRes = resize(gray, height, width)
	difference = intensity_diff(rowRes, colRes)
	return difference

def difference_score_dict_hash(imageList):
	"""Creates a hash of the images with their difference score being keys,
	if two images have the same key, they are similar"""
	dsDict = {}
	duplicates = []
	hash_ds = []
	for image in imageList:
		ds = difference_score(image)
		hash_ds.append(ds)
		filehash = hashlib.md5(ds).hexdigest()
		if filehash not in dsDict:
			dsDict[filehash] = image
		else:
			duplicates.append((image, dsDict[filehash]))
	return duplicates, dsDict, hash_ds

def hamming_distance(image, image2):
	"""Calculates the hamming distance (or similarity) between two images (converted to 1d arrays)"""
	score = sci.hamming(image, image2)
	return score

def difference_score_dict(imageList):
	"""Create a dictionary of imageList with the image being the key and
	the difference score being the value. If a key already exists, the
	image is put into a duplicate list"""
	dsDict = {}
	duplicates = []
	for image in imageList:
		ds = difference_score(image)
		if image not in dsDict:
			dsDict[image] = ds
		else:
			duplicates.append((image, dsDict[image]))
	return duplicates, dsDict

def get_similarity_list(dsDict, value):
	"""Get a list of similarities by comparing the hamming distance of every image"""
	duplicates = []
	for k1, k2 in itertools.combinations(dsDict, 2):
		if hamming_distance(dsDict[k1], dsDict[k2]) < value:
			duplicates.append((k1, k2))
	return duplicates

def move_similar_images(sourcePath, destPath, value):
	"""Moves similar images to a separate folder under a given value"""
	imageFiles = filter_images(get_images(sourcePath))
	duplicates, dsDict = difference_score_dict(imageFiles)

	duplicates = get_similarity_list(dsDict, value)
	print(duplicates)

def create_memory(sourcePath):
	"""Creates a memory folder if it doesn't exist, and then
	sends all jpgs to a folder if it falls between a specific
	date"""
	while True:
		memoryFolder = input('Enter a directory to send the pictures too')
		if os.path.isdir(memoryFolder):
			break
	while True:
		minDate = input('Enter a minimum date (YYYY:MM:DD):\n')
		if len(minDate) == 10:
			minDate = minDate + ' 00:00:00'
			break
	while True:
		maxDate = input('Enter a maximum date (YYYY:MM:DD):\n')
		if len(maxDate) == 10:
			maxDate = maxDate + ' 00:00:00'
			break

	#sort the pictures into the folder
	sort_pictures(sourcePath, memoryFolder, minDate, maxDate)


def sort_pictures(sourcePath, destPath, mindate, maxdate):
	"""#sort_pictures(sourcePath, destPath, '18/09/19 00:00:00', '18/09/19 00:00:00')
	sends a picture to a destPathination if it lies between two dates"""
	#convert dates to datetime
	minDate = datetime.strptime(mindate, '%Y:%m:%d %H:%M:%S')
	maxDate = datetime.strptime(maxdate, '%Y:%m:%d %H:%M:%S')
	imageList = get_images(sourcePath)
	for img in imageList:
		imgDate = image_date(img)
		if imgDate is not None:
			if imgDate <= maxDate and imgDate >= minDate:
				send_img_to_dest(img, destPath)

def send_img_to_dest(img, destPath):
	"""get the imageList,
	read each image and see if the date range in between min and max date"""
	fileName = path_leaf(img)
	fileName, ext = os.path.splitext(fileName)#get the name and extension of the file
	destination = os.path.join(destPath, fileName + ext)
	# make sure the file doesn't exist in the folder already
	j = 1
	while os.path.isfile(destination):
		destination = os.path.join(destPath, fileName
		                           + ' - ' + str(j) + ext) #get a new filepath to rename to (filename - 1)
		if os.path.isfile(destination):
			j = i + 1

	j = 1			
	os.rename(img, destination)

def image_date(image):
	"""Get the date of an image from it's EXIF data"""
	try:
		dateString = Image.open(image).getexif()[36867]
		return datetime.strptime(dateString, '%Y:%m:%d %H:%M:%S')
	except (KeyError, TypeError):
		print('error: no exif datetime value for for %r.' % image)
		return None
	except:
		print('Unhandled error occured with image. %r' % image)
		return None

if __name__ == "__main__":
	while True:
		source = input('Enter source folder path:\n')
		source = r"{}".format(source)
		if os.path.isdir(source):
			break
	while True:
		dest = input('Enter destination folder path:\n')
		dest = r"{}".format(dest)
		if os.path.isdir(dest):
			break
	while True:
		review = input('Enter a folder to send images you would like to review before deleting:\n')
		review = r"{}".format(review)
		if os.path.isdir(review):
			break

	i = 0
	while i != -1:
		print('1. Move duplicates\n2. Move Blurry Images\n' +
		      '3. Move similar images (doesn\'t work atm)' +
		      '\n4. Create a new memory folder')
		i = int(input('\nWhich operation do you want to do? (-1 to exit)\n'))
		if i == 1:
			print('Moving duplicates to ' + dest)
			move_duplicates(source, dest)
		if i == 2:
			print('hi')
			move_blurry_images(source, dest, 50)
		if i == 3:
			move_similar_images(source, dest, .1)
			print('complete')
		if i == 4:
			create_memory(source)
			print('hi')

	print('Thank you for using the image checker! Have a fantastic day you sexy beast')
	quit()
