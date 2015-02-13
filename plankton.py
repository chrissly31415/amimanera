#!/usr/bin/python 
# coding: utf-8

#http://nbviewer.ipython.org/github/udibr/datasciencebowl/blob/master/141215-tutorial.ipynb
#Import libraries for doing image analysis
#http://www.pyimagesearch.com/2014/09/22/getting-started-deep-learning-python/
#mklhttp://verahill.blogspot.de/2013/06/465-intel-mkl-math-kernel-library-on.html

import glob
import os
import sys
import pickle

from skimage.io import imread
from skimage.transform import resize
from skimage import segmentation
from skimage.morphology import watershed
from skimage import measure
from skimage import morphology
from skimage.feature import peak_local_max
from scipy.ndimage import convolve
from scipy import ndimage

from nolearn.dbn import DBN
import cv2

from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold,LeavePLabelOut
from sklearn.metrics import classification_report
from sklearn.lda import LDA

from sklearn.feature_extraction import image

from sklearn.neural_network import BernoulliRBM

from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib.cm as cm
from pylab import cm

import zipfile

import numpy as np
import pandas as pd
import sklearn as sl

from qsprLib import *


#TODO
#http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/
#http://www.kaggle.com/c/datasciencebowl/forums/t/11421/converting-images-to-standard-scale-code
#transformation
#http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
#extract patches
#http://scikit-learn.org/stable/auto_examples/decomposition/plot_image_denoising.html#example-decomposition-plot-image-denoising-py
#onlnie
#http://scikit-learn.org/stable/auto_examples/cluster/plot_dict_face_patches.html#example-cluster-plot-dict-face-patches-py


def prepareData(subset=1000,loadData=False,extractFeatures=False,maxPixel = 48,doSVD=25,subsample=None,nudgeData=False,dilation=4,kmeans=None,randomRotate=True,useOnlyFeats=False,stripFeats=True,createExtraFeatures=True,patchWork=True,standardize=True,location='train'):
  # just load processed data
  if loadData:
      X = pd.read_csv('train_tmp.csv',index_col=0)
      X_test = pd.read_csv('test_tmp.csv',index_col=0)      
      y = pd.read_csv('labels_tmp.csv', sep=",", na_values=['?'],index_col=0).iloc[:,-1]
      
  else:
    #extract the features
    if extractFeatures:
      df = createFeatures(maxPixel=maxPixel,location=location,dilation=dilation)
      X_test = createFeatures(maxPixel,location='test',dilation=dilation)
      #createFeatures(resizeIt,rawFeats,maxPixel=maxPixel,location='train_48_mod',dilation=dilation)
      #createFeatures(resizeIt,rawFeats,maxPixel,location='test_48_mod',dilation=dilation)
    else:
     #load extracted features 
      print "Load extracted features from: ",location
      df = pd.read_csv('competition_data/'+location+'_'+str(maxPixel)+'.csv', sep=",", na_values=['?'],index_col=0)
      X_test = pd.read_csv('competition_data/'+'test'+'_'+str(maxPixel)+'.csv', sep=",", na_values=['?'],index_col=0)
      
    #print df_test
    #df.loc[:,['ratio','label']].hist()
    if subset is not None:
      df = df.loc[np.random.choice(df.index, subset, replace=False)]
    
    idx = df.shape[1]-1
    X = df.iloc[:,0:idx]
    y = df.iloc[:,-1]

    if subsample is not None:
      cv = StratifiedShuffleSplit(y, n_iter=1, test_size=1.0-subsample)
      for train,test in cv:
	X = X.iloc[train]
	y = y.iloc[train]
  
    print "Shape X:",X.shape
    print "Shape X_test:",X_test.shape
    print "Shape y:",y.shape
    
    X,X_feat = splitFrame(X,n_features=23)
    X_test,X_test_feat = splitFrame(X_test,n_features=23)
    
    if nudgeData:
      X,y = nudge_dataset(X,y,maxPixel,dilation)
      
    if randomRotate:
      X,y = makeRotation(X,y,maxPixel)
      X_test = makeRotation(X_test,None,maxPixel)
    
    
    if createExtraFeatures:
	#makeBriefFeatures(X,maxPixel)
	X = makeExtraFeatures(X,maxPixel)
	X_test = makeExtraFeatures(X_test,maxPixel)
    #if computeAddFeats:
      #X = makeAddFeats(X,maxPixel,dilation)
      #X_test = makeAddFeats(X_test,maxPixel,dilation)
      #print "New shape X:",X.shape
      #print "New shape X_test:",X_test.shape
    
	
    #Do dimension reduction on train and! test
    if doSVD is not None:
	  print "Singular value decomposition..."
	  tsvd=TruncatedSVD(n_components=doSVD, algorithm='randomized', n_iter=5, tol=0.0)
	  X_all = pd.concat([X_test, X])
	  print "Shape all data:",X_all.shape
	  

	  X_SVD=tsvd.fit_transform(X_all)
	  X_all=pd.DataFrame(np.asarray(X_SVD),index=X_all.index)
	  print "Shape all data, transformed:",X_all.shape
	  
	  X = X_all[len(X_test.index):]
	  X_test = X_all[:len(X_test.index)]


	      
	  print "Shape train data, transformed:",X.shape
	  print "Shape test data, transformed:",X_test.shape
    
    if kmeans is not None:
	  print "Create kmeans features k=",kmeans
	  #kmeans_job = KMeans(init='k-means++', n_clusters=kmeans, n_init=3,n_jobs=4)
	  kmeans_job = MiniBatchKMeans(init='k-means++', n_clusters=kmeans, n_init=3,n_jobs=4)
	  X_all = pd.concat([X_test, X])
	  print "Shape all data:",X_all.shape
	  kmeans_job.fit(X_all)
	  ck = kmeans_job.cluster_centers_
	  print "Cluster centers:",ck.shape
	  
	  X_kmeans = kmeans_job.transform(X_all)
	  X_all=pd.DataFrame(np.asarray(X_kmeans),index=X_all.index)
	  
	  X = X_all[len(X_test.index):]
	  X_test = X_all[:len(X_test.index)]
    
    if not stripFeats:
	X = attachFeatures(X,X_feat)
	X_test = attachFeatures(X_test,X_test_feat)
    
    if useOnlyFeats:
	print "Use only add. features."
	X = X_feat
	X_test = X_test_feat
	  
    X.to_csv('train_tmp.csv')
    X_test.to_csv('test_tmp.csv')
    y.to_csv('labels_tmp.csv')
  
  class_names = getClassNames()
  
  if standardize==True:
    print "Standardize data..."
    X_all = pd.concat([X_test, X])
    X_all = scaleData(X_all)
    X = X_all[len(X_test.index):]
    X_test = X_all[:len(X_test.index)]
    
  
  if patchWork:
	doPatchWork(X,y,maxPixel)
  
  print "Final Shape X:",X.shape
  print "Final Shape X_test:",X_test.shape
  #print "Shape y:",y.shape
  
  return X,y,class_names,X_test


def doPatchWork(lX,ly=None,maxPixel=None,showEigenImages=False):
    """
    Extract patches....
    """
    #http://scikit-learn.org/stable/auto_examples/cluster/plot_dict_face_patches.html
    #http://scikit-learn.org/stable/modules/preprocessing.html
    #http://nbviewer.ipython.org/github/dmrd/hackclass-sklearn/blob/master/image_classification.ipynb
    #http://scikit-learn.org/stable/auto_examples/applications/face_recognition.html
    patchSize=20
    n_components=20
    max_patches=5
    
    print "Extract patches, original shape:",lX.shape
    tmp=[]
    tmpy=[]
    for i,row in enumerate(lX.values):
	img = np.reshape(row, (maxPixel, maxPixel))
	patches = image.extract_patches_2d(img, (patchSize, patchSize), max_patches=max_patches)
	patches = np.reshape(patches, (len(patches), -1))
	tmp.append(patches)
	tmpy.append([ly for _ in range(max_patches)])
	#if ly is not None: ly = pd.Series(np.concatenate([ly for _ in range(6)], axis=0))
    
    data = np.concatenate(tmp, axis=0)
    ly = np.concatenate(tmpy, axis=0)
    print "Extract patches, new shape:",data.shape
    print "New shape y:",ly.shape
    
    pca = RandomizedPCA(n_components, whiten=True)
    data = pca.fit_transform(data)
    print "Extracted patches, after PCA shape:",data.shape
    print "PCA components shape:",pca.components_.shape
    
    eigenfaces = pca.components_.reshape((n_components, patchSize, patchSize))
    
    print "Show eigen data:",eigenfaces.shape
    
    if showEigenImages:
      for img in eigenfaces:
	  showImage(img,patchSize,fac=10,matrix=True)

    lX = pd.DataFrame(data)
    return (lX,ly)
	

def makeBriefFeatures(lX,maxPixel):
    #http://docs.opencv.org/trunk/doc/py_tutorials/py_feature2d/py_orb/py_orb.html
    #http://stackoverflow.com/questions/7232651/how-does-opencv-orb-feature-detector-work
    #http://stackoverflow.com/questions/7232651/how-does-opencv-orb-feature-detector-work
    #http://stackoverflow.com/questions/10168686/algorithm-improvement-for-coca-cola-can-shape-recognition/10169025#10169025
    print "Make Brief Descriptors"
    print cv2.__version__
    #orb = cv2.ORB(nfeatures=500)
    surf = cv2.SURF(4000)#http://docs.opencv.org/trunk/doc/py_tutorials/py_feature2d/py_surf_intro/py_surf_intro.html
    #fast = cv2.FastFeatureDetector()
    #
    star = cv2.FeatureDetector_create("STAR")
    brief = cv2.DescriptorExtractor_create("BRIEF")
    
    for row in lX.values:
      #32bit float grayscale
      img = np.reshape(row, (maxPixel, maxPixel)).astype("float32")#!!!
      print img.shape
      showImage(img,maxPixel,fac=10,matrix=True)
      img2 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
      print img2.shape
      showImage(img2,maxPixel,fac=10,matrix=True)

      #8bit int grayscale
      img = (img * 255).astype("uint8")
      #plt.imshow(img)#wrong colorscale
      #plt.show()
      
      #kp = orb.detect(img,None)
      
      #kp, des = orb.compute(img, kp)
      kp, des = surf.detectAndCompute(img,None)
      #kp = star.detect(img,None)
      #kp, des = brief.compute(img, kp)
      #kp = fast.detect(img,None)
      
      print "Keyp",kp
      print "des",des
      
      #img2 = cv2.drawKeypoints(img,kp,color=(0,255,0), flags=0)
      img = cv2.drawKeypoints(img, kp, color=(255,0,0))
      showImage(img,maxPixel,fac=10,matrix=True)

    

def attachFeatures(lX,lX_feat):
    """
    attach and multiply extra features
    """
    print "Attach extra features."
    factor = lX.shape[0]/lX_feat.shape[0]-1
    print "factor:",factor
    
    X_tmp = lX_feat.copy()
    for i in xrange(factor):
	
	lX_feat = pd.concat([lX_feat,X_tmp.copy()],axis=0)
    
    
    print "X FEAT:",lX_feat.shape
    lX_feat.index = lX.index
    
    lX = pd.concat([lX,lX_feat],axis=1)
  
    print "X SHAPE:",lX.shape
    return lX

def splitFrame(lX,n_features):
    print "Remove extra features..."
    X_pure = lX.iloc[:,:-n_features]
    X_feat = lX.iloc[:,-n_features:] 
    print "PURE",X_pure.shape
    print "FEATURES",X_feat.shape
    print X_feat.columns
    return X_pure, X_feat
    

def getClassNames():
    namesClasses=[]
    directory_names = getDirectoryNames()
    for label,folder in enumerate(directory_names):
	currentClass = os.path.basename(folder)
	namesClasses.append(currentClass)
    return namesClasses

def getDirectoryNames(location="train"):
    directory_names = list(set(glob.glob(os.path.join("competition_data",location, "*"))).difference(set(glob.glob(os.path.join("competition_data",location,"*.*")))))
    return sorted(directory_names)


def makeRowProps(imageName,imageSize,lX,row,dilation=4):

    image = imread(imageName, as_grey=True)     
    image_dilated, axisratio,area,euler_number,perimeter,convex_area,eccentricity,equivalent_diameter,extent,filled_area,orientation,solidity,nregions,inertia_te,moments_hu,centroid = getImageFeatures(image,dilation)
    image = resize(image, (maxPixel, maxPixel))#before or after image creation???

    # Store the rescaled image pixels and the axis ratio
    lX[row, 0:imageSize] = np.reshape(image, (1, imageSize))
    lX[row, imageSize] = axisratio
    lX[row, imageSize+1] = area
    lX[row, imageSize+2] = euler_number
    lX[row, imageSize+3] = perimeter
    lX[row, imageSize+4] = convex_area
    lX[row, imageSize+5] = eccentricity
    lX[row, imageSize+6] = equivalent_diameter
    lX[row, imageSize+7] = extent
    lX[row, imageSize+8] = filled_area
    lX[row, imageSize+9] = orientation
    lX[row, imageSize+10] = solidity
    lX[row, imageSize+11] = nregions
    lX[row, imageSize+12] = inertia_te[0]
    lX[row, imageSize+13] = inertia_te[1]
    lX[row, imageSize+14] = moments_hu[0]
    lX[row, imageSize+15] = moments_hu[1]
    lX[row, imageSize+16] = moments_hu[2]
    lX[row, imageSize+17] = moments_hu[3]
    lX[row, imageSize+18] = moments_hu[4]
    lX[row, imageSize+19] = moments_hu[5]
    lX[row, imageSize+20] = moments_hu[6]
    lX[row, imageSize+21] = centroid[0]
    lX[row, imageSize+22] = centroid[1]
    

def createFeatures(maxPixel = 25,location="train",dilation=4,appendFeats=True):
    print "Extracting features for ",location,"  dilation:",dilation," maxpixel:",maxPixel 
    #get the total training images
    numberofImages = 0
    
    if 'train' in location:
        directory_names = getDirectoryNames(location)
	for folder in directory_names:
	    for fileNameDir in os.walk(folder):
		for fileName in fileNameDir[2]:
		    # Only read in the images
		    if fileName[-4:] != ".jpg":
		      continue
		    numberofImages += 1
	
	ly = np.zeros((numberofImages,1), dtype=int)
	
    else:
        directory_names = set(glob.glob(os.path.join("competition_data",location, "*")))
        directory_names = sorted(directory_names)
        #print directory_names
	for entry in directory_names:
	     if entry[-4:] != ".jpg":
		continue
	     numberofImages += 1

    print ", number of images:",numberofImages
    
    imageSize = maxPixel * maxPixel
    if appendFeats:
      n_addFeats = 23
    else:
      n_addFeats = 0
    num_features = imageSize + n_addFeats # for our ratio
    
    lX = np.zeros((numberofImages, num_features), dtype=float)


    print "Reading images"
    # Navigate through the list of directories
    i = 0
    testFiles=[]
    for label,folder in enumerate(directory_names):
	# Append the string class name for each class
	if 'train' in location: 
	    currentClass = folder.split(os.pathsep)[-1]
	    print "Label: %4d Class: %-32s"%(label,currentClass)
	  
	    for fileNameDir in os.walk(folder):
		for fileName in fileNameDir[2]:
		    # Only read in the images
		    if fileName[-4:] != ".jpg":
		      continue
		    #if i>1000: continue
		    
		    # Read in the images and create the features
		    nameFileImage = "{0}{1}{2}".format(fileNameDir[0], os.sep, fileName)
		    if appendFeats:
			makeRowProps(nameFileImage,imageSize,lX,i,dilation)
		    else:
			image = imread(nameFileImage, as_grey=True)
			image = resize(image, (maxPixel, maxPixel))		    
			lX[i, 0:imageSize] = np.reshape(image, (1, imageSize))
		    #
		    ly[i] = label		    
		    i += 1
		#print "row: %4d label: %4d"%(i,label)
        else:
	  if appendFeats:
	    makeRowProps(folder,imageSize,lX,i,dilation)
	  else:
	    image = imread(folder, as_grey=True)
	    image = resize(image, (maxPixel, maxPixel))
	    lX[i, 0:imageSize] = np.reshape(image, (1, imageSize))
	  #
	  fileName = os.path.basename(folder)
	  testFiles.append(fileName)
	  i += 1
	  if i%5000==0:
	    print "i: %4d image: %-32s"%(i,folder)
	  
    print "done"
    #create dataframes
    # Loop through the classes two at a time and compare their distributions of the Width/Length Ratio
    colnames = [ "p"+str(x+1) for x in xrange(lX.shape[1]-n_addFeats)]
    if appendFeats:
      colnames = colnames + ['axisratio','area','euler_number','perimeter','convex_area','eccentricity','equivalent_diameter','extent','filled_area','orientation','solidity','nregions']
      colnames = colnames + ['inertia_te1','inertia_te2','moments_hu1','moments_hu2','moments_hu3','moments_hu4','moments_hu5','moments_hu6','moments_hu7','centroidx','centroidy']
    #Create a DataFrame object to make subsetting the data on the class 
    df = pd.DataFrame(lX)
    if 'train' in location: 
	colnames.append('label')
	ly = pd.DataFrame(ly)
	df = pd.concat([df, ly],axis=1,ignore_index=True)
    print "Final shape of (incl. labels)",location," :",df.shape
    df.columns = colnames 
    #df.label.hist()
    df.to_csv('competition_data/'+location+'_'+str(maxPixel)+'.csv')
    print df.describe()
    return df
    

# find the largest nonzero region
def getLargestRegion(props, labelmap, imagethres):
    regionmaxprop = None
    for regionprop in props:
      
        # check to see if the region is at least 50% nonzero
        if sum(imagethres[labelmap == regionprop.label])*1.0/regionprop.area < 0.50:
            continue
        if regionmaxprop is None:
            regionmaxprop = regionprop
        if regionmaxprop.filled_area < regionprop.filled_area:
            regionmaxprop = regionprop
    return regionmaxprop
  
  
def getImageFeatures(image,pdilation=4):
    #http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_feature2d/py_brief/py_brief.html
    #http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_feature2d/py_table_of_contents_feature2d/py_table_of_contents_feature2d.html
    #http://stackoverflow.com/questions/13329357/extractsurf-always-returning-the-same-direction
    #http://scikit-image.org/docs/dev/api/skimage.measure.html#regionprops
    image = image.copy()
    # Create the thresholded image to eliminate some of the background
    imagethr = np.where(image > np.mean(image),0.,1.0)

    #Dilate the image
    imdilated = morphology.dilation(imagethr, np.ones((pdilation,pdilation)))

    # Create the label list
    label_list = measure.label(imdilated)
    label_list = imagethr*label_list
    label_list = label_list.astype(int)
    
    region_list = measure.regionprops(label_list)
    maxregion = getLargestRegion(region_list, label_list, imagethr)

    # guard against cases where the segmentation fails by providing zeros
    ratio = 0.0
    try:
      if ((not maxregion is None) and  (maxregion.major_axis_length != 0.0)):
	  if maxregion.minor_axis_length<0: print maxregion.minor_axis_length
	  if maxregion.major_axis_length<0: print maxregion.major_axis_length
	  ratio = 0.0 if maxregion is None else  maxregion.minor_axis_length*1.0 / maxregion.major_axis_length
    except ValueError:
	  print "Some internal error with this image..."
	  
    area = 0.0
    if hasattr(maxregion,'area'):
      area = maxregion.area
    
    euler_number = 0
    if hasattr(maxregion,'euler_number'): 
      euler_number = maxregion.euler_number
      
    perimeter = 0.0
    if hasattr(maxregion,'perimeter'): 
      perimeter = maxregion.perimeter
    
    convex_area = 0.0
    if hasattr(maxregion,'convex_area'): 
      convex_area = maxregion.convex_area
    
    eccentricity = 0.0
    if hasattr(maxregion,'eccentricity'): 
      eccentricity = maxregion.eccentricity
    
    equivalent_diameter = 0.0
    if hasattr(maxregion,'equivalent_diameter'): 
      equivalent_diameter = maxregion.equivalent_diameter
    
    extent = 0.0
    if hasattr(maxregion,'extent'): 
      extent = maxregion.extent
      
    filled_area = 0.0
    if hasattr(maxregion,'filled_area'): 
      filled_area = maxregion.filled_area
      
    orientation = 0.0
    if hasattr(maxregion,'orientation'): 
      orientation = maxregion.orientation
      
    solidity = 0.0
    if hasattr(maxregion,'solidity'): 
      solidity = maxregion.solidity
    
    nregions = len(region_list)
        
    inertia_te = np.zeros((2,1))
    if hasattr(maxregion,'inertia_tensor_eigvals'):
      inertia_te[0] = maxregion.inertia_tensor_eigvals[0]
      inertia_te[1] = maxregion.inertia_tensor_eigvals[1]
      
    moments_hu = np.zeros((7,1))
    if hasattr(maxregion,'inertia_tensor_eigvals'):
      moments_hu[0] = maxregion.moments_hu[0]
      moments_hu[1] = maxregion.moments_hu[1]
      moments_hu[2] = maxregion.moments_hu[2]
      moments_hu[3] = maxregion.moments_hu[3]
      moments_hu[4] = maxregion.moments_hu[4]
      moments_hu[5] = maxregion.moments_hu[5]
      moments_hu[6] = maxregion.moments_hu[6]
      
    centroid = np.zeros((2,1))
    if hasattr(maxregion,'centroid'):
      centroid[0] = maxregion.centroid[0]
      centroid[1] = maxregion.centroid[1]
    
    
    return imdilated,ratio,area,euler_number,perimeter,convex_area,eccentricity,equivalent_diameter,extent,filled_area,orientation,solidity,nregions,inertia_te,moments_hu,centroid


def multiclass_log_loss(y_true, y_pred, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    https://www.kaggle.com/wiki/MultiClassLogLoss

    Parameters
    ----------
    y_true : array, shape = [n_samples]
            true class, intergers in [0, n_classes - 1)
    y_pred : array, shape = [n_samples, n_classes]

    Returns
    -------
    loss : float
    """
    predictions = np.clip(y_pred, eps, 1 - eps)
    # normalize row sums to 1
    predictions /= predictions.sum(axis=1)[:, np.newaxis]
    actual = np.zeros(y_pred.shape)
    n_samples = actual.shape[0]
    actual[np.arange(n_samples), y_true.astype(int)] = 1
    vectsum = np.sum(actual * np.log(predictions))
    loss = -1.0 / n_samples * vectsum
    return loss



  

def testRBM():
  X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
  print X
  model = BernoulliRBM(n_components=2)
  model.fit(X)
  print dir(model)
  print model.transform(X)
  print model.score_samples(X)
  print model.gibbs

def calcEntropy(img):
    #hist,_ = np.histogram(img, np.arange(0, 256), normed=True)
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    hist = hist.ravel()/hist.sum()
    #logs = np.nan_to_num(np.log2(hist))
    #print hist
    #plt.plot(hist)
    #plt.show()
    logs = np.log2(hist+1E-5)
    #hist_loghist = hist * logs
    entropy = -1 * (hist*logs).sum()
    return entropy

def entropyJob(row,maxPixel):
      #8bit int grayscale
      img = np.reshape(row*255, (maxPixel, maxPixel)).astype("uint8")
      result2 = np.zeros(img.shape, dtype=np.float32)
      h, w = img.shape
      subwin_size = 5
      for y in xrange(subwin_size, h-subwin_size):
	for x in xrange(subwin_size, w-subwin_size):
	    subwin = img[y-subwin_size:y+subwin_size, x-subwin_size:x+subwin_size]
	    entropy = calcEntropy(subwin)    # Calculate entropy
	    result2.itemset(y,x,entropy)

      result2 = result2.ravel()
      #plt.plot(result2)
      #plt.show()
      return result2

#http://stackoverflow.com/questions/16647116/faster-way-to-analyze-each-sub-window-in-an-image
def makeExtraFeatures(lX,maxPixel):
    print "Make Extra Descriptors"
    tmp = np.apply_along_axis(entropyJob, 1, lX,maxPixel)
    tmp = pd.DataFrame(tmp)
    colnames = [ 'entr'+str(x) for x in xrange(tmp.shape[1]) ]
    tmp.columns = colnames
    tmp = removeLowVariance(tmp)
    print tmp.describe()
    lX = pd.concat([lX,tmp],axis=1)
    print "Shape after entropy features:",lX.shape
    return lX


def rotJob(row,matrix):
    image = row.reshape(maxPixel, maxPixel)
    image_new = cv2.warpAffine(image,matrix,(maxPixel,maxPixel))
    image_new = image_new.ravel()
    return image_new

def flipJob(row,mode):
    image = row.reshape(maxPixel, maxPixel)
    image_new = cv2.flip(image,mode)
    image_new = image_new.ravel()
    return image_new
  

def makeRotation(lX, ly=None,maxPixel=25):
    """
    Make rotations
    """
    imageSize = maxPixel * maxPixel
    
    if ly is not None: ly = ly.values
    names = lX.columns[0:imageSize]
    lX = lX.values[:,0:imageSize]
       
    #Rotation
    M1 = cv2.getRotationMatrix2D((maxPixel/2,maxPixel/2),90,1)
    M2 = cv2.getRotationMatrix2D((maxPixel/2,maxPixel/2),270,1)
    M3 = cv2.getRotationMatrix2D((maxPixel/2,maxPixel/2),180,1)
    matrices = [M1,M2,M2]
    tmp = np.asarray([np.apply_along_axis(rotJob, 1, lX, m) for m in matrices])
    newn = tmp.shape[0]*tmp.shape[1]
    tmp = np.reshape(tmp,(newn,maxPixel*maxPixel))

    #Flip
    mode = [1,0]
    tmp2 = np.asarray([np.apply_along_axis(flipJob, 1, lX, m) for m in mode])
    newn = tmp2.shape[0]*tmp2.shape[1]
    tmp2 = np.reshape(tmp2,(newn,maxPixel*maxPixel))
    print "Shape tmp2:",tmp2.shape
    
    #showImage(lX[44],maxPixel,10)
    #showImage(tmp[44],maxPixel,10)
    lX = np.concatenate((lX,tmp),axis=0)
    lX = np.concatenate((lX,tmp2),axis=0)
    lX = pd.DataFrame(lX,columns=names)
    if ly is not None: ly = pd.Series(np.concatenate([ly for _ in range(6)], axis=0))
  
    print "Shape X after rotation:",lX.shape
    
    if ly is not None:
      print "Shape y after rotation:",ly.shape
      return lX,ly
    else:
      return lX
    


def showImage(row,maxPixel,fac=10,matrix=True):
    
    if matrix:
      #image = (row * 255).astype("uint8") 
      image = row
    else:
      #image = (row * 255).reshape((maxPixel, maxPixel)).astype("uint8") 
      image = row.reshape((maxPixel, maxPixel)).astype("uint8")
	
    newx,newy = image.shape[1]*fac,image.shape[0]*fac #new size (w,h)
    newimage = cv2.resize(image,(newx,newy))
    
    cv2.imshow("row", newimage)
    cv2.waitKey(0)


def nudge_dataset(lX, ly,maxPixel=25,dilation=4):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the 8x8 images in X around by 1px to left, right, down, up
    """
    imageSize = maxPixel * maxPixel
    
    ly = ly.values
    #print ly.colnames
    
    names = lX.columns[0:imageSize]
    lX = lX.values[:,0:imageSize]
    
    
    print "Shape before nudging:"
    print lX.shape
    
    direction_vectors = [
        [[0, 1, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [1, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 1],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 0]]]

    shift = lambda x, w: convolve(x.reshape((maxPixel, maxPixel)), mode='constant',
                                  weights=w).ravel()
    lX = np.concatenate([lX] +
                       [np.apply_along_axis(shift, 1, lX, vector)
                        for vector in direction_vectors])
		       
    
    
    lX = pd.DataFrame(lX,columns=names)
    
    #recompute other features  
    print "Shape after nudging:"
    print X_new.shape
    ly = pd.Series(np.concatenate([ly for _ in range(4)], axis=0))
    return X_new, ly

#def makeAddFeats(lX,maxPixel,dilation):
    #imageSize = maxPixel * maxPixel
    ##recompute other features
    #print "recompute additional features!"
    #num_features = lX.shape[1] + 12 # for our ratio
    #X_new = np.zeros((lX.shape[0], num_features), dtype=float)
    #for i,image in enumerate(lX.values):
	#image = image.reshape(maxPixel, maxPixel)
	#image_dilated,axisratio,area,euler_number,perimeter,convex_area,eccentricity,equivalent_diameter,extent,filled_area,orientation,solidity,nregions = getImageFeatures(image,dilation)
	#X_new[i, 0:imageSize] = np.reshape(image, (1, imageSize))
	#X_new[i, imageSize] = axisratio
	#X_new[i, imageSize+1] = area
	#X_new[i, imageSize+2] = euler_number
	#X_new[i, imageSize+3] = perimeter
	#X_new[i, imageSize+4] = convex_area
	#X_new[i, imageSize+5] = eccentricity
	#X_new[i, imageSize+6] = equivalent_diameter
	#X_new[i, imageSize+7] = extent
	#X_new[i, imageSize+8] = filled_area
	#X_new[i, imageSize+9] = orientation
	#X_new[i, imageSize+10] = solidity
	#X_new[i, imageSize+11] = nregions
	#if i%5000==0:
	  #print "Feature for image %4d "%(i)
    ## Loop through the classes two at a time and compare their distributions of the Width/Length Ratio
    #colnames = [ "p"+str(x+1) for x in xrange(X_new.shape[1]-12)]
    #colnames = colnames + ['axisratio','area','euler_number','perimeter','convex_area','eccentricity','equivalent_diameter','extent','filled_area','orientation','solidity','nregions']
    ##Create a DataFrame object to make subsetting the data on the class 
    #X_new = pd.DataFrame(X_new)
    #X_new.columns = colnames 
    #return(X_new)
  


def showClasses(Xtrain, ytrain, shownames=None,class_names=None,maxPixel=25,fac=10,nexamples=5):
  imageSize = maxPixel * maxPixel
  for name in shownames:
    if name in class_names:
      label = class_names.index(name)
      print name
      idx = ytrain == label
      Xtmp = Xtrain.loc[idx,:].values
      Xtmp = Xtmp[:,0:imageSize]
      ytmp = ytrain.loc[idx]
      # randomly select a few of the test instances
      for i in np.random.choice(np.arange(0, len(ytmp)), size = (nexamples,)):     
	image = (Xtmp[i] * 255).reshape((maxPixel, maxPixel)).astype("uint8")     
	
	newx,newy = image.shape[1]*fac,image.shape[0]*fac #new size (w,h)
	newimage = cv2.resize(image,(newx,newy))
	
	cv2.imshow(name, newimage)
	cv2.waitKey(0)
  

#@profile
def buildModelMLL(clf,lX,ly,class_names,trainFull=True):
  print "Training the model..."
  print clf
  ly = ly.values
  lX = lX.values
  multiplier = 4
  cv = StratifiedKFold(ly, n_folds=5)
  #create vector with length of lX.shape[0] but with 1,2,3,...n,1,2,3,...n,1,2,3,
  #we do not want to have leakage in cv
  #labels = np.repeat(numpy.random.shuffle(numpy.arange(lX.shape[0]/4),lX.shape[0]/multiplier)
  labels = np.tile(np.random.randint(0,5,lX.shape[0]/4),4)
  print labels
  print "shape labels:",labels.shape
  
  #cv = LeavePLabelOut(labels,1)
  #cv = StratifiedShuffleSplit(ly, n_iter=10, test_size=0.5)
  cv =StratifiedKFold(ly,5)
   
  ypred = ly * 0
  yproba = np.zeros((len(ly),len(set(ly))))
  
  for i,(train, test) in enumerate(cv):
      print "train set: ",i," shape: ",lX[train,:].shape
      ytrain, ytest = ly[train], ly[test]
      clf.fit(lX[train,:], ytrain)
      ypred[test] = clf.predict(lX[test,:])
      yproba[test] = clf.predict_proba(lX[test,:])
      
  print classification_report(ly, ypred, target_names=class_names)
  mll = multiclass_log_loss(ly, yproba)
  print "multiclass logloss: %6.2f" %(mll)
  #training on all data
  if trainFull:
    clf.fit(lX, ly)
  return(clf)


def makeSubmission(model,Xtest,class_names,filename='subXX.csv',zipping=False):
  print "Preparing submission..."
  ref = pd.read_csv('competition_data/submissions/cxx_preds2.csv', sep=",", na_values=['?'],index_col=0)
  nrows = ref.shape[0]
  
  if (Xtest.shape[0]>nrows):
    n_frames = Xtest.shape[0] / nrows
    print "We have to average prediction results on:", n_frames," replications."   
    preds_all = np.zeros((n_frames,nrows,ref.shape[1]))
    #tmp = Xtest.values
    #tmp = np.reshape(tmp, (rows, ref.shape[1]*n_frames), order='F')    
    #print "We have to average the predictions for ",n_frames," data frames."
    #print "New frame:",tmp.shape # should be
    idx_s = 0
    idx_end = nrows
    for i in xrange(n_frames):
	Xtest_act = Xtest.iloc[idx_s:idx_end,:]
	preds_all[i,:,:] = model.predict_proba(Xtest_act)
	idx_s = idx_end
	idx_end = idx_end + nrows
    
    preds = np.mean(preds_all,axis=0)
    print "Final shape:",preds.shape
	
  else:  
    preds = model.predict_proba(Xtest)
  
  directory_names = set(glob.glob(os.path.join("competition_data/test", "*")))
  directory_names = sorted(directory_names)
  row_names = [os.path.basename(x) for x in directory_names]
  #print row_names
  #print outdata.index
  #print class_names

  preds = pd.DataFrame(preds,index=row_names,columns=class_names)

  print preds.describe()
  
  filename = 'competition_data/submissions/'+filename
  preds.to_csv(filename,index_label='image')
  
  if zipping:
    with zipfile.ZipFile(filename.replace("csz","zip"), 'w') as myzip:
	myzip.write(filename)
  
  
  print ref.describe()
  
  ax1 = ref.iloc[:,2:10].hist(bins=40)
  plt.title('ref')
  ax2 = preds.iloc[:,2:10].hist(bins=40)
  plt.show()
  
  

def checkSubmission(subfile='cxx_standard2.csv'):
  df1 = pd.read_csv('/media/loschen/DATA/plankton_data/submissions/sub18012015.csv',index_col=0)
  df2 = pd.read_csv(subfile,index_col=0)
  
  ax1 = df1.iloc[:,2:10].hist(bins=40)
  plt.title('ref')
  ax2 = df2.iloc[:,2:10].hist(bins=40)
  plt.show()
  
  
  

if __name__=="__main__":
  np.random.seed(42)
  
  t0 = time() 
  print "numpy:",np.__version__
  print "pandas:",pd.__version__
  print "sklearn:",sl.__version__
  
  pd.set_option('display.max_columns', 14)
  pd.set_option('display.max_rows', 40)
  
  location='train'
  subset=None
  loadData=False
  extractFeatures=False
  nudgeData=False
  maxPixel=25#25
  doSVD=None
  subsample=None
  dilation=5
  dokmeans=None
  randomRotate=False
  useOnlyFeats=False
  stripFeats=True
  createExtraFeatures=False
  standardize=True
  patchWork=True
  
  #checkSubmission('/home/loschen/programs/cxxnet/example/kaggle_bowl/cxx_standard2.csv')
  #sys.exit(1)
  
  Xtrain, ytrain, class_names,Xtest = prepareData(subset=subset,loadData=loadData,extractFeatures=extractFeatures,maxPixel = maxPixel,doSVD=doSVD,subsample=subsample,nudgeData=nudgeData,dilation=dilation,kmeans=dokmeans,randomRotate=randomRotate,useOnlyFeats=useOnlyFeats,stripFeats=stripFeats,createExtraFeatures=createExtraFeatures,patchWork=patchWork,standardize=standardize,location=location)
  print Xtrain.describe()
  model =  RandomForestClassifier(n_estimators=500,max_depth=None,min_samples_leaf=5,n_jobs=4,criterion='gini', max_features='auto',oob_score=False)
  #model = SGDClassifier(loss="log", eta0=1.0, learning_rate="constant",n_iter=5, n_jobs=4, penalty=None, shuffle=False)#~percetpron
  #model = DBN([Xtrain.shape[1], 100, -1],learn_rates = 0.3,learn_rate_decays = 0.9,epochs = 10,verbose = 1)
  #model = GradientBoostingClassifier(loss='deviance',n_estimators=100, learning_rate=0.1, max_depth=2,subsample=.5,verbose=False)
  #model = LogisticRegression(C=100.0)
  #model = SGDClassifier(alpha=0.1,n_iter=25,shuffle=True,loss='log',penalty='l2',n_jobs=4,verbose=True)#mll=4.38?
  #model = Pipeline(steps=[('rbm', BernoulliRBM(n_components =300,learning_rate = 0.1,n_iter=15, random_state=0, verbose=True)), ('lr', LogisticRegression())])
  #model = LDA()#6.38
  
  model = buildModelMLL(model,Xtrain,ytrain,class_names,trainFull=False)
  
  #with open("tmp.pkl", "w") as f: pickle.dump(model, f)
  
  #with open("tmp.pkl", "r") as f: model = pickle.load(f)  
  
  
  #makeSubmission(model,Xtest,class_names,"sub10022015a.csv") 
  
  
  names=['amphipods','appendicularian_straight','artifacts']
  #names=['tunicate_doliolid_nurse','trichodesmium_multiple','siphonophore_other_parts','fish_larvae_deep_body','hydromedusae_haliscera_small_sideview']
  #showClasses(Xtrain, ytrain, shownames=names,class_names=class_names,maxPixel=maxPixel)
  
  
  #testRBM()
  print("Model building done on in %fs" % (time() - t0))
  plt.show()