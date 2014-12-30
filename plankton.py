#!/usr/bin/python 
# coding: utf-8

#http://nbviewer.ipython.org/github/udibr/datasciencebowl/blob/master/141215-tutorial.ipynb
#Import libraries for doing image analysis
#http://www.pyimagesearch.com/2014/09/22/getting-started-deep-learning-python/
#mklhttp://verahill.blogspot.de/2013/06/465-intel-mkl-math-kernel-library-on.html

import glob
import os

from skimage.io import imread
from skimage.transform import resize
from skimage import segmentation
from skimage.morphology import watershed
from skimage import measure
from skimage import morphology
from skimage.feature import peak_local_max
from scipy import ndimage

from nolearn.dbn import DBN
import cv2

from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.lda import LDA

from sklearn.neural_network import BernoulliRBM

from matplotlib import pyplot as plt
from matplotlib import colors
from pylab import cm

import numpy as np
import pandas as pd
import sklearn as sl

from qsprLib import *

def prepareData(extractFeatures=False,useRatio=True,maxPixel = 48,doSVD=100):
  
  #print directory_names
  if extractFeatures: createFeatures(maxPixel)
  
  
  df = pd.read_csv('competition_data/train_'+str(maxPixel)+'.csv', sep=",", na_values=['?'],index_col=0)

  #df.loc[:,['ratio','label']].hist()
  
  if not useRatio:
    df.drop(['ratio'],axis=1,inplace=True)
  
  idx = df.shape[1]-1
  X = df.iloc[:,0:idx]
  
  if doSVD is not None:
	print "Singular value decomposition..."
	tsvd=TruncatedSVD(n_components=doSVD, algorithm='randomized', n_iter=5, tol=0.0)
	X_SVD=tsvd.fit_transform(X)
	X=pd.DataFrame(np.asarray(X_SVD),index=X.index)
	X.to_csv('competition_data/train_SVD'+str(maxPixel)+'.csv')
  
  
  y = df.iloc[:,-1]
  class_names = getClassNames()
 
  return X,y,class_names


def getClassNames():
    namesClasses=[]
    directory_names = getDirectoryNames()
    for label,folder in enumerate(directory_names):
	currentClass = os.path.basename(folder)
	namesClasses.append(currentClass)
    return namesClasses

def getDirectoryNames():
    directory_names = list(set(glob.glob(os.path.join("competition_data","train", "*"))).difference(set(glob.glob(os.path.join("competition_data","train","*.*")))))
    return sorted(directory_names)

def createFeatures(maxPixel = 25):
    print "Creating features",
    directory_names = getDirectoryNames()
    #get the total training images
    numberofImages = 0
    for folder in directory_names:
	for fileNameDir in os.walk(folder):   
	    for fileName in fileNameDir[2]:
		# Only read in the images
		if fileName[-4:] != ".jpg":
		  continue
		numberofImages += 1

    print ", number of images:",numberofImages
    
    # We'll rescale the images to be 25x25
    imageSize = maxPixel * maxPixel
    num_rows = numberofImages # one row for each image in the training dataset
    num_features = imageSize + 12 # for our ratio
    
    X = np.zeros((num_rows, num_features), dtype=float)
    y = np.zeros((num_rows,1), dtype=int)

    files = []
    print "Reading images"
    # Navigate through the list of directories
    i = 0
    for label,folder in enumerate(directory_names):
	# Append the string class name for each class
	currentClass = folder.split(os.pathsep)[-1]
	print "Label: %4d Class: %-32s"%(label,currentClass)
	for fileNameDir in os.walk(folder):
	    for fileName in fileNameDir[2]:
		# Only read in the images
		if fileName[-4:] != ".jpg":
		  continue
		
		# Read in the images and create the features
		nameFileImage = "{0}{1}{2}".format(fileNameDir[0], os.sep, fileName)            
		image = imread(nameFileImage, as_grey=True)
		files.append(nameFileImage)
		axisratio,area,euler_number,perimeter,convex_area,eccentricity,equivalent_diameter,extent,filled_area,orientation,solidity,nregions = getImageFeatures(image)
		image = resize(image, (maxPixel, maxPixel))

		# Store the rescaled image pixels and the axis ratio
		X[i, 0:imageSize] = np.reshape(image, (1, imageSize))
		X[i, imageSize] = axisratio
		X[i, imageSize+1] = area
		X[i, imageSize+2] = euler_number
		X[i, imageSize+3] = perimeter
		X[i, imageSize+4] = convex_area
		X[i, imageSize+5] = eccentricity
		X[i, imageSize+6] = equivalent_diameter
		X[i, imageSize+7] = extent
		X[i, imageSize+8] = filled_area
		X[i, imageSize+9] = orientation
		X[i, imageSize+10] = solidity
		X[i, imageSize+11] = nregions

		# Store the classlabel
		y[i] = label
		i += 1
		#print "row: %4d label: %4d"%(i,label)


    # Loop through the classes two at a time and compare their distributions of the Width/Length Ratio
    colnames = [ "p"+str(x+1) for x in xrange(X.shape[1]-12)]
    colnames.append('ratio')
    colnames.append('area')
    colnames.append('euler_number')
    colnames.append('perimeter')
    colnames.append('convex_area')
    colnames.append('eccentricity')
    colnames.append('equivalent_diameter')
    colnames.append('extent')
    colnames.append('filled_area')
    colnames.append('orientation')
    colnames.append('solidity')
    colnames.append('nregions')
    
    colnames.append('label')
    #Create a DataFrame object to make subsetting the data on the class 
    df = pd.DataFrame(X)
    y = pd.DataFrame(y)
    df = pd.concat([df, y],axis=1,ignore_index=True)
    print df.shape
    df.columns = colnames 
    df.label.hist()
    df.to_csv('competition_data/train_'+str(maxPixel)+'.csv')

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
  
  
def getImageFeatures(image):
    #http://scikit-image.org/docs/dev/api/skimage.measure.html#regionprops
    image = image.copy()
    # Create the thresholded image to eliminate some of the background
    imagethr = np.where(image > np.mean(image),0.,1.0)

    #Dilate the image
    imdilated = morphology.dilation(imagethr, np.ones((4,4)))

    # Create the label list
    label_list = measure.label(imdilated)
    label_list = imagethr*label_list
    label_list = label_list.astype(int)
    
    region_list = measure.regionprops(label_list)
    maxregion = getLargestRegion(region_list, label_list, imagethr)

    # guard against cases where the segmentation fails by providing zeros
    ratio = 0.0
    if ((not maxregion is None) and  (maxregion.major_axis_length != 0.0)):
        ratio = 0.0 if maxregion is None else  maxregion.minor_axis_length*1.0 / maxregion.major_axis_length
    
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
    
    return ratio,area,euler_number,perimeter,convex_area,eccentricity,equivalent_diameter,extent,filled_area,orientation,solidity,nregions


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


#@profile
def buildModelMLL(clf,lX,ly,class_names):
  print "Training the model..."
  print clf
  ly = ly.values
  lX = lX.values
  
  cv = StratifiedKFold(ly, n_folds=5)
   
  ypred = ly * 0
  yproba = np.zeros((len(ly),len(set(ly))))
  
  for train, test in cv:
      ytrain, ytest = ly[train], ly[test]
      clf.fit(lX[train,:], ytrain)
      ypred[test] = clf.predict(lX[test,:])
      yproba[test] = clf.predict_proba(lX[test,:])
      
  print classification_report(ly, ypred, target_names=class_names)
  mll = multiclass_log_loss(ly, yproba)
  print "multiclass logloss: %6.2f" %(mll)
  

def testRBM():
  X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
  print X
  model = BernoulliRBM(n_components=2)
  model.fit(X)
  print dir(model)
  print model.transform(X)
  print model.score_samples(X)
  print model.gibbs


def nudge_dataset(X, Y):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the 8x8 images in X around by 1px to left, right, down, up
    """
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

    shift = lambda x, w: convolve(x.reshape((8, 8)), mode='constant',
                                  weights=w).ravel()
    X = np.concatenate([X] +
                       [np.apply_along_axis(shift, 1, X, vector)
                        for vector in direction_vectors])
    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return X, Y


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
  

if __name__=="__main__":
  np.random.seed(42)
  
  t0 = time() 
  print "numpy:",np.__version__
  print "pandas:",pd.__version__
  print "sklearn:",sl.__version__
  
  pd.set_option('display.max_columns', 14)
  pd.set_option('display.max_rows', 40)
  
  extractFeatures=False
  useRatio=True
  maxPixel=36#25
  doSVD=None
  
  Xtrain, ytrain, class_names = prepareData(extractFeatures,useRatio,maxPixel,doSVD)
  print Xtrain
  #model =  RandomForestClassifier(n_estimators=100,max_depth=None,min_samples_leaf=5,n_jobs=4,criterion='gini', max_features='auto',oob_score=False)
  
  #model = DBN([Xtrain.shape[1], 100, -1],learn_rates = 0.3,learn_rate_decays = 0.9,epochs = 10,verbose = 1)
  #model = GradientBoostingClassifier(loss='deviance',n_estimators=100, learning_rate=0.1, max_depth=2,subsample=.5,verbose=False)
  #model = LogisticRegression(C=100.0)
  #model = SGDClassifier(alpha=0.1,n_iter=25,shuffle=True,loss='log',penalty='l2',n_jobs=4,verbose=True)#mll=4.38?
  #model = Pipeline(steps=[('rbm', BernoulliRBM(n_components =300,learning_rate = 0.1,n_iter=15, random_state=0, verbose=True)), ('lr', LogisticRegression())])
  #model = LDA()#6.38
  
  buildModelMLL(model,Xtrain,ytrain,class_names)
  
  names=['tunicate_doliolid_nurse','trichodesmium_multiple','siphonophore_other_parts','fish_larvae_deep_body','hydromedusae_haliscera_small_sideview']
  #showClasses(Xtrain, ytrain, shownames=names,class_names=class_names,maxPixel=maxPixel)
  
  
  #testRBM()
  print("Model building done on in %fs" % (time() - t0))
  plt.show()