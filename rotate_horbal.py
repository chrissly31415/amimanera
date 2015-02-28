#!/usr/bin/python 
# coding: utf-8

import os
import glob
import sys
from datetime import datetime
import PIL
from PIL import Image,ImageFilter
import matplotlib.pyplot as plt

#Usage: python ./rotate_images.py /home/xxx/data/train
rotation_angles = [90, 180, 270]
 
def create_image_rotations(image):
    whitec = PIL.ImageColor.getrgb("white")
    
    imgs = [(image.convert("RGBA").rotate(a, expand = 1), a) for a in rotation_angles]
        
    bgs =  [Image.new("RGB", im.size, whitec) for (im,a) in imgs]
    #print bg.size
    for bg,(im,a) in zip(bgs,imgs):
	bg.paste(im, im)
    imgs =  [(bg,a) for bg,(im,a) in zip(bgs,imgs)]  
    
    #create mirrors
    img_mirror = image.convert("RGBA").transpose(PIL.Image.FLIP_LEFT_RIGHT)
    img_mirror = (img_mirror, 'fliplr')
    imgs.append(img_mirror)
    
    img_mirror2 = image.convert("RGBA").transpose(PIL.Image.FLIP_TOP_BOTTOM)
    img_mirror2 = (img_mirror2, 'fliptb')
    imgs.append(img_mirror2)
    
    return imgs

    
def getDirectoryNames(location="train"):
    directory_names = list(set(glob.glob(os.path.join(location, "*"))).difference(set(glob.glob(os.path.join(location,"*.*")))))
    return sorted(directory_names)

if __name__ == '__main__':  
    gradient_only=False
    if len(sys.argv) < 2:
	raise ValueError('Must pass directory of images as the first parameter')

    img_dir = getDirectoryNames(sys.argv[1])
    if not img_dir:
       img_dir = [sys.argv[1]]
    
    for img_data in img_dir:
	filenames = [ os.path.join(img_data,f) for f in os.listdir(img_data) if os.path.isfile(os.path.join(img_data,f)) ]
	n_images = len(filenames)
	print 'Processing %i images in %s' % (n_images,img_data)
    
	start_time = datetime.now()
    
	for i, imgf in enumerate(filenames):
	    spimgf = imgf.split('/')
	    image_path = '/'.join(spimgf[:-1])
	    image_file = spimgf[-1].split('.')[0]
	  	  
	    img = Image.open(imgf)
	    
	    if gradient_only:
	      img = img.filter(ImageFilter.FIND_EDGES)
	      img.save(image_path + '/' + image_file + '.jpg')
	    else:
	      rimgs = create_image_rotations(img)
	      for rimg, rot in rimgs:
		  rimg.save(image_path + '/' + image_file + '_mod' + str(rot) + '.jpg')
    
	    if ((i+1) % 10000) == 0:
		print 'Processed %i files in %is' % (i+1, (datetime.now() - start_time).seconds)
