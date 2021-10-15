import geodist
import seg
from scipy.misc import imread
import matplotlib.pyplot as plt
from roipoly import RoiPoly
import numpy as np
import sys
import plotly.plotly as py
import plotly.graph_objs as go
import matplotlib.image as mpimg

import matplotlib.pylab as plb
import cv2

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

prob = 2
if prob==1:
	img = mpimg.imread('shapes.png')
	im = rgb2gray(img)  
elif prob==2:
	img = mpimg.imread('chest.jpg')   
	im = rgb2gray(img)  
else:
	sys.exit('Need to add more examples')

print("\nImage segmentation (problem {0})\n".format(prob))

im = im.astype(float)
n,m = im.shape

#scale to [0,1]
im = im-im.min()
im = im/im.max() 

# Show the image
fig = plt.figure()
plt.imshow(im, cmap=plt.get_cmap('gray'))
plt.title("Click 3 points for a triangle --  DOUBLE click to end")
plt.plot()

# ROI of OpenCv
point = plb.ginput(3)
x = []
y = []
for temp in point:
    x.append(temp[0])
    y.append(temp[1])

rc = np.array((x, y)).T
mask = np.zeros([n,m], dtype= 'float')
cv2.drawContours(mask, [rc.astype(int)],0,255,-1)

plt.title('Ending initilization -> compute d=d(x,y) next')
plt.show()

gd = geodist.geodist(im,mask,-1, -1) #geodesic distance
geodist.draw(gd)
#np.savetxt('geodesic', gd)
 
#lambd = 1 intensity fitting term parameter ( (z-c1)^2 - (z-c2)^2 )
#theta = 8 distance constraint parameter (geodesic distance)
u = seg.seg(im,1,8,mask,gd)