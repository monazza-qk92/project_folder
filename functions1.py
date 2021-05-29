import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import copy
def kmeans(k, df):
 max_x = max(df['x'])
 max_y = max(df['y'])
# centroids[i] = [x, y]
 centroids = {
 i+1: [np.random.randint(0, max_x), np.random.randint(0, max_y)]
 for i in range(k)
  }
 print(centroids)
## Assignment Stage
def assignment(df, centroids):
 for i in centroids.keys():
# sqrt((x1 - x2)^2 - (y1 - y2)^2)
  df['distance_from_{}'.format(i)] = (np.sqrt((df['x'] - centroids[i][0]) ** 2 + (df['y'] - centroids[i][1]) ** 2 ) )
  centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
  df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
  df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
  return df
df = assignment(df, centroids)
print(centroids)
## Update Stage
old_centroids = copy.deepcopy(centroids)
def update(k):
 for i in centroids.keys():
  me = np.mean(df[df['closest'] == i]['x'])
  if(me>0):
   centroids[i][0] = me
   mee = np.mean(df[df['closest'] == i]['y'])
   if (mee > 0):
     centroids[i][1] = mee
     return k
 centroids = update(centroids)
 print(centroids)
## Repeat Assigment Stage
df = assignment(df, centroids)
# Continue until all assigned categories don't change any more
while True:
 closest_centroids = df['closest'].copy(deep=True)
 centroids = update(centroids)
 df = assignment(df, centroids)
 if closest_centroids.equals(df['closest']):
break
print(centroids)
return df, centroids

#r_b extration
def r_b_pixels(strokeimg):
[xs, ys] = strokeimg.size
r_xind = []
r_yind = []
b_xind = []
b_yind = []
rindalt=0
bindalt=0
for x in range(0, xs):
for y in range(0, ys):
# (4) Get the RGB color of the pixel
[r, g, b] = strokeimg.getpixel((x, y))
if (r == 255):
r_xind.insert(rindalt, x);
r_yind.insert(rindalt, y);
rindalt = rindalt + 1;
if (b == 255):
b_xind.insert(bindalt, x);
b_yind.insert(bindalt, y);
bindalt = bindalt + 1;
r_df = pd.DataFrame({'x': r_xind, 'y': r_yind})
b_df = pd.DataFrame({'x': b_xind, 'y': b_yind})
return r_df, b_df,xs

 #Computing Wk
def Wk(xcentroids,k, xdf,xs):
lenCent = []
wk = []
lindalt = 0
wkindalt=0
centroiddatacontn = copy.deepcopy(xcentroids)
for x in range(k):
centroiddatacontn[x + 1][0] = (xdf[xdf['closest'] == x + 1]['x'])
lenCent.insert(lindalt, len(centroiddatacontn[x + 1][0]));
lindalt = lindalt + 1;
for x in range(k):
w = lenCent[x] / xs;
wk.insert(wkindalt, w);
wkindalt = wkindalt + 1;
return wk
#for Ck pixels
def Ck(k,oimg,xcentroids):
Ckval = []
Ckvalind = 0
for x in range(k):
Ckval.insert(Ckvalind, oimg.getpixel((xcentroids[x+1][0],xcentroids[x+1][1])))
Ckvalind = Ckvalind + 1;
return Ckval
 #p
def p_of_oimPixels(oimg,k,Ckval,wk):
[oxs, oys] = oimg.size
IpMinusCk = []
IpMinusCkindex = 0
prob = []
pindex = 0
for x in range(100, 140):
for y in range(140, 180):
[r,g,b] = oimg.getpixel((x, y))
for z in range(k):
[rr, gg, bb] = Ckval[z]
dist = (r-rr)** 2+(g-gg)** 2+(b-bb)** 2

# dist = numpy.linalg.norm(a - b)
expval = math.exp(-1*(dist))
p = wk[z]*(expval);
IpMinusCk.insert(IpMinusCkindex, p);
IpMinusCkindex = IpMinusCkindex +1;
IpMinusCkindex=0;
prob.insert(pindex, sum(IpMinusCk))
pindex = pindex + 1
pindex = 0
print(prob)
return prob,oxs,oys

def fg_bg_assign(r_prob,b_prob):
assign = []
for a in range(len(r_prob)):
if (r_prob[a]>b_prob[a]):
assign.insert(a, 1);
else:
assign.insert(a, 0);
return assign

def show_fg(oxs,oys,oimg,assign):
pixind = 0;
for x in range(100, 140):
for y in range(140, 180):
if (assign[pixind] == 0):
oimg.putpixel((x, y), 0)
pixind = pixind + 1
return oimg

def show_bg(oxs,oys,oimg,assign):
pixind = 0;
for x in range(100, 140):
for y in range(140, 180):
if (assign[pixind] == 1):
oimg.putpixel((x, y), 0)
pixind = pixind + 1
return oimg
