from functions import *
from PIL import Image

strokeimgs = ['dance stroke 1.PNG','dance stroke 2.PNG','dog stroke.PNG','lady stroke 1.PNG','lady stroke 2.PNG',
              'Mona-lisa stroke 1.PNG','Mona-lisa stroke 2.PNG','van Gogh stroke.JPG']
oimgs = ['dance.PNG','dance.PNG','dog.PNG','lady.PNG','lady.PNG','Mona-lisa.PNG','Mona-lisa.PNG','van Gogh.PNG']

for ii in range(8):
    strokeimg = Image.open(strokeimgs[ii])
    oimg = Image.open(oimgs[ii])
    k = 64

    r_df, b_df,xs = r_b_pixels(strokeimg)
    #for red pixels i.e foreground
    rdf, rcentroids = kmeans(k, r_df)
    rwk = Wk(rcentroids,k, rdf,xs)
    rCkval = Ck(k,oimg,rcentroids)
    r_prob,oxs,oys = p_of_oimPixels(oimg,k,rCkval,rwk)

    #for blue pixels i.e background
    bdf, bcentroids = kmeans(k, b_df)
    bwk = Wk(bcentroids,k, bdf,xs)
    bCkval = Ck(k,oimg,bcentroids)
    b_prob,oxs,oys = p_of_oimPixels(oimg,k,bCkval,bwk)

    #assigning
    assign = fg_bg_assign(r_prob,b_prob)

    fgImg_copy = oimg.copy()
    bgImg_copy = oimg.copy()
    #show foreground image
    fg_oimg = show_fg(oxs,oys,fgImg_copy,assign)
    fg_oimg.show()
    #show background image
    bg_oimg = show_bg(oxs,oys,bgImg_copy,assign)
    bg_oimg.show()