from scipy.misc import imsave
import sys

infile = sys.argv[1]
outfile = sys.argv[2]

with open(infile, "r") as ifile:
    im_info = ifile.readline().split(" ")
    height = im_info[0]
    width = im_info[1]
    image_arr = []
    for i in range(int(height)):
        imline = ifile.readline().split(" ")
        image_arr.append([])
        for j in range(int(width)):
            image_arr[i].append(float(imline[j]))
    imsave(outfile, image_arr)
            
        
