from scipy.misc import imread
import sys

inputfile = sys.argv[1]
filename = sys.argv[2]
im = imread(inputfile, mode='L')
print(im.shape)
with open(filename, "w") as f:
    f.write('%d'%im.shape[0])
    f.write(" ") 
    f.write('%d'%im.shape[1])
    f.write("\n") 
    for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                f.write('%d'%im[i][j])
                f.write(" ")
            f.write('\n')
                
