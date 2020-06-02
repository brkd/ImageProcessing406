# ImageProcessing406

CS406 Project – Final Report

Parallelized Digital Image Processing

Ege Şenses 23570, Berkay Demireller 23810, Selen Özcan 23914

GitHub link: https://github.com/brkd/ImageProcessing406

1. Problem Definition
	Digital Image Processing refers to the set of algorithms written for making various operations on images, by using the digital representations of said images. Images are kept in a matrix format where each pixel is located in one call, and carries an integer that decides its intensity. The pixels can be encoded for different color models like grayscale, RGB or CMYK. Every task done for image processing is fundamentally just some calculations, ranging from simple to very complex, on these matrices and pixel values.
	Digital Image Processing has been developing since the 1960s and has become the norm for applying virtually any change or operation onto an image, be it color processing, image recognition or image compression. Its rise and progress have been mostly defined by the developments in Computer Science and Discrete Mathematics, as well as the increase in demands in sectors like environmental science, military, medical science and even agriculture.
	The domain of applications for efficient Digital Image Processing methods have grown substantially in recent history, thus we believed that working on the parallelization of said methods constituted a goal both significant and large in scope.

2. Need for Parallelization
	Image Processing is done with excessive loops that iterate over every single pixel, or in some cases, small batches of pixels. This means that any algorithm used for manipulating images is quite suitable for parallelization. As the resolution of an image increases, so does its size; which means that the need for high performance computing also increases due to how we’ll have more iterations and actions per loop. 
An example of an image processing algorithm that is amply eligible for parallelization is image warping. Image warping refers to significantly distorting the shapes present in an image or video; which is usually done for correcting image distortion or morphing. It is done by applying a transformation on every pixel and replacing the new pixel values with the result of the transformation, which is actually a technique seen regularly in Digital Image Processing.
We have parallelized multiple Digital Image Processing algorithms including Linear Scaling, Median Filter, Sobel Filters, Sobel Edge Detection, Grayworld, Ordered Dithering, Vertical Mid-axis Reflection and Rotations of varying degrees. And we believe we have been able to show that the run-time taken by these processing algorithms have been significantly reduced by employing parallel programming. 

3. Data Summary
	The data that we’ve worked on, in short, is composed of matrices with 8-bit values that represent the brightness of a pixel in grayscale. The values a pixel can take range from 0 for pitch black to 255 for completely white.
	All of our algorithms work on .txt files, so we transpose .jpg files into .txt before any processing is done. The outputs are also given in .txt file, which we convert back into .jpg. The .txt file starts with the dimensions of the image and then lists the values of all of the pixels in 8-bit grayscale:
 
Fig. Image converted to .txt file example
	The algorithms are then free to traverse these files however they need to in order to create their results; some read pixel by pixel whereas others read small KxK blocks at once.
	Luckily, there is an abundance of images available on the net for the purposes of testing, and thus acquiring the test data was not a challenge in any way. In fact, if there was a need for having an image with very specific dimensions, one could simply create said file on their own computer with the right amount of pixels and work on that. We’ve worked with images with sizes ranging from 900x1200 to 10000x10000 and managed to get correct results on all.

4. The Application
	The program takes an image as its input in .txt format, applies multiple image processing algorithms onto it and prints the runtime and the result images. It does so for the sequential,                 CPU-parallel and GPU implementations of the algorithms so that one can compare the results. Python scripts are used for converting .txt and .jpg files into one another. 
	The main program reads the grayscale data from the given.txt file and creates a                                  1-dimensional, “flattened” data structure for keeping the image’s information. After that, this array is fed into multiple algorithms and displays the run-times. If the user demands it, it prints out their outputs as well.
	An example of how we run the CPU code is given below:
 
Fig. Running the CPU code
	Notice that when running the executable, 3 things are given as input:
	An input .txt file that was converted from a .jpg file
	The name for the output files
	1 for printing output, 0 for skipping it
Also, the Python scripts used for converting files are called “txt2im.py” and “im2txt.py”.
 
Fig. Converting .txt and .jpg files with Python scripts
	We explain how exactly to run the code at the end of the report.
Loop parallelism is employed over the “for loops” in the algorithms we’ve written. SIMD versions have been created for most of these algorithms for a vectoral approach as well. Additionally, all of these algorithms except Sobel Edge Deteciton have also been implemented with CUDA for working with GPUs. As one can see from the test results given further down, although CPU parallelism does provide speed-ups, the GPU variants are by far the fastest versions.
	Before moving onto the algorithms, one should make note of what is possibly the most important technique in all of Digital Image Processing, which is Convolution. It operates over almost all kinds of digital signals; speech, image or video. It is also employed for creating a relation between the input and the output of linear systems.
	Convolution creates a new set of pixels from the previously given one by applying linear operations. These linear operations are enforced with a window, also called a Kernel, whose values and size are determined beforehand. A visual way of describing Convolution is imagining putting the Kernel onto the input matrix and multiplying the numbers that overlap before adding them all together. The center value of the Kernel coincides with the source pixel for the calculation. This pixel is then replaced with a weighted sum of itself and neighbors. 
	Different Kernels can be used for achieving different results. For instance, an Identity Kernel that consists of all 0s and only a single 1 in the center does not alter the image in any way. A Kernel where the center value is too high and the neighbors are close to 0 would apply a sharpening filter. A Gaussian Blur Kernel, where the center receives the biggest value and the surrounding values diminish the more they’re further from the center, would cause a blurring effect (Ahn, 2018). The application of the Kernel can be given as follows:
 
Fig. Convolution
4.0. Parallelization and GPGPU
	As it was discussed before, image processing algorithms are matrix operations in nature. For loops are used in high numbers for almost all image processing algorithms and this makes them an easy candidate for parallelization and GPGPU. Since these are simple algorithms, our OpenMP implementations consist of loop level parallelizations and certain private paramters. CUDA implementations make us of both threads and blocks, and it also uses shared memory for median filtering since window usage is very suitable for shared memory. The current CUDA implementation is suitable for larger images because it uses low amount of blocks and high amount of threads. To get better output images with smaller images (especially for median filtering) the block number could be decreased. Thread numbers should be adjusted accordingly if block numbers are decreased, since each pixel index should be accessible using a combination of block and thread IDs. The specifications are left as comments in the source code.

4.1. Linear Scaling
	This algorithm receives an image in grayscale and finds the minimum and maximum values inside it (in our case, we do so as a part of preprocessing and feed the minimum and maximum values as input), call them xmin and xmax. Afterwards, it scales xmin to 0 and xmax to 255, causing every pixel to undergo a similar change. We formulate this as follows:
 
	Linear Scaling helps find a balance in the brightness/contrast of an image. A mostly dark image would have a tiny difference between xmin and xmax, which means inflating the difference to the maximum possible value of 255 helps brighten the image:
 
Fig. Linear Scaling example
4.2. Median Filtering
	This algorithm is regularly used for noise reduction (Huang et al., 1979). Given a window size K, it looks up the KxK matrix around a pixel, finds the median value and replaces the pixel in the center with this median value. Notice that, much like how the Kernel operated in Convolution, the source pixel is fixated in the center of the KxK window. The window size we used is 3.
	Median Filtering usually acts as a pre-processing method before further applications that are very susceptible to noise, like edge detection. Such algorithms amplify the noise greatly, which is why removing said noise beforehand is mandatory.
One should note that although the edge preservation property of Median Filtering is quite preferable in this regard, it runs into a lot of issues on the boundaries of the image. This is because with pixels on the boundary, there are fewer neighbors to process. Especially in corners, our 3x3 window would have only 3 neighbors to look at instead of the regular 8. Hence noise removal from boundaries is not always possible. Repeating the source pixel’s value in the missing values’ stead is one possible workaround. Changing the shape of the window slightly for the boundaries is also a possibility. 

Here’s how Median Filtering removes salt-and-pepper noise:
 
Fig. Median Filtering example
	One can observe that the noise is greatly reduced, however a considerable amount of noise still remains on the boundaries.

4.3. Sobel Filtering
	Named after Irwin Sobel, this method consists of two convolutions, one in the horizontal axis and the other on the vertical. It produces a crude gradient approximation that emphasizes the edges so that one can later feed its input into an edge detection algorithm. 
	Visually, the edges will appear much brighter whereas the rest will appear very dark. The computations of this algorithm is relatively inexpensive since it is practically two separate convolutions. 
	The edge size for the Kernel is 3x3. The Kernels for Sobel Filtering are regularly in the following form:
 
Notice that they’re practically the same Kernel, given in both vertical and horizontal configurations. The most commonly used Kernel has a = 3 and b = 10. The convolution operation is done entirely in the same way as what has been described before.
An example of Sobel Filtering is given below:
 
Fig. Sobel Filtering example
4.4. Sobel Edge Detection
	Following Sobel Filtering, one can take these filtered images and feed them into Sobel Edge Detection. The output will have each edge outlined with bright pixels whereas the rest is pitch black, any other detail is basically lost. In applications regarding robotics and computer vision, edge detection algorithms enable machines to “understand” and interpret what they see.
 
Fig. Sobel Edge Detection example
4.5.GrayWorld
	This algorithm is commonly used for illuminant estimation. The premise is that, if a photo is color-balanced, the average of all the colors is neutral gray. Hence, one can calculate the average pixel value of an image and then calculate a scaling value by calculating the ratio between 127.5 (which is the expected average in a picture that has every shade equally) and the actual average of the image. This scaling value is then used to alter every single pixel. Do note that, since the algorithms we’re using deal with 8-bit grayscale values, the pictures are always in black and white. This does not stop us from implementing a Gray World function and testing it, however the changes it does is not apparent in the final visual result produced since it is in black and white anyway.
	In the example given below in color, one can see that Gray World estimates the actual illuminant of the environment and changes the image to match that:
 
Fig. GrayWorld example

4.6. Reflection
	We’ve scrapped other variants of reflecting algorithms since they fundamentally do the same work. This method places a vertical axis right in the middle of the image and computes the reflection based on that. It is also quite easy to alter this code to take a complete reflection rather than just half, as well as creating a horizontal reflection rather than a vertical one. 

      
Fig. Reflection example

4.7. Ordered Dithering
	Image Dithering is intentionally applying noise unto an image in the hopes of randomizing quantization error so that color banding or other large patterns can be avoided. Ordered Dithering is the variant that follows a pre-determined path. For each pixel, Ordered Dithering offsets its color by using a “threshold map”:    [■(128&0@192&64)]
	This matrix is placed over 2x2 areas of pixels, and the pixels that have larger values than the thresholds are maximized to 256 whereas the others are minimized to 0. Do note that for very large images, the threshold map’s size might need to be increased for better observing the results visually.
	Ordered Dithering is usually employed for reducing the color depth of images. Its computations result in a so-called “crosshatch” pattern.
	An example is given below:
      
Fig. Ordered Dithering example
4.8. Rotations
Lastly, we’ve implemented rotations of degrees 90o, 180o and 270o. Do notice that rotations of degrees 90o and 270o switch the dimensions of the image.
One can apply the following changes for rotating an image:
	For rotating 90o degrees, image[i][j]  image[width-j-1][i].
	For rotating 180o degrees, image[i][j]  image[height-i-1] [width-j-1].
	For rotating 270o degrees, image[i][j]  image[j][height-i-1].
      
Fig. Rotation 180o example

All of these algorithms also have parallel versions and most have additional SIMD versions implemented. The GPU versions of these algorithms have also been implemented and tested, and yield the best runtimes.

5.Results
CPU results:
 

The results for the GPU variants are given below:
 
Fig. GPU Results Average
These are the average of 10 tests, and are measured in seconds.
Enrollment in local colleges, 2005

Fig. CPU Sequential Results 

6. How to Run the Code
	GitHub link: https://github.com/brkd/ImageProcessing406.
	One can find all of the mentioned algorithms and scripts on the GitHub page, however we’ll also list the links for each important page for convenience:
CPU Code: https://github.com/brkd/ImageProcessing406/blob/master/improcess.cpp
GPU Code: https://github.com/brkd/ImageProcessing406/blob/master/GPU/gpimprocess.cu
txt2im.py script: https://github.com/brkd/ImageProcessing406/blob/master/txt2im.py
im2txt.py script: https://github.com/brkd/ImageProcessing406/blob/master/im2txt.py
Some test images: https://github.com/brkd/ImageProcessing406/tree/master/img

Like described above, we compile the code with the following command:
g++ -g -03 -std=c++17 improcess.cpp -o 406im -fopenmp
Once a .jpg file is chosen for input, one can turn this file into a .txt file with the “im2txt.py” script with the following command:
python im2txt.py input.jpg input.txt
Note that the names do not have to be “input.jpg” or “input.txt”. Then, one can run the executable with the following command:
./406im input.txt output 1
The input.txt is the input file, “output” is the name to be given to the output files and “1” means that we want the output printed out.
If 1 was selected instead of 0, we can again use a python script, this time “txt2im.py”, for converting this .txt file into a .jpg file with the following command:
python txt2im.py input.txt input.jpg
 
 
Fig. How to run the code
	For the GPU code, we run the following command:
nvcc -o gpu gpimprocess.cu
	Note that this can yield a “string to char” warning, but you can safely ignore it.
 
Fig. How to compile the GPU code
	Then, assuming the input file has been converted to a .jpg file, one can run the code with the following command:
./gpu input.txt 1
	 “input.txt” is the input file and 1 means “print the output” whereas 0 would mean the opposite.
 
Fig. How to run the GPU code








7. Conclusion and Future Work
Due to the nature of  2D images and their representation, image processing algorithms are very suitable to parallelization and GPGPU programming. Our project intended to show that using parallel algorithms most image processing algorithms can achieve considerable speed-ups and to a certain extend succeded to do so. Some of the algorithms described above behaved contrary to our expectations and resulted in low speed-ups or even in longer execution times. Certain ideas were discussed among team members to improve the implementations so that these algorithms would also have lower execution times but we couldn’t complete the discussed implementations and thus couldn’t achieve better execution times. A future iteration of the project could include those implementations as well. CUDA implementaions surpassed all of the parallel implementations as expected but could be further improved upon with more proper usage of shared memory. Some algorithms were not included in CUDA implementations but they can be easily implemented as well, which can also be included in future iterations. CUDA supports video operations as well and video processing is widely used as it’s one of the fundamental parts of robotics. Video processing algorithms such as velocity calculation of moving objects could be implemented with CUDA as well in the future iterations, if possible. Our implementations did not include any global operations on images, f possible to be used with CUDA the set of global operations can result in significant speed-ups as there are a lot of expensive global image processing operations. 





8. References
	Ahn, S.H. (2018). “Convolution”. Retrieved from: http://www.songho.ca/dsp/convolution/convolution.html#convolution_2d 
	Fig retrieved from https://www.mathworks.com/help/images/ref/illumgray.html
	Fig retrieved from https://medium.com/@bdhuma/6-basic-things-to-know-about-convolution-daef5e1bc411



