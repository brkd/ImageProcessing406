#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>


#define TILE_W 32
#define TILE_H 32
#define R 1
#define D ((R*2)+1)
#define S (D*D)


//For debug purposes
void printImageMatrix(unsigned char** image, int height, int width)
{
  for(int i = 0; i < height; i++)
    {
      for(int j = 0; j < width; j++)
        {
          std::cout << image[i][j] << " ";
        }
      std::cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << std::endl;
    }
}


unsigned char* flattenImage(unsigned char** image, int height, int width)
{
   unsigned char* image_1D = new unsigned char[height*width];
   for(int i = 0; i < height; i++)
      {
         for(int j = 0; j < width; j++)
	    {
	       image_1D[j + i*width] = image[i][j];
	    }
      }
   std::cout << "Image flattened" << std::endl;
   return image_1D;
}


void fileToMatrix(unsigned char** &image, char* fileName, int* height, int* width)
{
   std::ifstream imageFile(fileName);

   if(imageFile.is_open())
      {
	std::string line;
	getline(imageFile, line);
	std::istringstream iss(line);
	iss >> *(height) >> *(width);
	std::cout << "Height: " << *height << " ||-|| Width: " << *width << std::endl;

	image = new unsigned char*[*(height)];

	for(int i = 0; i < *(height); i++)
	   {
              image[i] = new unsigned char[*(width)];
           }	  

      	int h = 0;
      	int val;
        while(getline(imageFile, line))
           {
              int w = 0;
              std::istringstream iss(line);
              while(iss >> val)
                 {
                    image[h][w++] = val;
                 }
              h++;
           }
        std::cout << "Image saved to matrix..." << std::endl;
      }

    imageFile.close();      
}

void matrixToFile(char* fileName, unsigned char* image, int height, int width)
{
  std::ofstream outFile;
  outFile.open(fileName);
  outFile << height << " " << width << '\n';
  for(int i = 0; i < height*width; i++)
    {
	int x = i % width;
	int y = i / width;
	if(i != 0 && x == 0)
	   outFile << '\n';
        outFile << int(image[x + y*width]) << " ";
    }

  outFile.close();
}


void getMinAndMax(unsigned char** image, int height, int width, int* min, int* max)
{
   for(int i = 0; i < height; i++)
      {
         for(int j = 0; j < width; j++)
	    {
	       if(image[i][j] > *(max))
	          {
		     *max = image[i][j];
	          }
	       if(image[i][j] < *(min))
	          {
		     *min = image[i][j];
		  }
	    }
      }

   std::cout << "Min: " << *min << " ||-|| Max: " << *max << std::endl;
}



__global__ void linearScale(float* image, int* a, int* width, float* b)
{
   float u = image[blockIdx.x];
   image[blockIdx.x] = (*b) * (u + (*a));
}


__global__ void medianFilter(unsigned char* image, unsigned char* outImage, int* height, int* width)
{
   int im_y = blockIdx.y * blockDim.y + threadIdx.y;
   int im_x = blockIdx.x * blockDim.x + threadIdx.x; 	
	
   if(im_y == 0 || im_y == (*height) - 1 || im_x == 0 || im_x == (*width) - 1)
   {
      outImage[im_y * (*width) + im_x] = 0;
      return;
   }


   int index = im_y * (*width) + im_x;//(((blockIdx.x) + 1) * (*width)) + threadIdx.x + 1;
   if(index >= (*height) * (*width))
      return;
   int image_index;
   
   __shared__ unsigned char window[S];
   int window_index = 0;

   for(int i = -1; i <= 1; i++)
   {
      for(int j = -1; j <= 1; j++)
      {
         image_index = index + (i * (*width)) + j;
	 window[window_index++] = image[image_index];
      }
   }

   int key, i ,j;

   for(i = 1; i < S; i++)
   {
      key = window[i];
      j = i - 1;

      while(j >= 0 && window[j] > key)
      {
         window[j + 1] = window[j];
	 j = j - 1;
      }

      window[j + 1] = key;
   }

   outImage[index] = window[S/2];
}


int main(int argc, char** argv)
{
   char* inFileName = argv[1];
   char* outFileName = argv[2];
   int writeOut = atoi(argv[3]);
   std::cout << "BEFORE CUDA EXECUTION" << std::endl;
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);

   std::cout << "STARTED EXECUTION" << std::endl;

   //IMAGE INIT
   int height, width, max = -1, min = 256;
   unsigned char** hostImage;

   fileToMatrix(hostImage, inFileName, &height, &width);
   getMinAndMax(hostImage, height, width, &min, &max);
   unsigned char* hostFlattened = flattenImage(hostImage, height, width);  
   //printImageMatrix(hostImage, height, width);
   //GPU INIT
   const int image_size = sizeof(unsigned char) * (height*width);

   unsigned char* hostResult = new unsigned char[height*width];
   unsigned char* deviceImage;
   int *height_d, *width_d;

   cudaMalloc(&height_d, sizeof(int));
   cudaMalloc(&width_d, sizeof(int));   

   cudaMemcpy(height_d, &height, sizeof(int), cudaMemcpyHostToDevice);         
   cudaMemcpy(width_d, &width, sizeof(int), cudaMemcpyHostToDevice);         


   //LINSCALE
   int BLOCK_C = height*width;
   int a = -1 * min, *d_a;

   float gmax = 255.0;
   float b = gmax / (max - min), *d_b;

   //cudaMalloc(&d_a, sizeof(int));
   //cudaMalloc(&d_b, sizeof(float));

   //cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
   //cudaMemcpy(d_b, &b, sizeof(float), cudaMemcpyHostToDevice);

   //cudaMalloc(&deviceImage, image_size);
   //cudaMemcpy(deviceImage, hostFlattened, image_size, cudaMemcpyHostToDevice);	

   //linearScale<<<BLOCK_C, 1>>>(deviceImage, d_a, width_d, d_b);
   //cudaMemcpy(hostResult, deviceImage, image_size, cudaMemcpyDeviceToHost);

   //if(writeOut)
     // matrixToFile(outFileName, hostResult, height, width);
   

   //MEDFILTER

   unsigned char* outImage;
   cudaMalloc(&outImage, image_size);	

   cudaMalloc(&deviceImage, image_size);
   cudaMemcpy(deviceImage, hostFlattened, image_size, cudaMemcpyHostToDevice);


   int BLOCK_H = (int)std::ceil((float)width / TILE_H);
   int BLOCK_W = (int)std::ceil((float)height / TILE_W);
   std::cout << BLOCK_H << " " << BLOCK_W << std::endl;

   dim3 deviceBlocks(BLOCK_H, BLOCK_W);
   


   dim3 deviceThreads(32, 32);

   cudaEventRecord(start);
   medianFilter<<<deviceBlocks, deviceThreads>>>(deviceImage, outImage, height_d, width_d);
   cudaEventRecord(stop);

   cudaEventSynchronize(stop);
   cudaGetLastError();
   
   float elapsed = 0;
   cudaEventElapsedTime(&elapsed, start, stop);
   
   std::cout << "MEDIAN FILTER CALCULATION: " << elapsed / 1000<< std::endl;
   cudaMemcpy(hostResult, outImage, image_size, cudaMemcpyDeviceToHost);

   if(writeOut)
      matrixToFile(outFileName, hostResult, height, width);

   return 0;
}