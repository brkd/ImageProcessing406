#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>


#define TILE_W 32 //Decrease to increment block count and decrement thread count
#define TILE_H 32 //Decrease to increment block count and decrement thread count
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


unsigned char* char_flattenImage(unsigned char** image, int height, int width)
{
   unsigned char* image_1D = new unsigned char[height*width];
   for(int i = 0; i < height; i++)
      {
         for(int j = 0; j < width; j++)
	    {
	       image_1D[j + i*width] = image[i][j];
	    }
      }
   std::cout << "Char image flattened" << std::endl;
   return image_1D;
}

float* flattenImage(unsigned char** image, int height, int width)
{
   float* image_1D = new float[height*width];
   for(int i = 0; i < height; i++)
      {
         for(int j = 0; j < width; j++)
	    {
	       image_1D[j + i*width] = image[i][j];
	    }
      }
   std::cout << "Float image flattened" << std::endl;
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

void matrixToFile(char* fileName, float* image, int height, int width)
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

double getAverage(unsigned char** image, int height, int width)
{
	float sum = 0;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			sum += image[i][j];
		}
	}

	return sum / (width * height);
}


__global__ void linearScale(float* image, float* outImage, int* a, int* width, int* height, float* b)
{
   int im_y = blockIdx.y * blockDim.y + threadIdx.y;
   int im_x = blockIdx.x * blockDim.x + threadIdx.x; 	

   int index = im_y * (*width) + im_x;

   if(index >= (*height) * (*width))
      return;

   float u = image[index];
   outImage[index] = (*b) * (u + (*a));
}


__global__ void grayWorld(float* image, float* outImage, double* scalingValue, int* height, int* width)
{
   int im_y = blockIdx.y * blockDim.y + threadIdx.y;
   int im_x = blockIdx.x * blockDim.x + threadIdx.x; 	

   int index = im_y * (*width) + im_x;

   if(index >= (*height) * (*width))
      return;

   float u = image[index];
   outImage[index] = u * *scalingValue;
}

__global__ void reflection(float* image, float* outImage, int* height, int* width)
{
   int im_y = blockIdx.y * blockDim.y + threadIdx.y;
   int im_x = blockIdx.x * blockDim.x + threadIdx.x; 	

   int index = im_y * (*width) + im_x;

   if(index >= (*height) * (*width))
      return;

   int i = index / *width;
   int j = index - i * *width;

   int reflectedIndex = i * *width + (*width - j - 1);
   float u = image[reflectedIndex];
   outImage[index] = u;
}

__global__ void orderedDithering(float* image, float* outImage, int* height, int* width)
{
   int im_y = blockIdx.y * blockDim.y + threadIdx.y;
   int im_x = blockIdx.x * blockDim.x + threadIdx.x; 	

   int index = im_y * (*width) + im_x;

   if(index >= (*height) * (*width))
      return;
	
   int i = index / *width;
   int j = index - i * *width; 

   float u = image[index];

   if(i%2 == 0)
   {
   	if(j%2 == 0)
	{
 		if(u > 192)
			outImage[index] = 255;
		else
			outImage[index] = 0;
	}
	else
	{
		if(u > 64)
			outImage[index] = 255;
		else
			outImage[index] = 0;
	}
   }
   else
   {
	if(j%2 == 0)
		outImage[index] = 255;
	else
	{
		if(u > 128)
			outImage[index] = 255;
		else
			outImage[index] = 0;
	}
   }

}

__global__ void rotate90(float* result, float* image, int* height, int* width)
{
   int im_y = blockIdx.y * blockDim.y + threadIdx.y;
   int im_x = blockIdx.x * blockDim.x + threadIdx.x; 	

   int index = im_y * (*width) + im_x;

   if(index >= (*height) * (*width))
      return;

   int i = index / *width;
   int j = index - i * *width; 

   int rotatedIndex = (*width - j - 1) * *height + i;
   float u = image[index];
   result[rotatedIndex] = u;
}

__global__ void rotate180(float* result, float* image, int* height, int* width)
{
   int im_y = blockIdx.y * blockDim.y + threadIdx.y;
   int im_x = blockIdx.x * blockDim.x + threadIdx.x; 	

   int index = im_y * (*width) + im_x;

   if(index >= (*height) * (*width))
      return;

   int i = index / *width;
   int j = index - i * *width; 

   int rotatedIndex = (*height - i - 1) * *width + (*width - j - 1);
   float u = image[rotatedIndex];
   result[index] = u;
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

   int index = im_y * (*width) + im_x;

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
   char* inFileName = argv[1], *outFileName;
   int writeOut = atoi(argv[2]);

   //IMAGE INIT
   int height, width, max = -1, min = 256;
   unsigned char** hostImage;

   fileToMatrix(hostImage, inFileName, &height, &width);
   getMinAndMax(hostImage, height, width, &min, &max);
   float* hostFlattened = flattenImage(hostImage, height, width);  
   unsigned char* char_hostFlattened = char_flattenImage(hostImage, height, width);


   //GPU INIT
   const int image_size = sizeof(float) * (height*width);

   const int charImage_size = sizeof(unsigned char) * (height*width);
   unsigned char* char_deviceImage;

   float* deviceImage;
   float* hostResult = new float[height*width];

   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);

   int BLOCK_H = (int)std::ceil((float)width / TILE_H);
   int BLOCK_W = (int)std::ceil((float)height / TILE_W);

   dim3 deviceBlocks(BLOCK_H, BLOCK_W);
   dim3 deviceThreads(TILE_H, TILE_W);

   int *height_d, *width_d;
   float time = 0;

   cudaMalloc(&height_d, sizeof(int));
   cudaMalloc(&width_d, sizeof(int));   

   cudaMemcpy(height_d, &height, sizeof(int), cudaMemcpyHostToDevice);         
   cudaMemcpy(width_d, &width, sizeof(int), cudaMemcpyHostToDevice);         

   cudaMalloc(&deviceImage, image_size);
   cudaMemcpy(deviceImage, hostFlattened, image_size, cudaMemcpyHostToDevice);


   //*************************LINSCALE******************************//
   
   int a = -1 * min, *d_a;

   float gmax = 255.0;
   float b = gmax / (max - min), *d_b;

   float* linearScaleResult;

   cudaMalloc(&linearScaleResult, image_size);  
   cudaMalloc(&d_a, sizeof(int));
   cudaMalloc(&d_b, sizeof(float));

   cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(d_b, &b, sizeof(float), cudaMemcpyHostToDevice);

   cudaEventRecord(start, 0);

   linearScale<<<deviceBlocks, deviceThreads>>>(deviceImage, linearScaleResult, d_a, width_d, height_d, d_b);

   cudaEventRecord(stop, 0);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&time, start, stop);

   std::cout << "LinScale time: " << time/1000 << "sec" << std::endl;

   cudaMemcpy(hostResult, linearScaleResult, image_size, cudaMemcpyDeviceToHost);

   if(writeOut)
   {
      outFileName = "CUDA_linear_scale_.txt";
      matrixToFile(outFileName, hostResult, height, width);
   }

   cudaFree(d_a);
   cudaFree(d_b);
   cudaFree(linearScaleResult);

   //*************************LINSCALE******************************//


   //*************************GRAYWORLD******************************// 

   double average = getAverage(hostImage, height, width);
   double scalingValue = 127.5 / average, *d_scale;;
   float* grayWorldResult;

   cudaMalloc(&grayWorldResult, image_size);

   cudaMalloc((void**)&d_scale, sizeof(double));
   cudaMemcpy(d_scale, &scalingValue, sizeof(double), cudaMemcpyHostToDevice);

   cudaEventRecord(start, 0);
  
   grayWorld<<<deviceBlocks, deviceThreads>>>(deviceImage, grayWorldResult, d_scale, height_d, width_d);
   
   cudaEventRecord(stop, 0);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&time, start, stop);
   
   cudaMemcpy(hostResult, grayWorldResult, image_size, cudaMemcpyDeviceToHost);

   std::cout << "GrayWorld time: " << time/1000 << "sec" << std::endl;

   if(writeOut)
   {
      outFileName = "CUDA_gray_world_.txt";
      matrixToFile(outFileName, hostResult, height, width);
   }

   cudaFree(grayWorldResult);
   cudaFree(d_scale);

   //*************************GRAYWORLD******************************// 
      
  //REFLECTION
  
   float* reflectionResult;
   cudaMalloc(&reflectionResult, image_size);

   cudaEventRecord(start, 0);

   reflection<<<deviceBlocks, deviceThreads>>>(deviceImage, reflectionResult, height_d, width_d);
   
   cudaEventRecord(stop, 0);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&time, start, stop);
   
   cudaMemcpy(hostResult, reflectionResult, image_size, cudaMemcpyDeviceToHost);

   std::cout << "Reflection time: " << time/1000 << "sec" << std::endl;

   if(writeOut)
   {
      outFileName = "CUDA_reflection_.txt";
      matrixToFile(outFileName, hostResult, height, width);
   }

   cudaFree(reflectionResult);

   //ORDERED DITHERING
  
   float* orderedDitheringResult;
   cudaMalloc(&orderedDitheringResult, image_size);

   cudaEventRecord(start, 0);

   orderedDithering<<<deviceBlocks, deviceThreads>>>(deviceImage, orderedDitheringResult, height_d, width_d);
   
   cudaEventRecord(stop, 0);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&time, start, stop);
   
   cudaMemcpy(hostResult, orderedDitheringResult, image_size, cudaMemcpyDeviceToHost);
     
   std::cout << "Ordered Dithering time: " << time/1000 << "sec" << std::endl;
    
   if(writeOut)
   {
      outFileName = "CUDA_dithering_.txt";
      matrixToFile(outFileName, hostResult, height, width);
   }
 
   cudaFree(orderedDitheringResult);

   //ROTATE90
   float* result90;
   
   cudaMalloc((void**)&result90, image_size);

   cudaEventRecord(start, 0);

   rotate90<<<deviceBlocks, deviceThreads>>>(result90, deviceImage, height_d, width_d);
   
   cudaEventRecord(stop, 0);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&time, start, stop);
   
   cudaMemcpy(hostResult, result90, image_size, cudaMemcpyDeviceToHost);

   std::cout << "Rotation 90 time: " << time/1000 << "sec" << std::endl;

   if(writeOut)
   {
      outFileName = "CUDA_rotate_90_.txt";
      matrixToFile(outFileName, hostResult, width, height);
   }

   cudaFree(result90);
      
   //ROTATE180
   float* result180;
   
   cudaMalloc((void**)&result180, image_size);

   cudaEventRecord(start, 0);

   rotate180<<<deviceBlocks, deviceThreads>>>(result180, deviceImage, height_d, width_d);
   
   cudaEventRecord(stop, 0);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&time, start, stop);
   
   cudaMemcpy(hostResult, result180, image_size, cudaMemcpyDeviceToHost);
   
   std::cout << "Rotation 180 time: " << time/1000 << "sec" << std::endl;

   if(writeOut)
   {
      outFileName = "CUDA_rotate_180_.txt";
      matrixToFile(outFileName, hostResult, height, width);
   }

   cudaFree(result180);
   cudaFree(deviceImage);
      
   //MEDFILTER


   unsigned char* outImage;
   unsigned char* charResult = new unsigned char[height*width];

   cudaMalloc(&outImage, charImage_size);	

   cudaMalloc(&char_deviceImage, charImage_size);
   cudaMemcpy(char_deviceImage, char_hostFlattened, charImage_size, cudaMemcpyHostToDevice);

   cudaEventRecord(start);
   medianFilter<<<deviceBlocks, deviceThreads>>>(char_deviceImage, outImage, height_d, width_d);
   cudaEventRecord(stop);

   cudaEventSynchronize(stop);
   cudaGetLastError();
   
   float elapsed = 0;
   cudaEventElapsedTime(&elapsed, start, stop);
   
   std::cout << "Median Filter Time: " << elapsed / 1000 << std::endl;
   cudaMemcpy(charResult, outImage, charImage_size, cudaMemcpyDeviceToHost);

   if(writeOut)
   {
      outFileName = "CUDA_median_filter_.txt";
      matrixToFile(outFileName, charResult, height, width);
   }

   cudaFree(char_deviceImage);
   cudaFree(outImage);
   cudaFree(height_d);
   cudaFree(width_d);

   return 0;
}
