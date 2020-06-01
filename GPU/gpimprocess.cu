#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#define TILE_W 16
#define TILE_H 16
#define R 1
#define D ((R*2)+1)
#define S (D*D)
#define BLOCK_W TILE_W
#define BLOCK_H TILE_H


//For debug purposes
void printImageMatrix(float** image, int height, int width)
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


float* flattenImage(float** image, int height, int width)
{
   float* image_1D = new float[height*width];
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


void fileToMatrix(float** &image, char* fileName, int* height, int* width)
{
   std::ifstream imageFile(fileName);

   if(imageFile.is_open())
      {
	std::string line;
	getline(imageFile, line);
	std::istringstream iss(line);
	iss >> *(height) >> *(width);
	std::cout << "Height: " << *height << " ||-|| Width: " << *width << std::endl;

	image = new float*[*(height)];

	for(int i = 0; i < *(height); i++)
	   {
              image[i] = new float[*(width)];
           }	  

      	int h = 0;
      	float val;
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
        outFile << ("%.2f", image[x + y*width]) << " ";
    }

  outFile.close();
}


void getMinAndMax(float** image, int height, int width, int* min, int* max)
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

double getAverage(float** image, int height, int width)
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


__global__ void linearScale(float* image, int* a, int* width, float* b)
{
   float u = image[blockIdx.x];
   image[blockIdx.x] = (*b) * (u + (*a));
}


__global__ void grayWorld(float* image, double* scalingValue)
{
	float u = image[blockIdx.x];
	image[blockIdx.x] = u * *scalingValue;
}

__global__ void reflection(float* image, int* width)
{
	//blockIdx.x = i * *width + j
	int i = blockIdx.x / *width;
	int j = blockIdx.x - i * *width;  

	int reflectedIndex = i * *width + (*width - j - 1);
	float u = image[reflectedIndex];
	image[blockIdx.x] = u;
}

__global__ void orderedDithering(float* image, int* width)
{
	//blockIdx.x = i * *width + j
	int i = blockIdx.x / *width;
	int j = blockIdx.x - i * *width; 

	float u = image[blockIdx.x];

	if(i%2 == 0)
	{
		if(j%2 == 0)
		{
			if(u > 192)
				image[blockIdx.x] = 256;
			else
				image[blockIdx.x] = 0;
		}
		else
		{
			if(u > 64)
				image[blockIdx.x] = 256;
			else
				image[blockIdx.x] = 0;
		}
	}
	else
	{
		if(j%2 == 0)
			image[blockIdx.x] = 256;

		else
		{
			if(u > 128)
				image[blockIdx.x] = 256;
			else
				image[blockIdx.x] = 0;
		}
	}

}

__global__ void rotate90(float* result, float* image, int* height, int* width)
{
	//blockIdx.x = i * *width + j
	int i = blockIdx.x / *width;
	int j = blockIdx.x - i * *width; 

	int rotatedIndex = (*width - j - 1) * *height + i;
	float u = image[blockIdx.x];
	result[rotatedIndex] = u;
}

__global__ void rotate180(float* result, float* image, int* height, int* width)
{
	//blockIdx.x = i * *width + j
	int i = blockIdx.x / *width;
	int j = blockIdx.x - i * *width; 

	int rotatedIndex = (*height - i - 1) * *width + (*width - j - 1);
	float u = image[rotatedIndex];
	result[blockIdx.x] = u;
}


__global__ void medianFilter(float* image, float* outImage, int* height, int* width)
{
   int x = blockIdx.x * TILE_W + threadIdx.x;
   int y = blockIdx.y * TILE_H + threadIdx.y;

   unsigned int index = y * (*width) + x;
   unsigned int block_index = (threadIdx.y * blockDim.y + threadIdx.x);
   
   __shared__ float sharedWindow[BLOCK_W * BLOCK_H];
   sharedWindow[block_index] = image[index];

   __syncthreads();

  int i, key, j;
  for (i = 1; i < block_index; i++)
     {
       key = sharedWindow[i];
       j = i - 1;

       while (j >= 0 && sharedWindow[j] > key)
         {
           sharedWindow[j + 1] = sharedWindow[j];
           j = j - 1;
         }
       sharedWindow[j + 1] = key;
     }

   outImage[index] = sharedWindow[block_index / 2];
}

int main(int argc, char** argv)
{
   char* inFileName = argv[1];
   char* outFileName1 = argv[2];
   char* outFileName2 = argv[3];
   char* outFileName3 = argv[4];
   char* outFileName4 = argv[5];
   char* outFileName5 = argv[6];
   char* outFileName6 = argv[7];
   int writeOut = atoi(argv[8]);


   //IMAGE INIT
   int height, width, max = -1, min = 256;
   float** hostImage;

   fileToMatrix(hostImage, inFileName, &height, &width);
   getMinAndMax(hostImage, height, width, &min, &max);
   float* hostFlattened = flattenImage(hostImage, height, width);  

   //GPU INIT
   const size_t image_size = sizeof(float) * size_t(height*width);

   float* hostResult = new float[height*width];
   float* deviceImage;
   int *height_d, *width_d;

   cudaMalloc((void**)&height_d, sizeof(int));
   cudaMalloc((void**)&width_d, sizeof(int));   

   cudaMemcpy(height_d, &height, sizeof(int), cudaMemcpyHostToDevice);         
   cudaMemcpy(width_d, &width, sizeof(int), cudaMemcpyHostToDevice);         


   //LINSCALE
   
   cudaEvent_t start, stop;
   float time;
   
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   
   int BLOCK_C = height*width;
   int a = -1 * min, *d_a;

   float gmax = 255.0;
   float b = gmax / (max - min), *d_b;

   cudaMalloc((void**)&d_a, sizeof(int));
   cudaMalloc((void**)&d_b, sizeof(float));

   cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(d_b, &b, sizeof(float), cudaMemcpyHostToDevice);

   cudaMalloc((void**)&deviceImage, image_size);
   cudaMemcpy(deviceImage, hostFlattened, image_size, cudaMemcpyHostToDevice);	

   cudaEventRecord(start, 0);

   linearScale<<<BLOCK_C, 1>>>(deviceImage, d_a, width_d, d_b);
   
   cudaEventRecord(stop, 0);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&time, start, stop);
   
   cudaMemcpy(hostResult, deviceImage, image_size, cudaMemcpyDeviceToHost);
   
   std::cout << "Linear Scaling time: " << time/1000 << "sec" << std::endl;
   
     if(writeOut)
      matrixToFile(outFileName1, hostResult, height, width);
      
   //GRAYWORLD
   double average = getAverage(hostImage, height, width), *d_average;


   cudaMalloc((void**)&d_average, sizeof(double));
   cudaMemcpy(d_average, &average, sizeof(double), cudaMemcpyHostToDevice);

   cudaMalloc((void**)&deviceImage, image_size);
   cudaMemcpy(deviceImage, hostFlattened, image_size, cudaMemcpyHostToDevice);	

   cudaEventRecord(start, 0);
  
   grayWorld<<<BLOCK_C, 1>>>(deviceImage, d_average);
   
   cudaEventRecord(stop, 0);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&time, start, stop);
   
   cudaMemcpy(hostResult, deviceImage, image_size, cudaMemcpyDeviceToHost);

    std::cout << "GrayWorld time: " << time/1000 << "sec" << std::endl;

   if(writeOut)
      matrixToFile(outFileName2, hostResult, height, width);
      
  //REFLECTION
  
   cudaMalloc((void**)&width_d, sizeof(int));          
   cudaMemcpy(width_d, &width, sizeof(int), cudaMemcpyHostToDevice);  

   cudaMalloc((void**)&deviceImage, image_size);
   cudaMemcpy(deviceImage, hostFlattened, image_size, cudaMemcpyHostToDevice);	

   cudaEventRecord(start, 0);

   reflection<<<BLOCK_C, 1>>>(deviceImage, width_d);
   
   cudaEventRecord(stop, 0);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&time, start, stop);
   
   cudaMemcpy(hostResult, deviceImage, image_size, cudaMemcpyDeviceToHost);

   std::cout << "Reflection time: " << time/1000 << "sec" << std::endl;

   if(writeOut)
      matrixToFile(outFileName3, hostResult, height, width);

   //ORDERED DITHERING
  
   cudaMalloc((void**)&width_d, sizeof(int));          
   cudaMemcpy(width_d, &width, sizeof(int), cudaMemcpyHostToDevice);  

   cudaMalloc((void**)&deviceImage, image_size);
   cudaMemcpy(deviceImage, hostFlattened, image_size, cudaMemcpyHostToDevice);	

   cudaEventRecord(start, 0);

   orderedDithering<<<BLOCK_C, 1>>>(deviceImage, width_d);
   
   cudaEventRecord(stop, 0);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&time, start, stop);
   
   cudaMemcpy(hostResult, deviceImage, image_size, cudaMemcpyDeviceToHost);
     
   std::cout << "Ordered Dithering time: " << time/1000 << "sec" << std::endl;
     
   if(writeOut)
      matrixToFile(outFileName4, hostResult, height, width);

   //ROTATE90
   float* result90;
   
   cudaMalloc((void**)&height_d, sizeof(int));
   cudaMalloc((void**)&width_d, sizeof(int));   

   cudaMemcpy(height_d, &height, sizeof(int), cudaMemcpyHostToDevice);         
   cudaMemcpy(width_d, &width, sizeof(int), cudaMemcpyHostToDevice);          

   cudaMalloc((void**)&deviceImage, image_size);
   cudaMemcpy(deviceImage, hostFlattened, image_size, cudaMemcpyHostToDevice);	
  
   cudaMalloc((void**)&result90, image_size);
   cudaMemcpy(result90, hostFlattened, image_size, cudaMemcpyHostToDevice);	

   cudaEventRecord(start, 0);

   rotate90<<<BLOCK_C, 1>>>(result90, deviceImage, height_d, width_d);
   
   cudaEventRecord(stop, 0);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&time, start, stop);
   
   cudaMemcpy(hostResult, result90, image_size, cudaMemcpyDeviceToHost);

   std::cout << "Rotation 90 time: " << time/1000 << "sec" << std::endl;

   if(writeOut)
      matrixToFile(outFileName5, hostResult, width, height);
      
   //ROTATE180
   float* result180;
   
   cudaMalloc((void**)&height_d, sizeof(int));
   cudaMalloc((void**)&width_d, sizeof(int));   

   cudaMemcpy(height_d, &height, sizeof(int), cudaMemcpyHostToDevice);         
   cudaMemcpy(width_d, &width, sizeof(int), cudaMemcpyHostToDevice);          

   cudaMalloc((void**)&deviceImage, image_size);
   cudaMemcpy(deviceImage, hostFlattened, image_size, cudaMemcpyHostToDevice);	
  
   cudaMalloc((void**)&result180, image_size);
   cudaMemcpy(result180, hostFlattened, image_size, cudaMemcpyHostToDevice);	

   cudaEventRecord(start, 0);

   rotate180<<<BLOCK_C, 1>>>(result180, deviceImage, height_d, width_d);
   
   cudaEventRecord(stop, 0);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&time, start, stop);
   
   cudaMemcpy(hostResult, result180, image_size, cudaMemcpyDeviceToHost);
   
   std::cout << "Rotation 180 time: " << time/1000 << "sec" << std::endl;

   if(writeOut)
      matrixToFile(outFileName6, hostResult, height, width);
      
   /*MEDFILTER
   dim3 grid(TILE_W, TILE_H);
   dim3 block(BLOCK_W, BLOCK_H);

   float* outImage;
   cudaMalloc((void**)&outImage, image_size);	

   cudaMalloc((void**)&deviceImage, image_size);
   cudaMemcpy(deviceImage, hostFlattened, image_size, cudaMemcpyHostToDevice);

   medianFilter<<<grid, block>>>(deviceImage, outImage, height_d, width_d);
   cudaMemcpy(hostResult, outImage, image_size, cudaMemcpyDeviceToHost);

   if(writeOut)
      matrixToFile(outFileName, hostResult, height, width);*/

   return 0;
}
