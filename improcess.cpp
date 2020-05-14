#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <math.h>
#include <omp.h>

#define PI 3.141527f


//For reading the pixel values from given file name
void fileToMatrix(float** &image, std::string fileName, int& height, int& width)
{
  std::ifstream imageFile(fileName);    
  
  if(imageFile.is_open())
    {
      std::string line;
      getline(imageFile, line);
      std::istringstream iss(line);
      iss >> height >> width;
      std::cout << "Height: " << height << " ||-|| Width: " << width << std::endl;
      
      image = new float*[height];

      for(int i = 0; i < height; i++)
	{
	  image[i] = new float[width];
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
void findMinAndMax(float** image, int height, int width, int& max, int& min)
{
  for(int i = 0; i < height; i++)
    {
      for(int j = 0; j < width; j++)
	{
	  if(max < image[i][j])
	    max = image[i][j];

	  if(min > image[i][j])
	    min = image[i][j];
	}
    }
}

float** orderedDithering(float** image, int height, int width)
{
	float** ditheredImage = new float*[height];

	for (int i = 0; i < height - 2; i += 2)
	{
		for (int j = 0; j < width - 2; j += 2)
		{
			if (image[i][j] > 192)
				ditheredImage[i][j] = 256;
			else
				ditheredImage[i][j] = 0;

			if (image[i][j+1] > 64)
				ditheredImage[i][j+1] = 256;
			else
				ditheredImage[i][j+1] = 0;

			if (image[i+1][j+1] > 128)
				ditheredImage[i+1][j+1] = 256;
			else
				ditheredImage[i+1][j+1] = 0;

			ditheredImage[i+1][j] = 256;
		}
	}
	

	

	return ditheredImage;
}

float** par_orderedDithering(float** image, int height, int width)
{
	float** ditheredImage = new float*[height];

#pragma omp parallel for 
	for (int i = 0; i < height - 2; i += 2)
	{
#pragma omp parallel for 
		for (int j = 0; j < width - 2; j += 2)
		{
			if (image[i][j] > 192)
				ditheredImage[i][j] = 256;
			else
				ditheredImage[i][j] = 0;

			if (image[i][j + 1] > 64)
				ditheredImage[i][j + 1] = 256;
			else
				ditheredImage[i][j + 1] = 0;

			if (image[i + 1][j + 1] > 128)
				ditheredImage[i + 1][j + 1] = 256;
			else
				ditheredImage[i + 1][j + 1] = 0;

			ditheredImage[i + 1][j] = 256;
		}
	}

	return ditheredImage;
}


float** grayWorld(float** image, int height, int width, double average)
{
	float** scaledImage = new float*[height];

	double scalingValue = 127.5 / average;

	for (int i = 0; i < height; i++)
	{
		scaledImage[i] = new float[width];
	}


	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			scaledImage[i][j] = image[i][j] * scalingValue;
		}
	}

	return scaledImage;
}


float** par_grayWorld(float** image, int height, int width, double average)
{
	float** scaledImage = new float*[height];

	double scalingValue = 127.5 / average;

	for (int i = 0; i < height; i++)
	{
		scaledImage[i] = new float[width];
	}

#pragma omp parallel for 
	for (int i = 0; i < height; i++)
	{
#pragma omp parallel for 
		for (int j = 0; j < width; j++)
		{
			scaledImage[i][j] = image[i][j] * scalingValue;
		}
	}

	return scaledImage;
}



float** reflection(float** image, int height, int width)
{
	float** reflectedImage(image);

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			reflectedImage[i][j] = image[i][width - j - 1];
		}
	}

	return reflectedImage;
}

float** par_reflection(float** image, int height, int width)
{
	float** reflectedImage(image);

#pragma omp parallel for 
	for (int i = 0; i < height; i++)
	{
#pragma omp parallel for 
		for (int j = 0; j < width; j++)
		{
			reflectedImage[i][j] = image[i][width - j - 1];
		}
	}

	return reflectedImage;
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


// g(u) = b*(u + a)
float** linearScale(float** image, int height, int width, int max, int min)
{
  float** scaledImage = new float*[height];
  for(int i = 0; i < height; i++)
    {
      scaledImage[i] = new float[width];
    }

  int a = -1*min;
  float gmax = 255.0;
  float b;
  float u;

  for(int i = 0; i < height; i++)
    {
      for(int j = 0; j < width; j++)
	{
	  u = image[i][j];
	  b = gmax/(max - min);
	  
	  scaledImage[i][j] = b*(u + a);
	}
    }

  return scaledImage;
}

float** par_linearScale(float** image, int height, int width, int max, int min)
{
  float** scaledImage = new float*[height];
  for(int i = 0; i < height; i++)
    {
      scaledImage[i] = new float[width];
    }

  int a = -1*min;
  float gmax = 255.0;
  float b;
  float u;

#pragma omp parallel for collapse(2) private(u, b)
  for(int i = 0; i < height; i++)
    {
      for(int j = 0; j < width; j++)
	{
	  u = image[i][j];
	  b = gmax/(max - min);
	  
	  scaledImage[i][j] = b*(u + a);
	}
    }

  return scaledImage;
}

void insertionSort(float* window, int size)
{
  int i, key, j;  
  for (i = 1; i < size; i++) 
    {  
      key = window[i];  
      j = i - 1;  
  
      while (j >= 0 && window[j] > key) 
        {  
	  window[j + 1] = window[j];  
	  j = j - 1;  
        }  
      window[j + 1] = key;  
    }  
}

float getMedian(float* window, int size)
{
  insertionSort(window, size);
  
  return window[(size / 2)];
}

float** par_sobelFilterX(float** image, int height, int width)
{
  float** gradientX = new float*[height];
  
  for(int i = 0; i < height; i++)
    {
      gradientX[i] = new float[width];
    }
  
  float x_filter[3][3];

  x_filter[0][0] = -1;  x_filter[0][1] = 0; x_filter[0][2] = 1;
  x_filter[1][0] = -2;  x_filter[1][1] = 0; x_filter[1][2] = 2; 
  x_filter[2][0] = -1;  x_filter[2][1] = 0; x_filter[2][2] = 1; 

  float local_result = 0.0;  
#pragma omp parallel for collapse(2) firstprivate(local_result)
  for(int i = 1; i < height - 1; i++)
    {
      for(int j = 1; j < width - 1; j++)
	{
	  for(int a = 0; a < 3; a++)
	    {
	      for(int b = 0; b < 3; b++)
		{
		  local_result = local_result + image[i - 1 + a][j - 1 + b] * x_filter[a][b];
		}
	    }	  
	  if(local_result < 0) local_result = 0;
	  if(local_result > 255) local_result = 255;
	  gradientX[i][j] = local_result;
	  local_result = 0.0;	  
	}
    }
  
  return gradientX;
}

float** sobelFilterX(float** image, int height, int width)
{
  float** gradientX = new float*[height];
  
  for(int i = 0; i < height; i++)
    {
      gradientX[i] = new float[width];
    }
  
  float x_filter[3][3];

  x_filter[0][0] = -1;  x_filter[0][1] = 0; x_filter[0][2] = 1;
  x_filter[1][0] = -2;  x_filter[1][1] = 0; x_filter[1][2] = 2; 
  x_filter[2][0] = -1;  x_filter[2][1] = 0; x_filter[2][2] = 1; 

  float local_result = 0.0;  
  for(int i = 1; i < height - 1; i++)
    {
      for(int j = 1; j < width - 1; j++)
	{
	  for(int a = 0; a < 3; a++)
	    {
	      for(int b = 0; b < 3; b++)
		{
		  local_result += image[i - 1 + a][j - 1 + b] * x_filter[a][b];
		}
	    }	  
	  if(local_result < 0) local_result = 0;
	  if(local_result > 255) local_result = 255;
	  gradientX[i][j] = local_result;
	  local_result = 0.0;	  
	}
    }
  
  return gradientX;
}

float** par_sobelFilterY(float** image, int height, int width)
{
  float** gradientY = new float*[height];

  for(int i = 0; i < height; i++)
    {
      gradientY[i] = new float[width];
    }

  float y_filter[3][3];

  y_filter[0][0] = -1;  y_filter[0][1] = -2; y_filter[0][2] = -1;
  y_filter[1][0] = 0;  y_filter[1][1] = 0; y_filter[1][2] = 0; 
  y_filter[2][0] = 1;  y_filter[2][1] = 2; y_filter[2][2] = 1;
  
  float local_result = 0.0;  
#pragma omp parallel for collapse(2) firstprivate(local_result)
  for(int i = 1; i < height - 1; i++)
    {
      for(int j = 1; j < width - 1; j++)
	{
	  for(int a = 0; a < 3; a++)
	    {
	      for(int b = 0; b < 3; b++)
		{
		  local_result += image[i - 1 + a][j - 1 + b] * y_filter[a][b];
		}
	    }	  
	  if(local_result < 0) local_result = 0;
	  if(local_result > 255) local_result = 255;
	  gradientY[i][j] = local_result;
	  local_result = 0.0;	  
	}
    }
  
  return gradientY;
} 

float** sobelFilterY(float** image, int height, int width)
{
  float** gradientY = new float*[height];

  for(int i = 0; i < height; i++)
    {
      gradientY[i] = new float[width];
    }

  float y_filter[3][3];

  y_filter[0][0] = -1;  y_filter[0][1] = -2; y_filter[0][2] = -1;
  y_filter[1][0] = 0;  y_filter[1][1] = 0; y_filter[1][2] = 0; 
  y_filter[2][0] = 1;  y_filter[2][1] = 2; y_filter[2][2] = 1;
  
  float local_result = 0.0;  
  //printImageMatrix(image, height, width);
  for(int i = 1; i < height - 1; i++)
    {
      for(int j = 1; j < width - 1; j++)
	{
	  for(int a = 0; a < 3; a++)
	    {
	      for(int b = 0; b < 3; b++)
		{
		  local_result += image[i - 1 + a][j - 1 + b] * y_filter[a][b];
		}
	    }	  
	  if(local_result < 0) local_result = 0;
	  if(local_result > 255) local_result = 255;
	  gradientY[i][j] = local_result;
	  local_result = 0.0;	  
	}
    }
  
  return gradientY;
}

float** par_sobelEdgeDetection(float** gradX, float** gradY, int height, int width, float T)
{
  float** sobelEdgeDetectedImage = new float*[height];
  
  for(int i = 0; i < height; i++)
    {
      sobelEdgeDetectedImage[i] = new float[width];
    }

  float local_sum = 0;
#pragma omp parallel for collapse(2) firstprivate(local_sum)
  for(int i = 0; i < height; i++)
    {
      for(int j = 0; j < width; j++)
	{
	  local_sum = (gradX[i][j] * gradX[i][j]) + (gradY[i][j] * gradY[i][j]);
	  local_sum = sqrt(local_sum);
	  
	  if(local_sum > T)
	    sobelEdgeDetectedImage[i][j] = 255;
	  else
	    sobelEdgeDetectedImage[i][j] = 0;
	}      
    }

  return sobelEdgeDetectedImage;
}

float** sobelEdgeDetection(float** gradX, float** gradY, int height, int width, float T)
{
  float** sobelEdgeDetectedImage = new float*[height];
  
  for(int i = 0; i < height; i++)
    {
      sobelEdgeDetectedImage[i] = new float[width];
    }

  float local_sum = 0;
  for(int i = 0; i < height; i++)
    {
      for(int j = 0; j < width; j++)
	{
	  local_sum = (gradX[i][j] * gradX[i][j]) + (gradY[i][j] * gradY[i][j]);
	  local_sum = sqrt(local_sum);
	  
	  if(local_sum > T)
	    sobelEdgeDetectedImage[i][j] = 255;
	  else
	    sobelEdgeDetectedImage[i][j] = 0;
	}      
    }

  return sobelEdgeDetectedImage;
}

float** imageWarp(float** image, int height, int width)
{
  float** warpedImage = new float*[height];
  
  for(int i = 0; i < height; i++)
    {
      warpedImage[i] = new float[width];
    }

  int x, y, x_cursor, y_cursor, center_x = width/2, center_y = height/2; 
  float radius, theta;
  float DRAD = 180.0f / PI;

  for(y_cursor = 0; y_cursor < height; y_cursor++)
    {
      for(x_cursor = 0; x_cursor < width; x_cursor++)
	{
	  radius = sqrtf((x_cursor - center_x) * (x_cursor - center_x) + (y_cursor - center_y) * (y_cursor - center_y));
	  theta = (radius / 2) * DRAD;
	  x = cos(theta) * (x_cursor - center_x) - sin(theta) * (y_cursor - center_y) + center_x;
	  y = sin(theta) * (x_cursor - center_x) + cos(theta) * (y_cursor - center_y) + center_y;

	  std::cout << x << " " << y << std::endl;
	  warpedImage[y_cursor][x_cursor] = image[y_cursor][x_cursor];
	}
    }

  return warpedImage;
} 

float** par_medianFilter(float** image, int height, int width, const int k)
{
  float** medianFilteredImage(image);
  
  const int window_size = k*k;
  float window[window_size];

  float median;
  int w_index = 0;
#pragma omp parallel for collapse(2) private(median, window, w_index)
    for(int row = k/2; row <= height - k/2 - 1; row++)
      {
	for(int col = k/2; col <= width - k/2 - 1; col++)
	  {
	    for(int i = -k/2; i <= k/2; i++)
	      {
		for(int j = -k/2; j <= k/2; j++)
		  {
		    window[w_index] = image[row - i][col -j];
		    w_index++;
		  }
	      }
	    median = getMedian(window, k*k);
	    medianFilteredImage[row][col] = median;
	    w_index = 0;
	  }
      }
  
    return medianFilteredImage; 
}

float** medianFilter(float** image, int height, int width, int k)
{
  float** medianFilteredImage(image);

  int window_size = k*k;
  float* window = new float[window_size];

  for(int row = k/2; row <= height - k/2 - 1; row++)
    {
      for(int col = k/2; col <= width - k/2 - 1; col++)
	{
	  int w_index = 0;
	  for(int i = -k/2; i <= k/2; i++)
	    {
	      for(int j = -k/2; j <= k/2; j++)
		{
		  window[w_index++] = image[row - i][col -j];
		}
	    }
	  float median = getMedian(window, k*k);
	  medianFilteredImage[row][col] = median;
	}
    }

  return medianFilteredImage;
}


//For saving the output after image processing is done
void matrixToFile(std::string fileName, float** image, int height, int width)
{
  std::ofstream outFile;
  outFile.open(fileName);
  outFile << height << " " << width << '\n';
  for(int i = 0; i < height; i++)
    {
      for(int j = 0; j < width; j++)
	{
	  outFile << ("%.2f", image[i][j]) << " ";
	}
      outFile << '\n';
    }

  outFile.close();
}

bool checkEquality(float** image1, float** image2, int height, int width)
{
  bool equal = true;
  for(int i = 0; i < height; i++)
    {
      for(int j = 0; j < width; j++)
	{
	  if(image1[i][j] != image2[i][j])
	    equal = false;
	}
    }

  return equal;
}


int main(int argc, char** argv)
{
  std::string inFileName = argv[1];
  std::string outFileName = argv[2];
  int writeOut = atoi(argv[3]);
  int height, width, max = -1, min = 256;

  float** image;
  fileToMatrix(image, inFileName, height, width);
  findMinAndMax(image, height, width, max, min);
  
  std::cout << std::endl;
  std::cout << "*********************************MEDIAN FILTERING COMPARISON*********************************" << std::endl;
  std::cout << std::endl;

  /*********************************MEDIAN FILTERING COMPARISON*********************************/
  double start = omp_get_wtime();
  float** medianFilteredImage = medianFilter(image, height, width, 3);  
  double dur = omp_get_wtime() - start;
  std::cout << "Median Sequential: " << dur << std::endl;
  start = omp_get_wtime();
  float** parMedianFilteredImage = par_medianFilter(image, height, width, 3);  
  dur = omp_get_wtime() - start;
  std::cout << "Median Parallel: " << dur << std::endl;
  std::cout << "Median True: " << checkEquality(medianFilteredImage, parMedianFilteredImage, height, width) << std::endl;
  if(writeOut == 1)
    {
      matrixToFile(outFileName + "_seqMedian.txt", medianFilteredImage, height, width);
      matrixToFile(outFileName + "_parMedian.txt", parMedianFilteredImage, height, width);
    }
  /*********************************MEDIAN FILTERING COMPARISON*********************************/

  std::cout << std::endl;
  std::cout << "*********************************LINEAR SCALING COMPARISON*********************************" << std::endl;
  std::cout << std::endl;

  /*********************************LINEAR SCALING COMPARISON*********************************/
  start = omp_get_wtime();
  float** linScaledImage = linearScale(image, height, width, max, min);
  dur = omp_get_wtime() - start;
  std::cout << "LinScale Sequential: " << dur << std::endl;
  start = omp_get_wtime();
  float** parLinScaledImage = par_linearScale(image, height, width, max, min);
  dur = omp_get_wtime() - start;
  std::cout << "LinScale Parallel: " << dur << std::endl;
  std::cout << "LinScale True: " << checkEquality(linScaledImage, parLinScaledImage, height, width) << std::endl;
  if(writeOut == 1)
    {
      matrixToFile(outFileName + "_seqLinear.txt", linScaledImage, height, width);
      matrixToFile(outFileName + "_parLinear.txt", parLinScaledImage, height, width);
    }
  /*********************************LINEAR SCALING COMPARISON*********************************/
	
  std::cout << std::endl;
  std::cout << "*********************************GRAYWORLD COMPARISON*********************************" << std::endl;
  std::cout << std::endl;

  double average = getAverage(image, height, width);
  /*********************************GRAYWORLD COMPARISON*********************************/
  start = omp_get_wtime();
  float** grayWorldImage = grayWorld(image, height, width, average);
  dur = omp_get_wtime() - start;
  std::cout << "GrayWorld Sequential: " << dur << std::endl;
  start = omp_get_wtime();
  float** parGrayWorldImage = par_grayWorld(image, height, width, average);
  dur = omp_get_wtime() - start;
  std::cout << "GrayWorld Parallel: " << dur << std::endl;
  std::cout << "GrayWorld True: " << checkEquality(grayWorldImage, parGrayWorldImage, height, width) << std::endl;
  if (writeOut == 1)
    {
      matrixToFile(outFileName + "_seqGrayWorld.txt", grayWorldImage, height, width);
      matrixToFile(outFileName + "_parGrayWorld.txt", parGrayWorldImage, height, width);
    }
  /*********************************GRAYWORLD COMPARISON*********************************/
  
  std::cout << std::endl;
  std::cout << "*********************************REFLECTION COMPARISON*********************************" << std::endl;
  std::cout << std::endl;
  /*********************************REFLECTION COMPARISON*********************************/
  start = omp_get_wtime();
  float** reflectedImage = reflection(image, height, width);
  dur = omp_get_wtime() - start;
  std::cout << "Reflection Sequential: " << dur << std::endl;
  start = omp_get_wtime();
  float** parReflectedImage = par_reflection(image, height, width);
  dur = omp_get_wtime() - start;
  std::cout << "Reflection Parallel: " << dur << std::endl;
  std::cout << "Reflection True: " << checkEquality(grayWorldImage, parGrayWorldImage, height, width) << std::endl;
  if (writeOut == 1)
    {
      matrixToFile(outFileName + "_seqReflection.txt", reflectedImage, height, width);
      matrixToFile(outFileName + "_parReflection.txt", parReflectedImage, height, width);
    }
  /*********************************REFLECTION COMPARISON*********************************/
  
  std::cout << std::endl;
  std::cout << "*********************************DITHERING COMPARISON*********************************" << std::endl;
  std::cout << std::endl;
  /*********************************DITHERING COMPARISON*********************************/
  start = omp_get_wtime();
  float** ditheredImage = orderedDithering(image, height, width);
  dur = omp_get_wtime() - start;
  std::cout << "Dithered Sequential: " << dur << std::endl;
  start = omp_get_wtime();
  float** parDitheredImage = par_orderedDithering(image, height, width);
  dur = omp_get_wtime() - start;
  std::cout << "Dithered Parallel: " << dur << std::endl;
  std::cout << "Dithered True: " << checkEquality(ditheredImage, parDitheredImage, height-2, width-2) << std::endl;
  if (writeOut == 1)
    {
      matrixToFile(outFileName + "_seqDithering.txt", ditheredImage, height, width);
      matrixToFile(outFileName + "_parDithering.txt", parDitheredImage, height, width);
    }
  /*********************************DITHERING COMPARISON*********************************/
  
  std::cout << std::endl;
  std::cout << "*********************************SOBEL FILTER COMPARISON*********************************" << std::endl;
  std::cout << std::endl;
  
  /*********************************SOBEL FILTER COMPARISON*********************************/
  start = omp_get_wtime();
  float** gradientX = sobelFilterX(medianFilteredImage, height, width);
  dur = omp_get_wtime() - start;
  std::cout << "SobelFilter(X) Sequential: " << dur << std::endl;
  start = omp_get_wtime();
  float** parGradientX = par_sobelFilterX(medianFilteredImage, height, width);
  dur = omp_get_wtime() - start;
  std::cout << "SobelFilter(X) Parallel: " << dur << std::endl;
  std::cout << "SobelFilter(X) True: " << checkEquality(gradientX, parGradientX, height, width) << std::endl;
  start = omp_get_wtime();
  float** gradientY = sobelFilterY(medianFilteredImage, height, width);
  dur = omp_get_wtime() - start;
  std::cout << "SobelFilter(Y) Sequential: " << dur << std::endl;
  start = omp_get_wtime();
  float** parGradientY = par_sobelFilterY(medianFilteredImage, height, width);
  dur = omp_get_wtime() - start;
  std::cout << "SobelFilter(Y) Parallel: " << dur << std::endl;
  std::cout << "SobelFilter(Y) True: " << checkEquality(gradientY, parGradientY, height, width) << std::endl;
  if(writeOut == 1)
    {
      matrixToFile(outFileName + "_seqGradX.txt", gradientX, height, width);
      matrixToFile(outFileName + "_seqGradY.txt", gradientY, height, width);
      matrixToFile(outFileName + "_parGradX.txt", parGradientX, height, width);
      matrixToFile(outFileName + "_parGradY.txt", parGradientY, height, width);
    }
  /*********************************SOBEL FILTER COMPARISON*********************************/

  std::cout << std::endl;
  std::cout << "*********************************SOBEL EDGE DETECTION COMPARISON*********************************" << std::endl;
  std::cout << std::endl;

  /*********************************SOBEL EDGE DETECTION COMPARISON*********************************/
  start = omp_get_wtime();
  float** sobelEdgeDetectedImage = sobelEdgeDetection(gradientX, gradientY, height, width, 80);
  dur = omp_get_wtime() - start;
  std::cout << "SobelEdge Sequential: " << dur << std::endl;
  start = omp_get_wtime();
  float** parSobelEdgeDetectedImage = par_sobelEdgeDetection(parGradientX, parGradientY, height, width, 80);
  dur = omp_get_wtime() - start;
  std::cout << "SobelEdge Parallel: " << dur << std::endl;
  std::cout << "SobelEdge True: " << checkEquality(sobelEdgeDetectedImage, parSobelEdgeDetectedImage, height, width) << std::endl;
  if(writeOut == 1)
    {
      matrixToFile(outFileName + "_seqSobelEdgeX.txt", sobelEdgeDetectedImage, height, width);
      matrixToFile(outFileName + "_parSobelEdge.txt", parSobelEdgeDetectedImage, height, width);
    }
  /*********************************SOBEL EDGE DETECTION COMPARISON*********************************/
  std::cout << std::endl;




  return 0;
}
