#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>


/*struct Window
{
  int size;
  int center_x;
  int center_y;
  float* local_window;
  Window(int size, int x, int y, float** image): size(size), center_x(x), center_y(y)
  {
    local_window = new float[size * size];
    for(int i = size; i < ; i++)
      {
	local_window[i] = im
      }
  }
};*/


//For reading the pixel values from given file name
float** fileToMatrix(std::string fileName, int& height, int& width)
{
  std::ifstream imageFile(fileName);    
  float** image;
  
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

  return image;
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

float** medianFilter(float** image, int height, int width, int k)
{
  /*float** medianFilteredImage = new float*[height];
  for(int i = 0; i < height; i++)
    {
      medianFilteredImage[i] = new float[width];
      }*/
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
	  if(row == 5 && col == 0)
	    std::cout << median << " " << "kek" << std::endl;
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
	  outFile << std::fixed << std::setprecision(2) << image[i][j] << " ";
	}
      outFile << '\n';
    }

  outFile.close();
}


int main(int argc, char** argv)
{
  std::string inFileName = argv[1];
  std::string outFileName = argv[2];
  int height, width, max = -1, min = 256;

  float** image = fileToMatrix(inFileName, height, width);

  //findMinAndMax(image, height, width, max, min);
  //float** linScaledImage = linearScale(image, height, width, max, min);

  float** medianFilteredImage = medianFilter(image, height, width, 3);
  
  matrixToFile(outFileName, medianFilteredImage, height, width);

  return 0;
}
