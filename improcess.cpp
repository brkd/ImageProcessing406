#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>

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
  findMinAndMax(image, height, width, max, min);
  float** linScaledImage = linearScale(image, height, width, max, min);
  
  matrixToFile(outFileName, linScaledImage, height, width);

  return 0;
}
