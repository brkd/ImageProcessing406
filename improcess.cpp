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

	if (imageFile.is_open())
	{
		std::string line;
		getline(imageFile, line);
		std::istringstream iss(line);
		iss >> height >> width;
		std::cout << "Height: " << height << " ||-|| Width: " << width << std::endl;

		image = new float*[height];

		for (int i = 0; i < height; i++)
		{
			image[i] = new float[width];
		}

		int h = 0;
		float val;
		while (getline(imageFile, line))
		{
			int w = 0;
			std::istringstream iss(line);
			while (iss >> val)
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
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			std::cout << image[i][j] << " ";
		}
		std::cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << std::endl;
	}
}
void findMinAndMax(float** image, int height, int width, int& max, int& min)
{
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (max < image[i][j])
				max = image[i][j];

			if (min > image[i][j])
				min = image[i][j];
		}
	}
}

float** orderedDithering(float** image, int height, int width)
{
  float** ditheredImage = new float*[height];
	for (int i = 0; i < height; i++)
	{
		ditheredImage[i] = new float[width];
	}

	for (int i = 0; i < height - 2; i += 2)
	{
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

float** par_orderedDithering(float** image, int height, int width)
{
	  float** ditheredImage = new float*[height];
	for (int i = 0; i < height; i++)
	{
		ditheredImage[i] = new float[width];
	}

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


float** simd_orderedDithering(float** image, int height, int width)
{
	  float** ditheredImage = new float*[height];
	for (int i = 0; i < height; i++)
	{
		ditheredImage[i] = new float[width];
	}

	for (int i = 0; i < height - 2; i += 2)
	{
		#pragma simd
		#pragma vector aligned
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

float** simd_grayWorld(float** image, int height, int width, double average)
{
	float** scaledImage = new float*[height];

	double scalingValue = 127.5 / average;

	for (int i = 0; i < height; i++)
	{
		scaledImage[i] = new float[width];
	}

	#pragma simd
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			#pragma vector aligned
			scaledImage[i][j] = image[i][j] * scalingValue;
		}
	}

	return scaledImage;
}

float** reflection(float** image, int height, int width)
{
 	float** reflectedImage = new float*[height];
	for (int i = 0; i < height; i++)
	{
		reflectedImage[i] = new float[width];
	}

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
	 	float** reflectedImage = new float*[height];
	for (int i = 0; i < height; i++)
	{
		reflectedImage[i] = new float[width];
	}

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

float** simd_reflection(float** image, int height, int width)
{
	 	float** reflectedImage = new float*[height];
	for (int i = 0; i < height; i++)
	{
		reflectedImage[i] = new float[width];
	}

	#pragma simd
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{	
			
			#pragma vector aligned
			reflectedImage[i][j] = image[i][width - j - 1];
		}
	}
	return reflectedImage;
	
}

float** rotate90(float **image, int height, int width) {
	float **rotated;
	rotated = new float*[width];
	for (int i = 0; i < width; ++i)
		rotated[i] = new float[height];

	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			rotated[i][j] = image[height - j - 1][i];
		}
	}
	return rotated;
}

float** rotate90_par(float **image, int height, int width) {
	float **rotated;
	rotated = new float*[width];
#pragma omp parallel for
	for (int i = 0; i < width; ++i)
		rotated[i] = new float[height];

#pragma omp parallel for collapse(2)
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			rotated[i][j] = image[height - j - 1][i];
		}
	}
	return rotated;
}

float** rotate90_simd(float **image, int height, int width) {
	float **rotated;
	rotated = new float*[width];
#pragma omp parallel for
	for (int i = 0; i < width; ++i)
		rotated[i] = new float[height];

#pragma simd
#pragma omp parallel for collapse(2)
	for (int i = 0; i < width; i++) {
		#pragma vector aligned
		for (int j = 0; j < height; j++) {
			rotated[i][j] = image[height - j - 1][i];
		}
	}
	return rotated;
}

float** rotate180(float **image, int height, int width) {
	float **rotated1 = rotate90(image, height, width);
	float **result = rotate90(rotated1, width, height);
	return result;
}

float** rotate180_par(float **image, int height, int width) {
	float **rotated1 = rotate90_par(image, height, width);
	float **result = rotate90_par(rotated1, width, height);
	return result;
}

float** rotate180_simd(float **image, int height, int width) {
	float **rotated1 = rotate90_simd(image, height, width);
	float **result = rotate90_simd(rotated1, width, height);
	return result;
}

float** rotate270(float **image, int height, int width) {
	float **rotated;
	rotated = new float*[width];
	for (int i = 0; i < width; ++i)
		rotated[i] = new float[height];


	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			rotated[i][j] = image[j][width - i - 1];
		}
	}
	return rotated;
}

float** rotate270_par(float **image, int height, int width) {
	float **rotated;
	rotated = new float*[width];
	#pragma omp parallel for
	for (int i = 0; i < width; ++i)
		rotated[i] = new float[height];

	#pragma omp parallel for collapse(2)
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			rotated[i][j] = image[j][width - i - 1];
		}
	}
	return rotated;
}

float** rotate270_simd(float **image, int height, int width) {
	float **rotated;
	rotated = new float*[width];
	#pragma omp parallel for
	for (int i = 0; i < width; ++i)
		rotated[i] = new float[height];

	#pragma simd
	#pragma omp parallel for collapse(2)
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			#pragma vector aligned
			rotated[i][j] = image[j][width - i - 1];
		}
	}
	return rotated;
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
	for (int i = 0; i < height; i++)
	{
		scaledImage[i] = new float[width];
	}

	int a = -1 * min;
	float gmax = 255.0;
	float b;
	float u;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			u = image[i][j];
			b = gmax / (max - min);

			scaledImage[i][j] = b * (u + a);
		}
	}

	return scaledImage;
}

float** par_linearScale(float** image, int height, int width, int max, int min)
{
	float** scaledImage = new float*[height];
	for (int i = 0; i < height; i++)
	{
		scaledImage[i] = new float[width];
	}

	int a = -1 * min;
	float gmax = 255.0;
	float b;
	float u;

#pragma omp parallel for collapse(2) private(u, b)
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			u = image[i][j];
			b = gmax / (max - min);

			scaledImage[i][j] = b * (u + a);
		}
	}

	return scaledImage;
}

float** simd_linearScale(float** image, int height, int width, int max, int min)
{
	float** scaledImage = new float*[height];
	for (int i = 0; i < height; i++)
	{
		scaledImage[i] = new float[width];
	}

	int a = -1 * min;
	float gmax = 255.0;
	float b;
	float u;

#pragma simd
#pragma omp parallel for collapse(2) private(u, b)
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{	
			
			#pragma vector aligned
			u = image[i][j];
			b = gmax / (max - min);

			scaledImage[i][j] = b * (u + a);
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

	for (int i = 0; i < height; i++)
	{
		gradientX[i] = new float[width];
	}

	float x_filter[3][3];

	x_filter[0][0] = -1;  x_filter[0][1] = 0; x_filter[0][2] = 1;
	x_filter[1][0] = -2;  x_filter[1][1] = 0; x_filter[1][2] = 2;
	x_filter[2][0] = -1;  x_filter[2][1] = 0; x_filter[2][2] = 1;

	float local_result = 0.0;
#pragma omp parallel for collapse(2) firstprivate(local_result)
	for (int i = 1; i < height - 1; i++)
	{
		for (int j = 1; j < width - 1; j++)
		{
			for (int a = 0; a < 3; a++)
			{
				for (int b = 0; b < 3; b++)
				{
					local_result = local_result + image[i - 1 + a][j - 1 + b] * x_filter[a][b];
				}
			}
			if (local_result < 0) local_result = 0;
			if (local_result > 255) local_result = 255;
			gradientX[i][j] = local_result;
			local_result = 0.0;
		}
	}

	return gradientX;
}

float** sobelFilterX(float** image, int height, int width)
{
	float** gradientX = new float*[height];

	for (int i = 0; i < height; i++)
	{
		gradientX[i] = new float[width];
	}

  for(int j = 0; j < width - 1; j++)
    gradientX[0][j] = 0;

	float x_filter[3][3];

	x_filter[0][0] = -1;  x_filter[0][1] = 0; x_filter[0][2] = 1;
	x_filter[1][0] = -2;  x_filter[1][1] = 0; x_filter[1][2] = 2;
	x_filter[2][0] = -1;  x_filter[2][1] = 0; x_filter[2][2] = 1;

	float local_result = 0.0;
	for (int i = 1; i < height - 1; i++)
	{
		for (int j = 1; j < width - 1; j++)
		{
			for (int a = 0; a < 3; a++)
			{
				for (int b = 0; b < 3; b++)
				{
					local_result += image[i - 1 + a][j - 1 + b] * x_filter[a][b];
				}
			}
			if (local_result < 0) local_result = 0;
			if (local_result > 255) local_result = 255;
			gradientX[i][j] = local_result;
       
			local_result = 0.0;
		}
	}

	return gradientX;
}

float** par_sobelFilterY(float** image, int height, int width)
{
	float** gradientY = new float*[height];

	for (int i = 0; i < height; i++)
	{
		gradientY[i] = new float[width];
	}

	float y_filter[3][3];

	y_filter[0][0] = -1;  y_filter[0][1] = -2; y_filter[0][2] = -1;
	y_filter[1][0] = 0;  y_filter[1][1] = 0; y_filter[1][2] = 0;
	y_filter[2][0] = 1;  y_filter[2][1] = 2; y_filter[2][2] = 1;

	float local_result = 0.0;
#pragma omp parallel for collapse(2) firstprivate(local_result)
	for (int i = 1; i < height - 1; i++)
	{
		for (int j = 1; j < width - 1; j++)
		{
			for (int a = 0; a < 3; a++)
			{
				for (int b = 0; b < 3; b++)
				{
					local_result += image[i - 1 + a][j - 1 + b] * y_filter[a][b];
				}
			}
			if (local_result < 0) local_result = 0;
			if (local_result > 255) local_result = 255;
			gradientY[i][j] = local_result;
			local_result = 0.0;
		}
	}

	return gradientY;
}

float** sobelFilterY(float** image, int height, int width)
{
	float** gradientY = new float*[height];

	for (int i = 0; i < height; i++)
	{
		gradientY[i] = new float[width];
	}

    for(int j = 0; j < width - 1; j++)
    gradientY[0][j] = 0;


	float y_filter[3][3];

	y_filter[0][0] = -1;  y_filter[0][1] = -2; y_filter[0][2] = -1;
	y_filter[1][0] = 0;  y_filter[1][1] = 0; y_filter[1][2] = 0;
	y_filter[2][0] = 1;  y_filter[2][1] = 2; y_filter[2][2] = 1;

	float local_result = 0.0;
	//printImageMatrix(image, height, width);
	for (int i = 1; i < height - 1; i++)
	{
		for (int j = 1; j < width - 1; j++)
		{
			for (int a = 0; a < 3; a++)
			{
				for (int b = 0; b < 3; b++)
				{
					local_result += image[i - 1 + a][j - 1 + b] * y_filter[a][b];
				}
			}
			if (local_result < 0) local_result = 0;
			if (local_result > 255) local_result = 255;
			gradientY[i][j] = local_result;
			local_result = 0.0;
		}
	}

	return gradientY;
}

float** par_sobelEdgeDetection(float** gradX, float** gradY, int height, int width, float T)
{
	float** sobelEdgeDetectedImage = new float*[height];

	for (int i = 0; i < height; i++)
	{
		sobelEdgeDetectedImage[i] = new float[width];
	}

	float local_sum = 0;
#pragma omp parallel for collapse(2) firstprivate(local_sum)
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			local_sum = (gradX[i][j] * gradX[i][j]) + (gradY[i][j] * gradY[i][j]);
			local_sum = sqrt(local_sum);

			if (local_sum > T)
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

	for (int i = 0; i < height; i++)
	{
		sobelEdgeDetectedImage[i] = new float[width];
	}

	float local_sum = 0;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			local_sum = (gradX[i][j] * gradX[i][j]) + (gradY[i][j] * gradY[i][j]);
			local_sum = sqrt(local_sum);

			if (local_sum > T)
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

	for (int i = 0; i < height; i++)
	{
		warpedImage[i] = new float[width];
	}

	int x, y, x_cursor, y_cursor, center_x = width / 2, center_y = height / 2;
	float radius, theta;
	float DRAD = 180.0f / PI;

	for (y_cursor = 0; y_cursor < height; y_cursor++)
	{
		for (x_cursor = 0; x_cursor < width; x_cursor++)
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

	const int window_size = k * k;
	float window[9];

	float median;
	int w_index = 0;
#pragma omp parallel for collapse(2) private(median, window, w_index)
	for (int row = k / 2; row <= height - k / 2 - 1; row++)
	{
		for (int col = k / 2; col <= width - k / 2 - 1; col++)
		{
			for (int i = -k / 2; i <= k / 2; i++)
			{
				for (int j = -k / 2; j <= k / 2; j++)
				{
					window[w_index] = image[row - i][col - j];
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

float** simd_medianFilter(float** image, int height, int width, const int k)
{
	float** medianFilteredImage(image);

	const int window_size = k * k;
	float window[9];

	float median;
	int w_index = 0;

#pragma simd
#pragma vector aligned
#pragma omp parallel for collapse(2) private(median, window, w_index)
	for (int row = k / 2; row <= height - k / 2 - 1; row++)
	{
		for (int col = k / 2; col <= width - k / 2 - 1; col++)
		{
			for (int i = -k / 2; i <= k / 2; i++)
			{
				for (int j = -k / 2; j <= k / 2; j++)
				{
					window[w_index] = image[row - i][col - j];
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

	int window_size = k * k;
	float* window = new float[window_size];

	for (int row = k / 2; row <= height - k / 2 - 1; row++)
	{
		for (int col = k / 2; col <= width - k / 2 - 1; col++)
		{
			int w_index = 0;
			for (int i = -k / 2; i <= k / 2; i++)
			{
				for (int j = -k / 2; j <= k / 2; j++)
				{
					window[w_index++] = image[row - i][col - j];
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
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
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
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (image1[i][j] != image2[i][j])
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
	double start, dur;
	float** image;
	fileToMatrix(image, inFileName, height, width);
	findMinAndMax(image, height, width, max, min);


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
	start = omp_get_wtime();
	float** simdGrayWorldImage = simd_grayWorld(image, height, width, average);
	dur = omp_get_wtime() - start;
	std::cout << "GrayWorld SIMD: " << dur << std::endl;
	std::cout << "GrayWorld Seq - Par True: " << checkEquality(grayWorldImage, parGrayWorldImage, height, width) << std::endl;
	std::cout << "GrayWorld Seq - SIMD True: " << checkEquality(grayWorldImage, simdGrayWorldImage, height, width) << std::endl;
	if (writeOut == 1)
	{
		matrixToFile(outFileName + "_seqGrayWorld.txt", grayWorldImage, height, width);
		matrixToFile(outFileName + "_parGrayWorld.txt", parGrayWorldImage, height, width);
		matrixToFile(outFileName + "_simdGrayWorld.txt", simdGrayWorldImage, height, width);
	}
	/*********************************GRAYWORLD COMPARISON*********************************/

  std::cout << std::endl;
	std::cout << "*********************************MEDIAN FILTER COMPARISON*********************************" << std::endl;
	std::cout << std::endl;

	/*********************************MEDIAN FILTER COMPARISON*********************************/
	start = omp_get_wtime();
	float** medianFilteredImage = medianFilter(image, height, width, 3);
	dur = omp_get_wtime() - start;
	std::cout << "Median Filter Sequential: " << dur << std::endl;
	start = omp_get_wtime();
	float** parMedianFilteredImage = par_medianFilter(image, height, width, 3);
	dur = omp_get_wtime() - start;
	std::cout << "Median Filter Parallel: " << dur << std::endl;
	start = omp_get_wtime();
	float** simdMedianFilteredImage = simd_medianFilter(image, height, width, 3);
	dur = omp_get_wtime() - start;
	std::cout << "Median Filter SIMD: " << dur << std::endl;
	std::cout << "MedianFilter Seq - Par True: " << checkEquality(medianFilteredImage, parMedianFilteredImage, height, width) << std::endl;
	std::cout << "MedianFilter Seq - SIMD True: " << checkEquality(medianFilteredImage, simdMedianFilteredImage, height, width) << std::endl;
	if (writeOut == 1)
	{
		matrixToFile(outFileName + "_seqMedianFilter.txt", medianFilteredImage, height, width);
		matrixToFile(outFileName + "_parMedianFilter.txt", parMedianFilteredImage, height, width);
		matrixToFile(outFileName + "_simdMedianFilter.txt", simdMedianFilteredImage, height, width);
	}
	/*********************************MEDIAN FILTER COMPARISON*********************************/

  std::cout << std::endl;
	std::cout << "*********************************SOBEL FILTER X COMPARISON*********************************" << std::endl;
	std::cout << std::endl;

	/*********************************SOBEL FILTER X COMPARISON*********************************/
	start = omp_get_wtime();
	float** xFilteredImage = sobelFilterX(image, height, width);
	dur = omp_get_wtime() - start;
	std::cout << "Sobel Filter X Sequential: " << dur << std::endl;
	start = omp_get_wtime();
	float** parXFilteredImage = par_sobelFilterX(image, height, width);
	dur = omp_get_wtime() - start;
	std::cout << "Sobel Filter X Parallel: " << dur << std::endl;

	std::cout << "Sobel Filter X Seq - Par True: " << checkEquality(xFilteredImage, parXFilteredImage, height, width) << std::endl;
 
	if (writeOut == 1)
	{
		matrixToFile(outFileName + "_seqSobelFilterX.txt", xFilteredImage, height, width);
		matrixToFile(outFileName + "_parSobelFilterX.txt", parXFilteredImage, height, width);
	}
	/*********************************SOBEL FILTER X COMPARISON*********************************/
 
   std::cout << std::endl;
	std::cout << "*********************************SOBEL FILTER Y COMPARISON*********************************" << std::endl;
	std::cout << std::endl;

	/*********************************SOBEL FILTER Y COMPARISON*********************************/
	start = omp_get_wtime();
	float** yFilteredImage = sobelFilterY(image, height, width);
	dur = omp_get_wtime() - start;
	std::cout << "Sobel Filter Y Sequential: " << dur << std::endl;
	start = omp_get_wtime();
	float** parYFilteredImage = par_sobelFilterY(image, height, width);
	dur = omp_get_wtime() - start;
	std::cout << "Sobel Filter Y Parallel: " << dur << std::endl;

	std::cout << "Sobel Filter Y Seq - Par True: " << checkEquality(yFilteredImage, parYFilteredImage, height, width) << std::endl;
 
	if (writeOut == 1)
	{
		matrixToFile(outFileName + "_seqSobelFilterY.txt", yFilteredImage, height, width);
		matrixToFile(outFileName + "_parSobelFilterY.txt", parYFilteredImage, height, width);
	}
	/*********************************SOBEL FILTER Y COMPARISON*********************************/
 
   std::cout << std::endl;
	std::cout << "*********************************SOBEL EDGE DETECTION COMPARISON*********************************" << std::endl;
	std::cout << std::endl;

	/*********************************SOBEL EDGE DETECTION COMPARISON*********************************/
	start = omp_get_wtime();
	float** edgeDetectionImage = sobelEdgeDetection(xFilteredImage, yFilteredImage, height, width, 50);
	dur = omp_get_wtime() - start;
	std::cout << "Sobel Edge Detection Sequential: " << dur << std::endl;
	start = omp_get_wtime();
	float** parEdgeDetectionImage = par_sobelEdgeDetection(xFilteredImage, yFilteredImage, height, width, 50);
	dur = omp_get_wtime() - start;
	std::cout << "Sobel Edge Detection Parallel: " << dur << std::endl;

	std::cout << "Sobel Edge Detection Seq - Par True: " << checkEquality(edgeDetectionImage, parEdgeDetectionImage, height, width) << std::endl;
 
	if (writeOut == 1)
	{
		matrixToFile(outFileName + "_seqEdgeDetection.txt", edgeDetectionImage, height, width);
		matrixToFile(outFileName + "_parEdgeDetection.txt", parEdgeDetectionImage, height, width);
	}
	/*********************************SOBEL EDGE DETECTIONCOMPARISON*********************************/
 
 
 

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
	start = omp_get_wtime();
	float** simdReflectedImage = simd_reflection(image, height, width);
	dur = omp_get_wtime() - start;
	std::cout << "Reflection SIMD: " << dur << std::endl;
	std::cout << "Reflection Seq - Par True: " << checkEquality(reflectedImage, parReflectedImage, height, width) << std::endl;
	std::cout << "Reflection Seq - SIMD True: " << checkEquality(reflectedImage, simdReflectedImage, height, width) << std::endl;
	if (writeOut == 1)
	{
		matrixToFile(outFileName + "_seqReflection.txt", reflectedImage, height, width);
		matrixToFile(outFileName + "_parReflection.txt", parReflectedImage, height, width);
		matrixToFile(outFileName + "_simdReflection.txt", simdReflectedImage, height, width);
	}
	/*********************************REFLECTION COMPARISON*********************************/

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
	start = omp_get_wtime();
	float** simdLinScaledImage = simd_linearScale(image, height, width, max, min);
	dur = omp_get_wtime() - start;
	std::cout << "LinScale SIMD: " << dur << std::endl;
	std::cout << "LinScale Seq - Par True: " << checkEquality(linScaledImage, parLinScaledImage, height, width) << std::endl;
	std::cout << "LinScale Seq - SIMD True: " << checkEquality(linScaledImage, simdLinScaledImage, height, width) << std::endl;
	if (writeOut == 1)
	{
		matrixToFile(outFileName + "_seqLinear.txt", linScaledImage, height, width);
		matrixToFile(outFileName + "_parLinear.txt", parLinScaledImage, height, width);
		matrixToFile(outFileName + "_simdLinear.txt", simdLinScaledImage, height, width);
	}
	/*********************************LINEAR SCALING COMPARISON*********************************/

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
	start = omp_get_wtime();
	float** simdDitheredImage = simd_orderedDithering(image, height, width);
	dur = omp_get_wtime() - start;
	std::cout << "Dithered SIMD: " << dur << std::endl;
	std::cout << "Dithered Seq - Par True: " << checkEquality(ditheredImage, parDitheredImage, height - 2, width - 2) << std::endl;
	std::cout << "Dithered Seq - SIMD True: " << checkEquality(ditheredImage, simdDitheredImage, height - 2, width - 2) << std::endl;
	if (writeOut == 1)
	{
		matrixToFile(outFileName + "_seqDithering.txt", ditheredImage, height, width);
		matrixToFile(outFileName + "_parDithering.txt", parDitheredImage, height, width);
		matrixToFile(outFileName + "_simdDithering.txt", simdDitheredImage, height, width);
	}
	/*********************************DITHERING COMPARISON*********************************/

	std::cout << std::endl;
	std::cout << "*********************************ROTATE 90 COMPARISON*********************************" << std::endl;
	std::cout << std::endl;
	/*********************************ROTATE90 COMPARISON*********************************/
	start = omp_get_wtime();
	float** rotated90Image = rotate90(image, height, width);
	dur = omp_get_wtime() - start;
	std::cout << "Rotate90 Sequential: " << dur << std::endl;
	start = omp_get_wtime();
	float** parRotated90Image = rotate90_par(image, height, width);
	dur = omp_get_wtime() - start;
	std::cout << "Rotate90 Parallel: " << dur << std::endl;
	start = omp_get_wtime();
	float** simdRotated90Image = rotate90_simd(image, height, width);
	dur = omp_get_wtime() - start;
	std::cout << "Rotate90 SIMD: " << dur << std::endl;
	std::cout << "Rotate90 Seq - Par True: " << checkEquality(rotated90Image, parRotated90Image, width, height) << std::endl;
	std::cout << "Rotate90 Seq - SIMD True: " << checkEquality(rotated90Image, simdRotated90Image, width, height) << std::endl;
	if (writeOut == 1)
	{
		matrixToFile(outFileName + "_seqRotate90.txt", rotated90Image, width, height);
		matrixToFile(outFileName + "_parRotate90.txt", parRotated90Image, width, height);
		matrixToFile(outFileName + "_simdRotate90.txt", simdRotated90Image, width, height);
	}
	/*********************************ROTATE90 COMPARISON*********************************/


	std::cout << std::endl;
	std::cout << "*********************************ROTATE 180 COMPARISON*********************************" << std::endl;
	std::cout << std::endl;
	/*********************************ROTATE180 COMPARISON*********************************/
	start = omp_get_wtime();
	float** rotated180Image = rotate180(image, height, width);
	dur = omp_get_wtime() - start;
	std::cout << "Rotate180 Sequential: " << dur << std::endl;
	start = omp_get_wtime();
	float** parRotated180Image = rotate180_par(image, height, width);
	dur = omp_get_wtime() - start;
	std::cout << "Rotate180 Parallel: " << dur << std::endl;
	start = omp_get_wtime();
	float** simdRotated180Image = rotate180_simd(image, height, width);
	dur = omp_get_wtime() - start;
	std::cout << "Rotate180 SIMD: " << dur << std::endl;
	std::cout << "Rotate180 Seq - Par True: " << checkEquality(rotated180Image, parRotated180Image, height, width) << std::endl;
	std::cout << "Rotate180 Seq - SIMD True: " << checkEquality(rotated180Image, simdRotated180Image, height, width) << std::endl;
	if (writeOut == 1)
	{
		matrixToFile(outFileName + "_seqRotate180.txt", rotated180Image, height, width);
		matrixToFile(outFileName + "_parRotate180.txt", parRotated180Image, height, width);
		matrixToFile(outFileName + "_simdRotate180.txt", simdRotated180Image, height, width);
	}
	/*********************************ROTATE180 COMPARISON*********************************/
	

	std::cout << std::endl;
	std::cout << "*********************************ROTATE 270 COMPARISON*********************************" << std::endl;
	std::cout << std::endl;
	/*********************************ROTATE270 COMPARISON*********************************/
	start = omp_get_wtime();
	float** rotated270Image = rotate270(image, height, width);
	dur = omp_get_wtime() - start;
	std::cout << "Rotate270 Sequential: " << dur << std::endl;
	start = omp_get_wtime();
	float** parRotated270Image = rotate270_par(image, height, width);
	dur = omp_get_wtime() - start;
	std::cout << "Rotate270 Parallel: " << dur << std::endl;
	start = omp_get_wtime();
	float** simdRotated270Image = rotate270_simd(image, height, width);
	dur = omp_get_wtime() - start;
	std::cout << "Rotate270 SIMD: " << dur << std::endl;
	std::cout << "Rotation270 Seq - Par True: " << checkEquality(rotated270Image, parRotated270Image, width, height) << std::endl;
	std::cout << "Rotation270 Seq - SIMD True: " << checkEquality(rotated270Image, simdRotated270Image, width, height) << std::endl;
	if (writeOut == 1)
	{
		matrixToFile(outFileName + "_seqRotate270.txt", rotated270Image, width, height);
		matrixToFile(outFileName + "_parRotate270.txt", parRotated270Image, width, height);
		matrixToFile(outFileName + "_simdRotate270.txt", simdRotated270Image, width, height);
	}
	/*********************************ROTATE270 COMPARISON*********************************/



	return 0;
}
