#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>

#define smooth_kernel_size 9
#define sigma 1.0

int main() 
{
	// based on https://stackoverflow.com/questions/54614167/trying-to-implement-gaussian-filter-in-c

	double gauss[smooth_kernel_size][smooth_kernel_size];
	double sum = 0;
	int i, j;

	for (i = 0; i < smooth_kernel_size; i++) {
		for (j = 0; j < smooth_kernel_size; j++) {
			double x = i - (smooth_kernel_size - 1) / 2.0;
			double y = j - (smooth_kernel_size - 1) / 2.0;
			gauss[i][j] = 1.0 / (2.0 * M_PI * pow(sigma, 2.0)) * exp(-(pow(x, 2) + pow(y, 2)) / (2 * pow(sigma, 2)));
			sum += gauss[i][j];
		}
	}

	for (i = 0; i < smooth_kernel_size; i++) {
		for (j = 0; j < smooth_kernel_size; j++) {
			gauss[i][j] /= sum;
		}
	}

	printf("2D Gaussian filter kernel:\n");
	for (i = 0; i < smooth_kernel_size; i++) {
		for (j = 0; j < smooth_kernel_size; j++) {
			printf("%f, ", gauss[i][j]);
		}
		printf("\n");
	}

	double gaussSeparated[smooth_kernel_size];

	for (i = 0; i < smooth_kernel_size; i++) {
		gaussSeparated[i] = sqrt(gauss[i][i]);
	}
		
	printf("1D Separated Gaussian filter kernel:\n");
	for (i = 0; i < smooth_kernel_size; i++) {
		printf("%f, ", gaussSeparated[i]);
	}
	printf("\n");

	return 0;
}