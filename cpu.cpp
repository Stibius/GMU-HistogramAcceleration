#include "cpu.h"

void equalize(cl_uchar4* inputImage, cl_uchar4* outputImage, cl_uint* histogram)
{
	int newValues[HISTOGRAM_SIZE]; //each value represents a new pixel value for a pixel value given by its index

	//computing the cumulative histogram
	newValues[0] = histogram[0];
    for (int i = 1; i < HISTOGRAM_SIZE; i++)
	{
	    newValues[i] = newValues[i-1] + histogram[i];
	}

	float numberOfPixels = newValues[HISTOGRAM_SIZE-1]; //number of pixels in the input image is the last value in the cumulative histogram

	//computing the new pixel values
	for (int i = 0; i < HISTOGRAM_SIZE; i++)
	{
	    newValues[i] *=  HISTOGRAM_SIZE / numberOfPixels;
	}

	//assigning new values to pixels of the output image
	for (int i = 0; i < numberOfPixels; i++)
	{
		int newValue = newValues[inputImage[i].s[0]]; //get new value for current pixel
		memset(outputImage[i].s, newValue, 4); //write the new value to the output image
	}
}