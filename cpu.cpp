#include "cpu.h"

#define MIN_BRIGHTNESS 0
#define MAX_BRIGHTNESS 255

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

void otsu(cl_uchar4* inputImage, cl_uchar4* outputImage, cl_uint* histogram, int width, int height)
{
   long total = 0;
   float sum = 0;
   float sumB = 0, varMax = 0, varBetween;
   long wB = 0,  wF = 0, threshold = 0;
   float mB, mF;  

   for (int i = 0 ; i < HISTOGRAM_SIZE ; i++) {
      sum += i * histogram[i];
      total += histogram[i];
   }

   for (int i = 0 ; i < HISTOGRAM_SIZE; i++) {

      wB += histogram[i];             // Weight Background
      
	  if (wB == 0) 
		  continue;

      wF = total - wB;           // Weight Foreground
      if (wF == 0) 
		  break;

      sumB += (float) (i * histogram[i]);

      mB = sumB / wB;            // Mean Background 
      mF = (sum - sumB) / wF;    // Mean Foreground

      // Calculate Between Class Variance 
      varBetween = (float)wB * (float)wF * (mB - mF) * (mB - mF);

      // Check if new maximum found 
      if (varBetween > varMax) {
         varMax = varBetween;
         threshold = i+1;
      }
   }

	//assigning new values to pixels of the output image
	for (int i = 0; i < (width*height); i++)
	{
		if(inputImage[i].s[0] > threshold)
		{
			memset(outputImage[i].s, MAX_BRIGHTNESS, 4); 
		} else {
			memset(outputImage[i].s, MIN_BRIGHTNESS, 4); 
		}
	}


  //printf("tresh %d\n", threshold);
}