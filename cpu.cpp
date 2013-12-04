#include "cpu.h"



#define MIN_BRIGHTNESS 0
#define MAX_BRIGHTNESS 255

#define MAX(a,b)    (((a) > (b)) ? (a) : (b))
#define MIN(a,b)    (((a) < (b)) ? (a) : (b))

void histogram(cl_uchar4* inputImage, cl_uint* histogram, int width, int height)
{
	for (int i = 0; i < HISTOGRAM_SIZE; i++)
	{
		histogram[i] = 0;
	}

	for (int i = 0; i < (width*height); i++)
	{
		histogram[inputImage[i].s[0]]++;
	}
}

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
   unsigned long total = 0;
   unsigned long sum = 0;
   float sumB = 0, varMax = 0, varBetween;
   long wB = 0,  wF = 0, threshold = 0;
   float mB, mF;  

   for (int i = 0 ; i < HISTOGRAM_SIZE; i++) {
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


void segmentation(cl_uchar4* inputImage, cl_uchar4* outputImage, int width, int height)
{
    int subHist[HISTOGRAM_SIZE];
    int threshold = HISTOGRAM_SIZE / 2;

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            //for every pixel compute histogram of sub-image

            memset(subHist, 0, HISTOGRAM_SIZE * sizeof(int));

            for (int yy = -SEG_SUB_DIAMETER; yy <= SEG_SUB_DIAMETER; yy++)
            {
                int subPosY = y + yy;
                //check bounds
                if (subPosY < 0 || subPosY >= height)
                    continue;

                for (int xx = -SEG_SUB_DIAMETER; xx <= SEG_SUB_DIAMETER; xx++)
                {
                    int subPosX = x + xx;
                    //check bounds
                    if (subPosX < 0 || subPosX >= width)
                        continue;

                    subHist[inputImage[subPosY * width + subPosX].s[0]]++;
                }
            }
            
            //i have histogram

            int mean1 = 0;
            int mean2 = 0;
            int newTh = 128;
            int sum = 0;
            int count = 0;

            for (int iter = 0; iter < 15; iter++)
            {
                for (int i = 0; i < HISTOGRAM_SIZE; i++)
                {
                    if (i <= threshold)
                    {
                        //lower part
                        count += subHist[i];
                        sum += subHist[i] * i;
                    
                        if (i == threshold)
                        {
                            //last
                            if (count == 0)
                                mean1 = threshold;
                            else
                                mean1 = sum / count;
                            sum = count = 0;
                        }
                    }
                    else
                    {
                        //upper part                        
                        count += subHist[i];
                        sum += subHist[i] * i;
                    }
                }
                if (count == 0)
                    mean2 = threshold;
                else
                    mean2 = sum / count;
                sum = count = 0;

                newTh = (mean1 + mean2) / 2;

                if (abs(newTh - threshold) < 3)
                    break;

                threshold = newTh;
            }
            threshold = newTh;

            if (threshold < SEG_TH_BORDERS)
                threshold = SEG_TH_BORDERS;
            
            if (threshold > 255 - SEG_TH_BORDERS)
                threshold = 255 - SEG_TH_BORDERS;

            if (inputImage[y * width + x].s[0] <= threshold)
            {
                memset(outputImage[y * width + x].s, MIN_BRIGHTNESS, 4); 
            }
            else
            {
                memset(outputImage[y * width + x].s, MAX_BRIGHTNESS, 4); 
            }
        }
    }
}