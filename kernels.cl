
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

__constant uint HISTOGRAM_SIZE = 256;
__constant uint SIZE_OF_BLOCK = 16;


/*! Computes histogram of the input image in grayscale format with 255 levels of gray.
 *
 * \param[in] inputImage input image in grayscale format with 255 levels of gray
 * \param[in] width input image width
 * \param[in] height input image height
 * \param[out] histogram resulting histogram, an array of 255 integer values
 * \param[in] cache used for histogram values of a workgroup
 */
 __kernel void histogram( __global uchar4* inputImage, uint width, uint height, __global uint* histogram, __local uint* cache)
{
    //two dimensional matrix of work items
	int globalX = get_global_id(0);
	int globalY = get_global_id(1); 
	int localX = get_local_id(0);
	int localY = get_local_id(1);
	
	//first worker initializes histogram data
	if (globalX == 0 && globalY == 0 && localX == 0 && localY == 0) 
	{
		for (int i = 0; i < HISTOGRAM_SIZE; i++) 
		{
	        histogram[i] = 0;
	    }
	}
	
	//first local worker initializes cache data
	if (localX == 0 && localY == 0) 
	{
		for (int i = 0; i < HISTOGRAM_SIZE; i++) 
		{
		   cache[i] = 0;
		}
	}
	
	//wait until all local workers have initialized cache
	barrier(CLK_LOCAL_MEM_FENCE);
	barrier(CLK_GLOBAL_MEM_FENCE);

	if (globalX < width && globalY < height) //check if we are out of bounds
	{
	    int value = inputImage[globalY * width + globalX].x; //current pixel value

		atomic_inc(&cache[value]); //updating cache
		//cache[value]++;

		barrier(CLK_LOCAL_MEM_FENCE); //wait until all local workers have updated cache

		//first local worker adds the results from cache to global memory
		if (localX == 0 && localY == 0) 
		{
			for (int i = 0; i < HISTOGRAM_SIZE; i++) 
			{
				atomic_add(&histogram[i], cache[i]);
				//histogram[i] += cache[i];
			}
		}

	} 
}


/*! First part of the histogram equalization, determines a new pixel value for each possible pixel value in the input image.
 *
 * \param[in] histogram histogram of the input image, 255 values
 * \param[out] newValues an array of 255 values, each value represents a new pixel value for a pixel value given by its index
 */
__kernel void equalize1(__global uint* histogram, __global uint* newValues)
{
    //one dimensional matrix of work items
	uint globalX = get_global_id(0);
	uint localX = get_local_id(0);
	
	//first worker computes the cumulative histogram
	if (globalX == 0 && localX == 0)
	{
	    newValues[0] = histogram[0];
	    for (int i = 1; i < HISTOGRAM_SIZE; i++)
	    {
	        newValues[i] = newValues[i-1] + histogram[i];
	    }
	}
	
	//wait for the first worker to finish
	barrier(CLK_GLOBAL_MEM_FENCE);
	
	float numberOfPixels = newValues[HISTOGRAM_SIZE-1]; //number of pixels in the input image is the last value in the cumulative histogram
	newValues[globalX] *= HISTOGRAM_SIZE / numberOfPixels; //computing final output

	return;
}

/*! Second part of the histogram equalization, creates an output image from the input image using the output from the first part.
 *
 * \param[in] inputImage input image in grayscale format with 255 levels of gray
 * \param[out] outputImage an equalized input image in grayscale format with 255 levels of gray
 * \param[in] newValues an array of 255 values, each value represents a new pixel value for a pixel value given by its index
 * \param[in] width width of the input image
 * \param[in] height height of the input image
 */
__kernel void equalize2(__global uchar4* inputImage, __global uchar4* outputImage, __global uint* newValues, uint width, uint height)
{
    //two dimensional matrix of work items
    uint globalX = get_global_id(0);
	uint globalY = get_global_id(1);
	
	if (globalX < width && globalY < height) //chceck if we are out of bounds
	{
        uchar newValue = newValues[inputImage[globalY * width + globalX].x]; //get new value for current pixel
		outputImage[globalY * width + globalX] = (newValue, newValue, newValue, newValue); //write the new value to the output image
	}
	
	return;
}

__kernel void threshold(__constant uint* histogram, __global ulong* sumHistogram)
{
	int gid = get_local_id(0);

    local ulong sum[16];
    sum[gid] = 0;
	local ulong total[16];
	total[gid] = 0;

	float sumB = 0, varMax = 0, varBetween;
	long wB = 0,  wF = 0, threshold = 0;
	float mB, mF; 

    int i;

	for(i = gid*SIZE_OF_BLOCK; i < (gid*SIZE_OF_BLOCK)+SIZE_OF_BLOCK; i++)
	{
		sum[gid] += i * histogram[i];
		total[gid] += histogram[i];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	if(gid == 0){
        for(i = 1; i < SIZE_OF_BLOCK; i++){
            sum[0] += sum[i];
			total[0] += total[i];
        }

		for (i = 0 ; i < 256; i++) {

			wB += histogram[i];             // Weight Background
      
			if (wB == 0) 
				continue;

			wF = total[0] - wB;           // Weight Foreground
			if (wF == 0) 
				break;

			sumB += (i * histogram[i]);

			mB = sumB / wB;            // Mean Background 
			mF = (sum[0] - sumB) / wF;    // Mean Foreground

			// Calculate Between Class Variance 
			varBetween = wB * wF * (mB - mF) * (mB - mF);

			// Check if new maximum found 
			if (varBetween > varMax) {
				varMax = varBetween;
				threshold = i+1;
			}
		}
		sumHistogram[0] = threshold;
	}
	

	return;
}

__kernel void thresholding(__global uchar4* inputImage, __global uchar4* outputImage, __global uint* threshold, uint width, uint height)
{
    uint globalX = get_global_id(0);
	uint globalY = get_global_id(1);
	
	if (globalX < width && globalY < height)
	{
		if(inputImage[globalY * width + globalX].x > threshold[0]){
			outputImage[globalY * width + globalX] = (255, 255, 255, 255);
		} else {
			outputImage[globalY * width + globalX] = (0, 0, 0, 0); 
		}
	}

	return;
}

/*! Image Segmentation by Adaptive Histogram Thresholding
 *
 * \param[in] inputImage input image in grayscale format with 255 levels of gray
 * \param[out] outputImage an equalized input image in grayscale format with 255 levels of gray
 * \param[in] width width of the input image
 * \param[in] height height of the input image
 */
__kernel void segmentation(__global uchar4* inputImage, __global uchar4* outputImage, uint width, uint height)
{
    uint globalX = get_global_id(0);
	uint globalY = get_global_id(1);
	
    //local histogram for every pixel (subimage)
    int subHist[255];

    int threshold;
    int yy;
    int xx;
    int iter;
    int i;
    
    //initial threshold 
    threshold = 128;

    //initialize memory
    for (i = 0; i < HISTOGRAM_SIZE; i++) 
	{
	    subHist[i] = 0;
	}
    
    //compute histogram of 31 x 31 subimage around pixel
    for (yy = 0; yy <= 30; yy++)
    {
        int subPosY = globalY + yy - 15;

        //check bounds
        if (subPosY < 0 || subPosY >= height)
            continue;

        for (xx = 0; xx <= 30; xx++)
        {
            uint subPosX = globalX + xx - 15;
            //check bounds
            if (subPosX < 0 || subPosX >= width)
                continue;

            //modify histogram
            subHist[inputImage[subPosY * width + subPosX].x]++;
        }
    }       
    //i have histogram

    
    int mean1 = 0;
    int mean2 = 0;
    int newTh = 128;
    long sum = 0;
    long count = 0;

    //iterrate and try to find "ideal" threshold
    for (iter = 0; iter < 15; iter++)
    {       
        for (i = 0; i < 255; i++)
        {
            //separate histogram by threshold and compute mean of all values bellow nad above threshold
            if (i <= threshold)
            {
                //lower part
                count += subHist[i];
                sum += subHist[i] * i;
                    
                if (i == threshold)
                {
                    //last
                    if (count == 0) //if no valid samples, move mean to threshold
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
        if (count == 0) //if no valid samples, move mean to threshold
            mean2 = threshold;
        else
            mean2 = sum / count;
        sum = count = 0;

        //compute new threshold
        newTh = (mean1 + mean2) / 2;

        //is aproximated? - no change
        if (abs(newTh - threshold) < 3)
            break;

        threshold = newTh;
    }
    threshold = newTh;
    
    //dont let threshold move to borders
    if (threshold < 20)
        threshold = 20;
           
    if (threshold > 255 - 20)
        threshold = 255 - 20;
    
    //perform segmentation
    if (inputImage[globalY * width + globalX].x <= threshold)
    {
        outputImage[globalY * width + globalX] = (0, 0, 0, 0); 
    }
    else
    {
        outputImage[globalY * width + globalX] = (255, 255, 255, 255);
    }
    
	return;
}

