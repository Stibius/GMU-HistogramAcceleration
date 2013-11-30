
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

__constant uint HISTOGRAM_SIZE = 256;

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


