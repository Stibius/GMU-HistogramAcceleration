#ifndef CPU_H
#define CPU_H

#include <CL/opencl.h>
#include <vector>

const cl_uint HISTOGRAM_SIZE = 256; 

#define SEG_SUB_DIAMETER 15
#define SEG_TH_BORDERS 20

/*! Performs histogram equalization of the input image.
 *
 * \param[in] inputImage input image in grayscale format with 255 levels of gray
 * \param[out] outputImage an equalized input image in grayscale format with 255 levels of gray
 * \param[in] histogram histogram of the input image, 255 values
 */
void histogram(cl_uchar4* inputImage, cl_uint* histogram, int width, int height);
void equalize(cl_uchar4* inputImage, cl_uchar4* outputImage, cl_uint* histogram, float numberOfPixels);
void otsu(cl_uchar4* inputImage, cl_uchar4* outputImage, cl_uint* histogram, int width, int height);
void segmentation(cl_uchar4* inputImage, cl_uchar4* outputImage, int width, int height);

#endif