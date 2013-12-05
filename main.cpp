
#include "sdlwrapper.h"
#include "error.h"
#include <stdio.h>
#include <CL/opencl.h>
#include <stdlib.h>
#include "cpu.h"
#include <ctime>
#include <iostream>

using std::cout;

#define DEBUG
#define MIN(a, b) ((a) > (b) ? (b) : (a))

//global variables

cl_device_id *cdDevices = NULL;
unsigned int deviceIndex = 0;

SDL_Surface *screen;

cl_uchar4* h_inputImageData = NULL;
cl_uchar4* h_gpu_outputImageData = NULL;
cl_uint* h_gpu_histogramData = NULL;
cl_uint* h_gpu_histogramData2 = NULL;
cl_uint* h_cpu_histogramData = NULL;
cl_uchar4* h_cpu_outputImageData = NULL;
cl_uint* h_newValuesData = NULL; //mezivypocet pri ekvalizaci

//width and height of the image
int width = 0, height = 0;

int localThreadsHistogram2a;
int numSubHistograms;
int globalThreadsHistogram2a;

cl_uint pixelSize = 32; //rgba 8bits per channel

//opencl stuff
cl_context context;
cl_command_queue commandQueue;
cl_kernel histogramKernel1, histogramKernel2a, histogramKernel2b, equalizeKernel1, equalizeKernel2, thresholdKernel, thresholdingKernel, segKernel;
cl_program program;

/** CL memory buffer for images */
cl_mem d_inputImageBuffer = NULL; 
cl_mem d_histogramBuffer = NULL; 
cl_mem d_subHistogramsBuffer = NULL;
cl_mem d_outputImageBuffer = NULL; 
cl_mem d_newValuesBuffer = NULL; //mezivypocet pri ekvalizaci
cl_mem d_threshold = NULL;

cl_event event_histogram1, event_histogram2, event_equalize1, event_equalize2, event_threshold, event_thresholding, event_seg;

/** Possible methods*/
enum method_t {
	EQUALIZE,
	OTSU,
    SEGMENTATION
};

method_t method; //method for execution
int histogramMethod = 1;

//vola se po provedeni kernelu
int printTiming(cl_event event, const char* title){
    cl_ulong startTime;
    cl_ulong endTime;
    /* Display profiling info */
    cl_int status = clGetEventProfilingInfo(event,
                                     CL_PROFILING_COMMAND_START,
                                     sizeof(cl_ulong),
                                     &startTime,
                                     0);
    CheckOpenCLError(status, "clGetEventProfilingInfo.(startTime)");
       

    status = clGetEventProfilingInfo(event,
                                     CL_PROFILING_COMMAND_END,
                                     sizeof(cl_ulong),
                                     &endTime,
                                     0);
    
    CheckOpenCLError(status, "clGetEventProfilingInfo.(stopTime)");

    cl_double elapsedTime = (endTime - startTime) * 1e-6;

    printf("%s elapsedTime %.3lf ms\n", title, elapsedTime);

    return 0;
}

char* loadProgSource(const char* cFilename)
{
    // locals 
    FILE* pFileStream = NULL;
    size_t szSourceLength;

	pFileStream = fopen(cFilename, "rb");
	if(pFileStream == 0) 
	{       
		return NULL;
	}


    // get the length of the source code
    fseek(pFileStream, 0, SEEK_END); 
    szSourceLength = ftell(pFileStream);
    fseek(pFileStream, 0, SEEK_SET); 

    // allocate a buffer for the source code string and read it in
    char* cSourceString = (char *)malloc(szSourceLength + 1); 
    if (fread(cSourceString, szSourceLength, 1, pFileStream) != 1)
    {
        fclose(pFileStream);
        free(cSourceString);
        return 0;
    }

    // close the file and return the total length of the combined (preamble + source) string
    fclose(pFileStream);
    cSourceString[szSourceLength] = '\0';

    return cSourceString;
}

/**
 * Draw the output image to sdl surface
 */
int drawOutputImage(SDL_Surface *screen){

	
    SDL_Surface *temp = SDL_CreateRGBSurfaceFrom(h_gpu_outputImageData,
        width, height, pixelSize, width*4, 
        0x0000ff, 0x00ff00, 0xff0000, 0xff000000);
    SDL_Rect rec;
 
    rec.x = 0;
    rec.y = 0;
    rec.w = width;
    rec.h = height;
    
    SDL_Surface *output = SDL_DisplayFormatAlpha(temp);
    SDL_BlitSurface(output, &rec, screen, &rec);
    SDL_FreeSurface(temp);
    SDL_FreeSurface(output);
    return 0;
}

/**
 * Draw the output image to sdl surface
 */
int drawOutputImageCPU(SDL_Surface *screen){

	
    SDL_Surface *temp = SDL_CreateRGBSurfaceFrom(h_cpu_outputImageData,
        width, height, pixelSize, width*4, 
        0x0000ff, 0x00ff00, 0xff0000, 0xff000000);
    SDL_Rect rec;
 
    rec.x = 0;
    rec.y = 0;
    rec.w = width;
    rec.h = height;
    
    SDL_Surface *output = SDL_DisplayFormatAlpha(temp);
    SDL_BlitSurface(output, &rec, screen, &rec);
    SDL_FreeSurface(temp);
    SDL_FreeSurface(output);
    return 0;
}

void toGrayScale(cl_uchar4* imageData, int size)
{
	for (int i = 0; i < size; i++)
	{
	    cl_uchar red = imageData[i].s[0];
	    cl_uchar green = imageData[i].s[1];
	    cl_uchar blue = imageData[i].s[2];

		cl_uchar new_value = 0.299 * red + 0.587 * green + 0.114 * blue;

		imageData[i].s[0] = new_value;
		imageData[i].s[1] = new_value;
		imageData[i].s[2] = new_value;
	}
}

/**
 * Initialize stuff on the client side
 */
int setupHost(const char *inputImageName)
{
	SDL_Surface *inputImage;

	if(readImage(inputImageName, &inputImage) < 0)
	{
		return -1;
	}

	width = inputImage->w;
	height = inputImage->h;

	localThreadsHistogram2a = 128;
	globalThreadsHistogram2a = (width * height) / HISTOGRAM_SIZE; //512 * 512 / 256 = 1024
	
	numSubHistograms = globalThreadsHistogram2a / localThreadsHistogram2a;

	//allocate input image

	h_inputImageData = (cl_uchar4*) malloc(width * height * sizeof(cl_uchar4));

	if(h_inputImageData == NULL)
	{
		logMessage(DEBUG_LEVEL_ERROR, "Failed to allocate memory.");
		return -1;
	}

	memcpy(h_inputImageData, inputImage->pixels, width * height * sizeof(cl_uchar4));

	SDL_FreeSurface(inputImage);

	// prevod na grayscale format

	toGrayScale(h_inputImageData, width * height);

	//allocate output image

	h_gpu_outputImageData = (cl_uchar4 *) malloc(width * height * sizeof(cl_uchar4));

	if(h_gpu_outputImageData == NULL)
	{
		logMessage(DEBUG_LEVEL_ERROR, "Failed to allocate memory.");
		return -1;
	}

	memset(h_gpu_outputImageData, 0, width * height * sizeof(cl_uchar4));

	h_cpu_outputImageData = (cl_uchar4 *) malloc(width * height * sizeof(cl_uchar4));

	if(h_cpu_outputImageData == NULL)
	{
		logMessage(DEBUG_LEVEL_ERROR, "Failed to allocate memory.");
		return -1;
	}

	memset(h_cpu_outputImageData, 0, width * height * sizeof(cl_uchar4));

	//allocate cpu histogram

	h_cpu_histogramData = (cl_uint *) malloc(HISTOGRAM_SIZE * sizeof(cl_uint));

	if(h_cpu_histogramData == NULL)
	{
		logMessage(DEBUG_LEVEL_ERROR, "Failed to allocate memory for cpu histogram result data.");
		return -1;
	}

	memset(h_cpu_histogramData, 0, HISTOGRAM_SIZE * sizeof(cl_uint));

	//allocate gpu histogram

	h_gpu_histogramData = (cl_uint *) malloc(HISTOGRAM_SIZE * sizeof(cl_uint));

	if(h_gpu_histogramData == NULL)
	{
		logMessage(DEBUG_LEVEL_ERROR, "Failed to allocate memory for gpu histogram result data.");
		return -1;
	}

	memset(h_gpu_histogramData, 0, HISTOGRAM_SIZE * sizeof(cl_uint));

	//allocate gpu histogram 2

	h_gpu_histogramData2 = (cl_uint *) malloc(numSubHistograms * HISTOGRAM_SIZE * sizeof(cl_uint));

	if(h_gpu_histogramData2 == NULL)
	{
		logMessage(DEBUG_LEVEL_ERROR, "Failed to allocate memory for gpu histogram result data 2.");
		return -1;
	}

	memset(h_gpu_histogramData2, 0, ((width *height) / (HISTOGRAM_SIZE * localThreadsHistogram2a)) * HISTOGRAM_SIZE * sizeof(cl_uint));

	//allocate array for new values after equalization

	h_newValuesData = (cl_uint *) malloc(HISTOGRAM_SIZE * sizeof(cl_uint));

	if(h_newValuesData == NULL)
	{
		logMessage(DEBUG_LEVEL_ERROR, "Failed to allocate memory for equalized histogram result data.");
		return -1;
	}

	memset(h_newValuesData, 0, HISTOGRAM_SIZE * sizeof(cl_uint));

	return 0;
}

/**
 * Initialize host and opencl device
 */
int setupCL()
{
	cl_int ciErr = CL_SUCCESS;

	// Get Platform
	cl_platform_id *cpPlatforms;
	cl_uint cuiPlatformsCount;
	ciErr = clGetPlatformIDs(0, NULL, &cuiPlatformsCount); CheckOpenCLError( ciErr, "clGetPlatformIDs: cuiPlatformsNum=%i", cuiPlatformsCount );
	cpPlatforms = (cl_platform_id*)malloc(cuiPlatformsCount*sizeof(cl_platform_id));
	ciErr = clGetPlatformIDs(cuiPlatformsCount, cpPlatforms, NULL); CheckOpenCLError( ciErr, "clGetPlatformIDs" );
	
	cl_platform_id platform = 0;

	const unsigned int TMP_BUFFER_SIZE = 1024;
	char sTmp[TMP_BUFFER_SIZE];

	for(unsigned int f0=0; f0<cuiPlatformsCount; f0++){
		bool shouldBrake = false;
		ciErr = clGetPlatformInfo (cpPlatforms[f0], CL_PLATFORM_PROFILE, TMP_BUFFER_SIZE, sTmp, NULL);    CheckOpenCLError( ciErr, "clGetPlatformInfo: Id=%i: CL_PLATFORM_PROFILE=%s",f0, sTmp);
		ciErr = clGetPlatformInfo (cpPlatforms[f0], CL_PLATFORM_VERSION, TMP_BUFFER_SIZE, sTmp, NULL);    CheckOpenCLError( ciErr, "clGetPlatformInfo: Id=%i: CL_PLATFORM_VERSION=%s",f0, sTmp);
		ciErr = clGetPlatformInfo (cpPlatforms[f0], CL_PLATFORM_NAME, TMP_BUFFER_SIZE, sTmp, NULL);       CheckOpenCLError( ciErr, "clGetPlatformInfo: Id=%i: CL_PLATFORM_NAME=%s",f0, sTmp);
		ciErr = clGetPlatformInfo (cpPlatforms[f0], CL_PLATFORM_VENDOR, TMP_BUFFER_SIZE, sTmp, NULL);     CheckOpenCLError( ciErr, "clGetPlatformInfo: Id=%i: CL_PLATFORM_VENDOR=%s",f0, sTmp);

		//prioritize AMD and CUDA platforms
		
		if ((strcmp(sTmp, "Advanced Micro Devices, Inc.") == 0) || (strcmp(sTmp, "NVIDIA Corporation") == 0)) {
			platform = cpPlatforms[f0];
		}
		
		//prioritize Intel
		/*if ((strcmp(sTmp, "Intel(R) Corporation") == 0)) {
			platform = cpPlatforms[f0];
		}*/

		ciErr = clGetPlatformInfo (cpPlatforms[f0], CL_PLATFORM_EXTENSIONS, TMP_BUFFER_SIZE, sTmp, NULL); CheckOpenCLError( ciErr, "clGetPlatformInfo: Id=%i: CL_PLATFORM_EXTENSIONS=%s",f0, sTmp);
		printf("\n");
	}

	if(platform == 0)
	{ //no prioritized found
		if(cuiPlatformsCount > 0)
		{ 
			platform = cpPlatforms[0];
		} else {
			logMessage(DEBUG_LEVEL_ERROR, "No device was found");
			return -1;
		}
	}
	// Get Devices
	cl_uint cuiDevicesCount;
	ciErr = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &cuiDevicesCount); CheckOpenCLError( ciErr, "clGetDeviceIDs: cuiDevicesCount=%i", cuiDevicesCount );
	cdDevices = (cl_device_id*)malloc(cuiDevicesCount*sizeof(cl_device_id));
	ciErr = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, cuiDevicesCount, cdDevices, NULL); CheckOpenCLError( ciErr, "clGetDeviceIDs" );
	
	//unsigned int deviceIndex = 0;

	for(unsigned int f0=0; f0<cuiDevicesCount; f0++){
		cl_device_type cdtTmp;
		size_t iDim[3];

		ciErr = clGetDeviceInfo (cdDevices[f0], CL_DEVICE_TYPE, sizeof(cdtTmp), &cdtTmp, NULL);
		CheckOpenCLError( ciErr, "clGetDeviceInfo: Id=%i: CL_DEVICE_TYPE=%s%s%s%s",f0, cdtTmp&CL_DEVICE_TYPE_CPU?"CPU,":"", 
			cdtTmp&CL_DEVICE_TYPE_GPU?"GPU,":"", 
			cdtTmp&CL_DEVICE_TYPE_ACCELERATOR?"ACCELERATOR,":"", 
			cdtTmp&CL_DEVICE_TYPE_DEFAULT?"DEFAULT,":"");

		if(cdtTmp & CL_DEVICE_TYPE_GPU)
		{ //prioritize gpu if both cpu and gpu are available
			deviceIndex = f0;
		}

		cl_bool bTmp;
		ciErr = clGetDeviceInfo (cdDevices[f0], CL_DEVICE_AVAILABLE, sizeof(bTmp), &bTmp, NULL);             CheckOpenCLError( ciErr, "clGetDeviceInfo: Id=%i: CL_DEVICE_AVAILABLE=%s",f0, bTmp?"YES":"NO");
		ciErr = clGetDeviceInfo (cdDevices[f0], CL_DEVICE_NAME, TMP_BUFFER_SIZE, sTmp, NULL);                CheckOpenCLError( ciErr, "clGetDeviceInfo: Id=%i: CL_DEVICE_NAME=%s",f0, sTmp);
		ciErr = clGetDeviceInfo (cdDevices[f0], CL_DEVICE_VENDOR, TMP_BUFFER_SIZE, sTmp, NULL);              CheckOpenCLError( ciErr, "clGetDeviceInfo: Id=%i: CL_DEVICE_VENDOR=%s",f0, sTmp);
		ciErr = clGetDeviceInfo (cdDevices[f0], CL_DRIVER_VERSION, TMP_BUFFER_SIZE, sTmp, NULL);             CheckOpenCLError( ciErr, "clGetDeviceInfo: Id=%i: CL_DRIVER_VERSION=%s",f0, sTmp);
		ciErr = clGetDeviceInfo (cdDevices[f0], CL_DEVICE_PROFILE, TMP_BUFFER_SIZE, sTmp, NULL);             CheckOpenCLError( ciErr, "clGetDeviceInfo: Id=%i: CL_DEVICE_PROFILE=%s",f0, sTmp);
		ciErr = clGetDeviceInfo (cdDevices[f0], CL_DEVICE_VERSION, TMP_BUFFER_SIZE, sTmp, NULL);             CheckOpenCLError( ciErr, "clGetDeviceInfo: Id=%i: CL_DEVICE_VERSION=%s",f0, sTmp);
		ciErr = clGetDeviceInfo (cdDevices[f0], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(iDim), iDim, NULL);    CheckOpenCLError( ciErr, "clGetDeviceInfo: Id=%i: CL_DEVICE_MAX_WORK_ITEM_SIZES=%ix%ix%i",f0, iDim[0], iDim[1], iDim[2]);
		ciErr = clGetDeviceInfo (cdDevices[f0], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), iDim, NULL);  CheckOpenCLError( ciErr, "clGetDeviceInfo: Id=%i: CL_DEVICE_MAX_WORK_GROUP_SIZE=%i",f0, iDim[0]);
		ciErr = clGetDeviceInfo (cdDevices[f0], CL_DEVICE_EXTENSIONS, TMP_BUFFER_SIZE, sTmp, NULL);          CheckOpenCLError( ciErr, "clGetDeviceInfo: Id=%i: CL_DEVICE_EXTENSIONS=%s",f0, sTmp);
		printf("\n");
	}

	cl_context_properties cps[3] = 
    { 
        CL_CONTEXT_PLATFORM, 
        (cl_context_properties)platform, 
        0 
    };

	//create context
	context = clCreateContext(cps, 1, &cdDevices[deviceIndex], NULL, NULL, &ciErr);  CheckOpenCLError( ciErr, "clCreateContext" );
	//may use clCreateContextFromType than choose a device based on the returned devices
	
	//create a command queue
	commandQueue = clCreateCommandQueue(context, cdDevices[deviceIndex], 
			CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &ciErr); 
	CheckOpenCLError( ciErr, "clCreateCommandQueue" );

	//==================================================================================
	//allocate and initialize memory buffers

	//we are only going to read from this
	d_inputImageBuffer = clCreateBuffer(context,
										CL_MEM_READ_ONLY,
										width * height * pixelSize,
										0,
										&ciErr);
	CheckOpenCLError(ciErr, "CreateBuffer inputImage");

	//write our image to the buffer
	// Write Data to inputImageBuffer - blocking write
    ciErr = clEnqueueWriteBuffer(commandQueue,
                                  d_inputImageBuffer,
                                  CL_TRUE, //blocking write
                                  0,
                                  width * height * sizeof(cl_uchar4),
                                  h_inputImageData,
                                  0,
                                  0,
                                  0);

	CheckOpenCLError(ciErr, "Copy input image data");

	//output image buffer - write only
	d_outputImageBuffer = clCreateBuffer(context,
										CL_MEM_WRITE_ONLY,
										width * height * pixelSize,
										0,
										&ciErr);
	CheckOpenCLError(ciErr, "Allocate output buffer");

	//histogram buffer
	d_histogramBuffer = clCreateBuffer(context,
										CL_MEM_READ_WRITE,
										HISTOGRAM_SIZE * sizeof(cl_uint), // Histogram result - unsigned integer values (colors of grey) of occurence
										0, &ciErr);
	CheckOpenCLError(ciErr, "Allocate histogram buffer");

	//histogram buffer 2
	if (histogramMethod == 2)
	{
	    d_subHistogramsBuffer = clCreateBuffer(context,
										CL_MEM_READ_WRITE,
										((width *height) / (HISTOGRAM_SIZE * localThreadsHistogram2a)) * HISTOGRAM_SIZE * sizeof(cl_uint), // Histogram result - unsigned integer values (colors of grey) of occurence
										0, &ciErr);
	    CheckOpenCLError(ciErr, "Allocate subhistograms buffer");
	}

	//eq histogram buffer
	d_newValuesBuffer = clCreateBuffer(context,
										CL_MEM_READ_WRITE,
										HISTOGRAM_SIZE * sizeof(cl_uint), // Histogram result - unsigned integer values (colors of grey) of occurence
										0, &ciErr);
	CheckOpenCLError(ciErr, "Allocate eq histogram buffer");

	//treshhold
	d_threshold  = clCreateBuffer(context,
										CL_MEM_READ_WRITE,
										sizeof(cl_ulong),
										0, &ciErr);
	CheckOpenCLError(ciErr, "Allocate eq treshhold buffer");	
	

	//=================================================================================
	// Create and compile and openCL program

	char *cSourceCL = loadProgSource("kernels.cl");
	
	program = clCreateProgramWithSource(context, 1, (const char **)&cSourceCL, NULL, &ciErr);  CheckOpenCLError( ciErr, "clCreateProgramWithSource" );
	free(cSourceCL);

	ciErr = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	
	cl_int logStatus;

	//build log
    char *buildLog = NULL;
    size_t buildLogSize = 0;
    logStatus = clGetProgramBuildInfo( program, 
										cdDevices[deviceIndex], 
										CL_PROGRAM_BUILD_LOG, 
										buildLogSize, 
										buildLog, 
										&buildLogSize);
	
	CheckOpenCLError(logStatus, "clGetProgramBuildInfo.");

    buildLog = (char*)malloc(buildLogSize);
    if(buildLog == NULL)
    {
        printf("Failed to allocate host memory. (buildLog)");
        return -1;
    }
    memset(buildLog, 0, buildLogSize);

    logStatus = clGetProgramBuildInfo (program, 
										cdDevices[deviceIndex], 
										CL_PROGRAM_BUILD_LOG, 
										buildLogSize, 
										buildLog, 
										NULL);
	CheckOpenCLError(logStatus, "clGetProgramBuildInfo.");

    printf(" \n\t\t\tBUILD LOG\n");
    printf(" ************************************************\n");
    printf("%s", buildLog);
    printf(" ************************************************\n");
    free(buildLog);
	
	CheckOpenCLError( ciErr, "clBuildProgram" );

	//==========================================================================
	// kernels
	equalizeKernel1 = clCreateKernel(program, "equalize1", &ciErr);
	CheckOpenCLError( ciErr, "clCreateKernel equalize1" );
	histogramKernel1 = clCreateKernel(program, "histogram1", &ciErr);
	CheckOpenCLError( ciErr, "clCreateKernel histogram1" );
	histogramKernel2a = clCreateKernel(program, "histogram2a", &ciErr);
	CheckOpenCLError( ciErr, "clCreateKernel histogram2a" );
	histogramKernel2b = clCreateKernel(program, "histogram2b", &ciErr);
	CheckOpenCLError( ciErr, "clCreateKernel histogram2b" );
	equalizeKernel2 = clCreateKernel(program, "equalize2", &ciErr);
	CheckOpenCLError( ciErr, "clCreateKernel equalize2" );
	thresholdKernel = clCreateKernel(program, "threshold", &ciErr);
	CheckOpenCLError( ciErr, "clCreateKernel threshold" );
	thresholdingKernel = clCreateKernel(program, "thresholding", &ciErr);
	CheckOpenCLError( ciErr, "clCreateKernel thresholding" );
    segKernel = clCreateKernel(program, "segmentation", &ciErr);
	CheckOpenCLError( ciErr, "clCreateKernel segmentation" );

	return 0;
}

int checkWorkgroupSize(cl_kernel &pkernel, size_t &blockSizeX, size_t &blockSizeY)
{
	cl_int ciErr = CL_SUCCESS;
	size_t maxKernelWorkGroupSize;
	ciErr = clGetKernelWorkGroupInfo(pkernel,
									cdDevices[deviceIndex], //this only workes if single device
									CL_KERNEL_WORK_GROUP_SIZE,
									sizeof(size_t),
									&maxKernelWorkGroupSize,
									0);
	CheckOpenCLError(ciErr, "clGetKernelInfo %u", maxKernelWorkGroupSize);
   
	if((blockSizeX * blockSizeY) > maxKernelWorkGroupSize)
    {
        printf("Out of Resources!\n");
        printf("Group Size specified: %i\n", blockSizeX * blockSizeY);
        printf("Max Group Size supported on the kernel: %i\n", maxKernelWorkGroupSize);
        printf("Falling back to %i.\n", maxKernelWorkGroupSize);

        if(blockSizeX > maxKernelWorkGroupSize)
        {
            blockSizeX = maxKernelWorkGroupSize;
            blockSizeY = 1;
			return EXIT_FAILURE;
        }
    }
	
	return EXIT_SUCCESS;	
}

void runGpuHistogram1() {
	int status;

	/* Setup arguments to the kernel */

    /* input buffer */
    status = clSetKernelArg(histogramKernel1, 
                            0, 
                            sizeof(cl_mem), 
                            &d_inputImageBuffer);
	CheckOpenCLError(status, "clSetKernelArg. (inputImage)");

    /* image width */
    status = clSetKernelArg(histogramKernel1, 
                            1, 
                            sizeof(cl_uint), 
                            &width);

	CheckOpenCLError(status, "clSetKernelArg. (width)");

	/* image height */
    status = clSetKernelArg(histogramKernel1, 
                            2, 
                            sizeof(cl_uint), 
                            &height);

	CheckOpenCLError(status, "clSetKernelArg. (height)");
    
	/* histogram buffer */
	status = clSetKernelArg(histogramKernel1, 
                            3, 
                            sizeof(cl_mem), 
                            &d_histogramBuffer);
	
	CheckOpenCLError(status, "clSetKernelArg. (histogram)");

    /* cache */
	status = clSetKernelArg(histogramKernel1, 
	                        4, 
							HISTOGRAM_SIZE * sizeof(cl_uint),
	                        0);

	CheckOpenCLError(status, "clSetKernelArg. (cache) %u", sizeof(cl_uint)*HISTOGRAM_SIZE);

	//the global number of threads in each dimension has to be divisible
	// by the local dimension numbers
	//size_t globalThreadsHistogram[] = { ((height + blockSizeY - 1)/blockSizeY) * blockSizeY };
    //size_t localThreadsHistogram[] = { blockSizeY };
	size_t blockSizeX = 16;
	size_t blockSizeY = 1;

	checkWorkgroupSize(histogramKernel1, blockSizeX, blockSizeY);

	size_t globalThreadsHistogram[] = 
	{
		((width + blockSizeX - 1)/blockSizeX) * blockSizeX,
		((height + blockSizeY - 1)/blockSizeY) * blockSizeY
	};
	size_t localThreadsHistogram[] = {blockSizeX, blockSizeY};

    status = clEnqueueNDRangeKernel(commandQueue,
                                    histogramKernel1,
                                    2, // Dimensions
                                    NULL, //offset
                                    globalThreadsHistogram,
                                    localThreadsHistogram,
                                    0,
                                    NULL,
                                    &event_histogram1);
    CheckOpenCLError(status, "clEnqueueNDRangeKernel.");

    status = clWaitForEvents(1, &event_histogram1);
    CheckOpenCLError(status, "clWaitForEvents.");

	//Read back the histogram
	//blocking read

	status = clEnqueueReadBuffer(commandQueue,
                                d_histogramBuffer,
                                CL_TRUE,
                                0,
								HISTOGRAM_SIZE * sizeof(cl_uint),
                                h_gpu_histogramData,
                                0,
                                0,
                                0);
		
   CheckOpenCLError(status, "read histogram.");

   printTiming(event_histogram1, "GPU Histogram 1: ");

   return;
}

void runGpuHistogram2() {
	int status;

//////////////KERNEL 2A////////////////////////////////////////////////////////////////////////////////////////

	/* Setup arguments to the kernel */

    /* input buffer */
    status = clSetKernelArg(histogramKernel2a, 
                            0, 
                            sizeof(cl_mem), 
                            &d_inputImageBuffer);
	CheckOpenCLError(status, "clSetKernelArg. (inputImage)");

    /* image width */
    status = clSetKernelArg(histogramKernel2a, 
                            3, 
                            sizeof(cl_uint), 
                            &width);

	CheckOpenCLError(status, "clSetKernelArg. (width)");

	/* image height */
    status = clSetKernelArg(histogramKernel2a, 
                            4, 
                            sizeof(cl_uint), 
                            &height);

	CheckOpenCLError(status, "clSetKernelArg. (height)");

	/* shared array buffer */
    status = clSetKernelArg(histogramKernel2a, 
                            1, 
                            localThreadsHistogram2a * HISTOGRAM_SIZE * sizeof(cl_uchar), 
                            0);
	CheckOpenCLError(status, "clSetKernelArg. (sharedArray)");
    
	/* subhistograms buffer */
	status = clSetKernelArg(histogramKernel2a, 
                            2, 
                            sizeof(cl_mem), 
                            &d_subHistogramsBuffer);
	
	CheckOpenCLError(status, "clSetKernelArg. (subHistograms)");

	//the global number of threads in each dimension has to be divisible
	// by the local dimension numbers
	size_t blockSizeX = localThreadsHistogram2a;
	size_t blockSizeY = 1;

	checkWorkgroupSize(histogramKernel2a, blockSizeX, blockSizeY);

	size_t globalThreads[] = 
	{
		globalThreadsHistogram2a
	};
	size_t localThreads[] = {localThreadsHistogram2a};

    status = clEnqueueNDRangeKernel(commandQueue,
                                    histogramKernel2a,
									1,
                                    NULL, //offset
                                    globalThreads,
                                    localThreads,
                                    0,
                                    NULL,
                                    &event_histogram2);
    CheckOpenCLError(status, "clEnqueueNDRangeKernel.");

    status = clWaitForEvents(1, &event_histogram2);
    CheckOpenCLError(status, "clWaitForEvents.");

   printTiming(event_histogram2, "GPU Histogram 2a: ");

//////////////KERNEL 2B////////////////////////////////////////////////////////////////////////////////////////

   /* Setup arguments to the kernel */

    /* subhistograms */
    status = clSetKernelArg(histogramKernel2b, 
                            0, 
                            sizeof(cl_mem), 
                            &d_subHistogramsBuffer);
	CheckOpenCLError(status, "clSetKernelArg. (subHistograms)");

    /* output histogram */
    status = clSetKernelArg(histogramKernel2b, 
                            1, 
                            sizeof(cl_mem), 
                            &d_histogramBuffer);

	CheckOpenCLError(status, "clSetKernelArg. (histogram)");

	/* number of subhistograms */
    status = clSetKernelArg(histogramKernel2b, 
                            2, 
                            sizeof(cl_uint), 
                            &numSubHistograms);

	CheckOpenCLError(status, "clSetKernelArg. (numSubHistograms)");

	//the global number of threads in each dimension has to be divisible
	// by the local dimension numbers
	blockSizeX = 256;
	blockSizeY = 1;

	checkWorkgroupSize(histogramKernel2b, blockSizeX, blockSizeY);

	size_t globalThreadsHistogram2b[] = 
	{
		256
	};
	size_t localThreadsHistogram2b[] = {256};

	cl_event histogram2b_wait_events[] = { event_histogram2 };

    status = clEnqueueNDRangeKernel(commandQueue,
                                    histogramKernel2b,
									1,
                                    NULL, //offset
                                    globalThreadsHistogram2b,
                                    localThreadsHistogram2b,
                                    1,
                                    histogram2b_wait_events,
                                    &event_histogram1);
    CheckOpenCLError(status, "clEnqueueNDRangeKernel.");

    status = clWaitForEvents(1, &event_histogram1);
    CheckOpenCLError(status, "clWaitForEvents.");
	
	//Read back the histogram
	//blocking read

	status = clEnqueueReadBuffer(commandQueue,
                                d_histogramBuffer,
                                CL_TRUE,
                                0,
								HISTOGRAM_SIZE * sizeof(cl_uint),
                                h_gpu_histogramData,
                                0,
                                0,
                                0);
		
   CheckOpenCLError(status, "read histogram.");

   printTiming(event_histogram1, "GPU Histogram 2b: ");

   return;
}

void runCpuHistogram() 
{
	printf("Running CPU histogram implementation.\n");
	volatile double t1 = getTime();
	histogram(h_inputImageData, h_cpu_histogramData, width, height);
	volatile double t2 = getTime();
    double elapsedTime = (t2 - t1) * 1000.0f;
    printf("CPU histogram:  elapsedTime %.3lf ms\n", elapsedTime);
}

void runCpuEqualize() 
{
	printf("Running CPU equalization implementation.\n");
	volatile double t1 = getTime();
	equalize(h_inputImageData, h_cpu_outputImageData, h_gpu_histogramData);
	volatile double t2 = getTime();
    double elapsedTime = (t2 - t1) * 1000.0f;
    printf("CPU equalize:  elapsedTime %.3lf ms\n", elapsedTime);
}

void runGpuEqualization1() {
	int status;

	/* histogram buffer */
	status = clSetKernelArg(equalizeKernel1, 
	                        0, 
	                        sizeof(cl_mem), 
	                        &d_histogramBuffer);
	CheckOpenCLError(status, "clSetKernelArg. (histogram)");

	/* output buffer */
	status = clSetKernelArg(equalizeKernel1, 
	                        1, 
	                        sizeof(cl_mem), 
	                        &d_newValuesBuffer);
	CheckOpenCLError(status, "clSetKernelArg. (output)");

	//the global number of threads in each dimension has to be divisible
	// by the local dimension numbers
	size_t blockSizeX = 1;
	size_t blockSizeY = 1;

	checkWorkgroupSize(histogramKernel1, blockSizeX, blockSizeY);

	size_t globalThreadsEqualize1[] = 
	{
		256
	};
	size_t localThreadsEqualize1[] = { 256 };

	cl_event equalize_wait_events[] = { event_histogram1 };

	status = clEnqueueNDRangeKernel(commandQueue,
		equalizeKernel1,
		1, // Dimensions
		NULL, //offset
		globalThreadsEqualize1,
		localThreadsEqualize1,
		1, //num events in wait list
		equalize_wait_events,
		&event_equalize1);

	CheckOpenCLError(status, "clEnqueueNDRangeKernel.");

	status = clWaitForEvents(1, &event_equalize1);
	CheckOpenCLError(status, "clWaitForEvents.");

	//Read back the image - if textures were used for showing this wouldn't be necessary
	//blocking read
	status = clEnqueueReadBuffer(commandQueue,
	                            d_newValuesBuffer,
                                CL_TRUE,
                                0,
                                HISTOGRAM_SIZE * sizeof(cl_uint),
                                h_newValuesData,
                                0,
                                0,
                                0);
	
    CheckOpenCLError(status, "read equalize1 output.");

	printTiming(event_equalize1, "GPU Equalize1: ");

	return;
}

void runGpuEqualization2() {
	int status;

	/* Setup arguments to the kernel */

    /* input buffer */
    status = clSetKernelArg(equalizeKernel2, 
                            0, 
                            sizeof(cl_mem), 
                            &d_inputImageBuffer);
	CheckOpenCLError(status, "clSetKernelArg. (inputImage)");

	/* output buffer */
	status = clSetKernelArg(equalizeKernel2, 
	                        1, 
	                        sizeof(cl_mem), 
	                        &d_outputImageBuffer);
	CheckOpenCLError(status, "clSetKernelArg. (outputImage)");

	/* histogram buffer */
	status = clSetKernelArg(equalizeKernel2, 
                            2, 
                            sizeof(cl_mem), 
                            &d_newValuesBuffer);
	
	CheckOpenCLError(status, "clSetKernelArg. (histogram)");

    /* image width */
    status = clSetKernelArg(equalizeKernel2, 
                            3, 
                            sizeof(cl_uint), 
                            &width);

	CheckOpenCLError(status, "clSetKernelArg. (width)");

	/* image height */
    status = clSetKernelArg(equalizeKernel2, 
                            4, 
                            sizeof(cl_uint), 
                            &height);

	CheckOpenCLError(status, "clSetKernelArg. (height)");

	//the global number of threads in each dimension has to be divisible
	// by the local dimension numbers
	size_t blockSizeX = 16;
	size_t blockSizeY = 16;

	checkWorkgroupSize(histogramKernel1, blockSizeX, blockSizeY);

	size_t globalThreadsEqualize2[] = 
	{
		((width + blockSizeX - 1)/blockSizeX) * blockSizeX,
		((height + blockSizeY - 1)/blockSizeY) * blockSizeY
	};
	size_t localThreadsEqualize2[] = {blockSizeX, blockSizeY};

	cl_event equalize2_wait_events[] = { event_equalize1 };

    status = clEnqueueNDRangeKernel(commandQueue,
                                    equalizeKernel2,
                                    2, // Dimensions
                                    NULL, //offset
                                    globalThreadsEqualize2,
                                    localThreadsEqualize2,
                                    1,
                                    equalize2_wait_events,
                                    &event_equalize2);
    CheckOpenCLError(status, "clEnqueueNDRangeKernel.");

    status = clWaitForEvents(1, &event_equalize2);
    CheckOpenCLError(status, "clWaitForEvents.");

	//Read back the histogram
	//blocking read

	status = clEnqueueReadBuffer(commandQueue,
                                d_outputImageBuffer,
                                CL_TRUE,
                                0,
								width * height * sizeof(cl_uchar4),
                                h_gpu_outputImageData,
                                0,
                                0,
                                0);
		
   CheckOpenCLError(status, "read output.");

   printTiming(event_equalize2, "GPU Equalize2: ");

   return;
}

void runCpuOtsu() 
{
	printf("Running CPU otsu implementation.\n");
	volatile double t1 = getTime();
	otsu(h_inputImageData, h_cpu_outputImageData, h_gpu_histogramData, width, height);
	volatile double t2 = getTime();
    double elapsedTime = (t2 - t1) * 1000.0f;
    printf("CPU otsu:  elapsedTime %.3lf ms\n", elapsedTime);
}

void runGpuOtsu() 
{
	int status;

	cl_ulong h_threshold[1];

	/* Setup arguments to the kernel */

	/* histogram buffer */
	status = clSetKernelArg(thresholdKernel, 
                            0, 
                            sizeof(cl_mem), 
                            &d_histogramBuffer);
	
	CheckOpenCLError(status, "clSetKernelArg. (histogram)");


	/* output buffer */
	status = clSetKernelArg(thresholdKernel, 
	                        1, 
	                        sizeof(cl_mem), 
	                        &d_threshold);
	CheckOpenCLError(status, "clSetKernelArg. (threshold)");
	

	//the global number of threads in each dimension has to be divisible
	// by the local dimension numbers

	size_t blockSizeX = 1;
	size_t blockSizeY = 1;


	checkWorkgroupSize(thresholdKernel, blockSizeX, blockSizeY);

	size_t globalThreadsThreshold[] = { 16 };
	size_t localThreadsThreshold[] = { 16 };

	cl_event threshold_wait_events[] = { event_histogram1 };

	status = clEnqueueNDRangeKernel(commandQueue,
									thresholdKernel,
									1, // Dimensions
									NULL, //offset
									globalThreadsThreshold,
									localThreadsThreshold,
									1, //num events in wait list
									threshold_wait_events,
									&event_threshold);

	CheckOpenCLError(status, "clEnqueueNDRangeKernel.");

	status = clWaitForEvents(1, &event_threshold);
	CheckOpenCLError(status, "clWaitForEvents.");

	//Read back the image - if textures were used for showing this wouldn't be necessary
	//blocking read
	status = clEnqueueReadBuffer(commandQueue,
	                            d_threshold,
                                CL_TRUE,
                                0,
                                sizeof(cl_ulong),
                                h_threshold,
                                0,
                                0,
                                0);
	
    CheckOpenCLError(status, "read threshold output.");

	printTiming(event_threshold, "GPU threshold: ");

	// thresholding 

	/* Setup arguments to the kernel */

    /* input buffer */
    status = clSetKernelArg(thresholdingKernel, 
                            0, 
                            sizeof(cl_mem), 
                            &d_inputImageBuffer);
	CheckOpenCLError(status, "clSetKernelArg. (inputImage)");

	/* output buffer */
	status = clSetKernelArg(thresholdingKernel, 
	                        1, 
	                        sizeof(cl_mem), 
	                        &d_outputImageBuffer);
	CheckOpenCLError(status, "clSetKernelArg. (outputImage)");

	/* image threshold */
    status = clSetKernelArg(thresholdingKernel, 
                            2, 
                            sizeof(cl_mem), 
                            &d_threshold);

	CheckOpenCLError(status, "clSetKernelArg. (threshold)");

    /* image width */
    status = clSetKernelArg(thresholdingKernel, 
                            3, 
                            sizeof(cl_uint), 
                            &width);

	CheckOpenCLError(status, "clSetKernelArg. (width)");

	/* image height */
    status = clSetKernelArg(thresholdingKernel, 
                            4, 
                            sizeof(cl_uint), 
                            &height);

	CheckOpenCLError(status, "clSetKernelArg. (height)");

	//the global number of threads in each dimension has to be divisible
	// by the local dimension numbers
	blockSizeX = 16;
	blockSizeY = 16;

	checkWorkgroupSize(thresholdingKernel, blockSizeX, blockSizeY);

	size_t globalThreadsthresholding[] = 
	{
		((width + blockSizeX - 1)/blockSizeX) * blockSizeX,
		((height + blockSizeY - 1)/blockSizeY) * blockSizeY
	};
	size_t localThreadsthresholding[] = {blockSizeX, blockSizeY};

	cl_event thresholding_wait_events[] = { event_threshold };

    status = clEnqueueNDRangeKernel(commandQueue,
                                    thresholdingKernel,
                                    2, // Dimensions
                                    NULL, //offset
                                    globalThreadsthresholding,
                                    localThreadsthresholding,
                                    1,
                                    thresholding_wait_events,
                                    &event_thresholding);
    CheckOpenCLError(status, "clEnqueueNDRangeKernel.");

    status = clWaitForEvents(1, &event_thresholding);
    CheckOpenCLError(status, "clWaitForEvents.");

	//Read back the histogram
	//blocking read

	status = clEnqueueReadBuffer(commandQueue,
                                d_outputImageBuffer,
                                CL_TRUE,
                                0,
								width * height * sizeof(cl_uchar4),
                                h_gpu_outputImageData,
                                0,
                                0,
                                0);
		
   CheckOpenCLError(status, "read output.");

   printTiming(event_thresholding, "GPU thresholding: ");

   return;
}

void runCpuSeg() 
{
	printf("Running CPU segmentation implementation.\n");
	volatile double t1 = getTime();
    segmentation(h_inputImageData, h_cpu_outputImageData, width, height);
	volatile double t2 = getTime();
    double elapsedTime = (t2 - t1) * 1000.0f;
    printf("CPU segmentation:  elapsedTime %.3lf ms\n", elapsedTime);
}

void runGpuSeg() 
{
	int status;
    
	/* Setup arguments to the kernel */

    /* input buffer */
    status = clSetKernelArg(segKernel, 
                            0, 
                            sizeof(cl_mem), 
                            &d_inputImageBuffer);
	CheckOpenCLError(status, "clSetKernelArg. (inputImage)");

	/* output buffer */
	status = clSetKernelArg(segKernel, 
	                        1, 
	                        sizeof(cl_mem), 
	                        &d_outputImageBuffer);
	CheckOpenCLError(status, "clSetKernelArg. (outputImage)");
    
    /* image width */
    status = clSetKernelArg(segKernel, 
                            2, 
                            sizeof(cl_uint), 
                            &width);

	CheckOpenCLError(status, "clSetKernelArg. (width)");

	/* image height */
    status = clSetKernelArg(segKernel, 
                            3, 
                            sizeof(cl_uint), 
                            &height);

	CheckOpenCLError(status, "clSetKernelArg. (height)");

	//the global number of threads in each dimension has to be divisible
	// by the local dimension numbers
	size_t blockSizeX = 16;
	size_t blockSizeY = 32;

	checkWorkgroupSize(segKernel, blockSizeX, blockSizeY);

	size_t globalSize[] = 
	{
		((width + blockSizeX - 1)/blockSizeX) * blockSizeX,
		((height + blockSizeY - 1)/blockSizeY) * blockSizeY
	};
	size_t localSize[] = {blockSizeX, blockSizeY};

	cl_event wait_events[] = { event_seg };

    status = clEnqueueNDRangeKernel(commandQueue,
                                    segKernel,
                                    2, // Dimensions
                                    NULL, //offset
                                    globalSize,
                                    localSize,
                                    0,
                                    NULL,
                                    &event_seg);
    CheckOpenCLError(status, "clEnqueueNDRangeKernel.");

    status = clWaitForEvents(1, &event_seg);
    CheckOpenCLError(status, "clWaitForEvents.");

	//Read back the histogram
	//blocking read

	status = clEnqueueReadBuffer(commandQueue,
                                d_outputImageBuffer,
                                CL_TRUE,
                                0,
								width * height * sizeof(cl_uchar4),
                                h_gpu_outputImageData,
                                0,
                                0,
                                0);
		
   CheckOpenCLError(status, "read output.");

   printTiming(event_seg, "GPU segmentation: ");

   return;
}

int cleanup()
{
	/* Releases OpenCL resources (Context, Memory etc.) */
    cl_int status;

    status = clReleaseKernel(histogramKernel1);
    CheckOpenCLError(status, "clReleaseKernel histogram1.");

	status = clReleaseKernel(histogramKernel2a);
    CheckOpenCLError(status, "clReleaseKernel histogram2a.");

	status = clReleaseKernel(histogramKernel2b);
    CheckOpenCLError(status, "clReleaseKernel histogram2b.");

	status = clReleaseKernel(equalizeKernel1);
	CheckOpenCLError(status, "clReleaseKernel equalize1.");

	status = clReleaseKernel(equalizeKernel2);
	CheckOpenCLError(status, "clReleaseKernel equalize2.");

	status = clReleaseKernel(thresholdKernel);
	CheckOpenCLError(status, "clReleaseKernel threshold.");

	status = clReleaseKernel(thresholdingKernel);
	CheckOpenCLError(status, "clReleaseKernel thresholding.");

    status = clReleaseProgram(program);
    CheckOpenCLError(status, "clReleaseProgram.");

    status = clReleaseMemObject(d_inputImageBuffer);
    CheckOpenCLError(status, "clReleaseMemObject input");
    
    status = clReleaseMemObject(d_histogramBuffer);
    CheckOpenCLError(status, "clReleaseMemObject histogram");

	if (histogramMethod == 2)
	{
	    status = clReleaseMemObject(d_subHistogramsBuffer);
        CheckOpenCLError(status, "clReleaseMemObject histogram 2");
	}
	
    status = clReleaseMemObject(d_outputImageBuffer);
    CheckOpenCLError(status, "clReleaseMemObject output");

	status = clReleaseMemObject(d_threshold);
    CheckOpenCLError(status, "clReleaseMemObject threshold");

    status = clReleaseCommandQueue(commandQueue);
    CheckOpenCLError(status, "clReleaseCommandQueue.");

    status = clReleaseContext(context);
    CheckOpenCLError(status, "clReleaseContext.");

    /* release program resources (input memory etc.) */
    if(h_inputImageData) 
        free(h_inputImageData);

    if(h_gpu_outputImageData)
        free(h_gpu_outputImageData);

	if(h_cpu_outputImageData)
        free(h_cpu_outputImageData);

	if(h_cpu_histogramData)
        free(h_cpu_histogramData);

	if(h_gpu_histogramData)
        free(h_gpu_histogramData);

	if(h_gpu_histogramData2)
        free(h_gpu_histogramData2);

	if(h_newValuesData)
        free(h_newValuesData);

    return 0;
}

int main(int argc, char* argv[])
{
	if(argc != 4) {
		cout << "Usage: gmu.exe <metoda histogramu> <metoda> <cesta k obrazku>\n";
		cout << "  <metoda histogramu> - Moznosti: hist1, hist2\n";
		cout << "  <metoda> - Moznosti: equalize, otsu, segmentation\n";

		return 1;
	}

	if(!strcmp(argv[1], "hist1"))
	{
		histogramMethod = 1;
	}
    else if(!strcmp(argv[1], "hist2"))
	{
        histogramMethod = 2;
	}
	else
	{
		cout << "Usage: gmu.exe <metoda histogramu> <metoda> <cesta k obrazku>\n";
		cout << "  <metoda histogramu> - Moznosti: hist1, hist2\n";
		cout << "  <metoda> - Moznosti: equalize, otsu, segmentation\n";

		return 1;
	}

	if (!strcmp(argv[2], "equalize"))
	{
		method = EQUALIZE;
	}
	else if(!strcmp(argv[2], "otsu"))
	{
		method = OTSU;
	}
    else if(!strcmp(argv[2], "segmentation"))
	{
        method = SEGMENTATION;
	}
	else
	{
		cout << "Usage: gmu.exe <metoda histogramu> <metoda> <cesta k obrazku>\n";
		cout << "  <metoda histogramu> - Moznosti: hist1, hist2\n";
		cout << "  <metoda> - Moznosti: equalize, otsu, segmentation\n";

		return 1;
	}

	// Init SDL - only video subsystem will be used
    if(SDL_Init(SDL_INIT_VIDEO) < 0) throw SDL_Exception();
    // Shutdown SDL when program ends
    atexit(SDL_Quit);

#if SDL_IMAGE_PATCHLEVEL >= 10
	// load support for the JPG and PNG image formats
	int flags=IMG_INIT_JPG|IMG_INIT_PNG;
	int initted=IMG_Init(flags);
	if((initted&flags) != flags) {
		logMessage(DEBUG_LEVEL_ERROR, "IMG_Init: Failed to init required jpg and png support!");
		logMessage(DEBUG_LEVEL_ERROR, IMG_GetError());
		throw SDL_Exception();
	}
	atexit(IMG_Quit);
#endif

	//load image
	if(setupHost(argv[3]) != 0)
	{
		cleanup();
		return 1;
	}

	screen = initScreen(width, height, 24);

	mainLoop(screen);
	
	cleanup();

	return 0;
}

void compareResults()
{
	printf("Comparing gpu and cpu histogram:\n");
	for (int i = 0; i < HISTOGRAM_SIZE; i++) 
	{
        if (h_cpu_histogramData[i] != h_gpu_histogramData[i]) 
		{
            printf("GPU and CPU histogams are different!\n");
            break;
        }
		if (i == HISTOGRAM_SIZE - 1) printf("GPU and CPU histograms are the same!\n");
    }

	printf("Comparing gpu and cpu output:\n");
	for (int i = 0; i < width * height; i++) 
	{
        if (h_cpu_outputImageData[i].s[0] != h_gpu_outputImageData[i].s[0]) 
		{
            printf("GPU and CPU outputs are different!\n");
            break;
        }
		if (i == width * height - 1) printf("GPU and CPU outputs are the same!\n");
    }
}

/**
 * Called after context was created
 */
void onInit()
{
	if(setupCL() != 0)
		return;
  
	

	switch (method)
	{
	case EQUALIZE:
        runCpuHistogram();
		if (histogramMethod == 1)
	        runGpuHistogram1();
		else if (histogramMethod == 2)
			runGpuHistogram2();
		runCpuEqualize();
	    runGpuEqualization1();
	    runGpuEqualization2();
        compareResults();
		break;
	case OTSU:
        runCpuHistogram();
	    if (histogramMethod == 1)
	        runGpuHistogram1();
		else if (histogramMethod == 2)
			runGpuHistogram2();
		runCpuOtsu();
		runGpuOtsu();
        compareResults();
		break;
    case SEGMENTATION:
		runCpuSeg();
		runGpuSeg();
		break;
	default:
		break;
	}

	
}

/**
 * Called when the window should be redrawn
 */
void onWindowRedraw()
{
	drawOutputImage(screen);
    SDL_UpdateRect(screen, 0, 0, 0, 0);
}

/**
 * Called when the window was resized
 * @param width The new width
 * @param height The new height
 */
void onWindowResized(int width, int height)
{
    onWindowRedraw();
}

/**
 * Called when the key was pressed
 * @param key The key that was pressed
 * @param mod Modifiers
 */
void onKeyDown(SDLKey key, SDLMod mod)
{}

/**
 * Called when the key was released
 * @param key The key that was released
 * @param mod Modifiers
 */
void onKeyUp(SDLKey key, SDLMod mod)
{}

/**
 * Called when the mouse moves over the window
 * @param x The new x position
 * @param y The new y position
 * @param xrel Relative move from last x position
 * @param yrel Relative move from last y position
 * @param buttons Mask of the buttons that are pressed
 */
void onMouseMove(unsigned x, unsigned y, int xrel, int yrel, Uint8 buttons)
{}

/**
 * Called when a mouse button was pressed
 * @param button The button that was pressed
 * @param x The x position where the mouse was clicked
 * @param y The y position where the mouse was clicked
 */
void onMouseDown(Uint8 button, unsigned x, unsigned y)
{}

/**
 * Called when a mouse button was released
 * @param button The button that was released
 * @param x The x position where the mouse was clicked
 * @param y The y position where the mouse was clicked
 */
void onMouseUp(Uint8 button, unsigned x, unsigned y)
{}
