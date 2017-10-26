//Author:Sanjana Kalyanapagol
// Description: Image Procesing Algorithms
//Date: 06th May, 2017

// Includes: system
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdint.h>
#include <errno.h>
#include <assert.h>
#include <string.h>
#include <sys/io.h>

#include <cutil_inline.h>

// Includes: local
#include "bmp.h"


enum {GRAYSCALE=1, INVERT};

#define CLAMP_8bit(x) max(0, min(255, (x)))
// Encode a 32-bit unsigned RGBA value from individual
// red, green, blue, and alpha component values.
#define RGBA(r,g,b,a) ((((a) << 24)) | (((b) << 16)) | (((g) << 8)) | ((r)))


char *BMPInFile = "lena.bmp";
char *BMPOutFile = "output.bmp";
char *Filter = "sobel";
int FilterMode  = SWAP;

// Functions
void Cleanup(void);
void ParseArguments(int, char**);
void FilterWrapper(uint32_t* pImageIn, int Width, int Height);

// Kernels

__global__ void GrayscaleFilter(uint32_t *g_DataIn, uint32_t *g_DataOut, int width, int height);


__global__ void InvertFilter(uint32_t *g_DataIn, uint32_t *g_DataOut, int width, int height);


/* Device Memory */
uint32_t *d_In;
uint32_t *d_Out;

// Setup for kernel size
const int TILE_WIDTH    = 6;
const int TILE_HEIGHT   = 6;

const int FILTER_RADIUS = 0;
//  const int FILTER_RADIUS = 3;

const int FILTER_DIAMETER = 2 * FILTER_RADIUS + 1;
const int FILTER_AREA   = FILTER_DIAMETER * FILTER_DIAMETER;

const int BLOCK_WIDTH   = TILE_WIDTH + 2*FILTER_RADIUS;
const int BLOCK_HEIGHT  = TILE_HEIGHT + 2*FILTER_RADIUS;

const int EDGE_VALUE_THRESHOLD = 70;
const int HIGH_BOOST_FACTOR = 4;

unsigned int timer_MemCpyHostToDevice;
unsigned int timer_MemCpyDeviceToHost;
unsigned int kernel_timer;
unsigned int CPU_timer;
unsigned int total_timer;

unsigned int rc = 0, gc = 0, bc = 0;
uint32_t g_av = 0;
uint8_t cnt = 0;



#include "filter_kernel.cu"

void BitMapRead(char *file, struct bmp_header *bmp, struct dib_header *dib, uint32_t **data, uint32_t **palete)
{
    size_t palete_size;
    int fd;
    
    if((fd = open(file, O_RDONLY )) < 0)
        FATAL("Open Source");
    
    if(read(fd, bmp, BMP_SIZE) != BMP_SIZE)
        FATAL("Read BMP Header");
    
    if(read(fd, dib, DIB_SIZE) != DIB_SIZE)
        FATAL("Read DIB Header");
    printf("\n bpp is %d ", dib->bpp);
    
    assert(dib->bpp == 32);
    
    palete_size = bmp->offset - BMP_SIZE - DIB_SIZE;
    if(palete_size > 0) {
        *palete = (uint32_t *)malloc(palete_size);
        int go = read(fd, *palete, palete_size);
        if (go != palete_size) {
            FATAL("Read Palete");
        }
        
        
    }
    
    *data = (uint32_t *)malloc(dib->image_size);
    if(read(fd, *data, dib->image_size) != dib->image_size)
        FATAL("Read Image");
    
    close(fd);
}


void BitMapWrite(char *file, struct bmp_header *bmp, struct dib_header *dib, uint32_t *data, uint32_t *palete)
{
    size_t palete_size;
    int fd;
    
    palete_size = bmp->offset - BMP_SIZE - DIB_SIZE;
    
    if((fd = open(file, O_WRONLY | O_CREAT | O_TRUNC,
                  S_IRUSR | S_IWUSR |S_IRGRP)) < 0)
        FATAL("Open Destination");
    
    if(write(fd, bmp, BMP_SIZE) != BMP_SIZE)
        FATAL("Write BMP Header");
    
    if(write(fd, dib, DIB_SIZE) != DIB_SIZE)
        FATAL("Write BMP Header");
    
    if(palete_size != 0) {
        if(write(fd, palete, palete_size) != palete_size)
            FATAL("Write Palete");
    }
    if(write(fd, data, dib->image_size) != dib->image_size)
        FATAL("Write Image");
    close(fd);
    printf("\nDone Write");
}



void CPU_Grayscale(uint32_t* imageIn, uint32_t * imageOut, int width, int height)
{
    printf("\nHeight:%d Width %d", height, width);
    for(int i = 0; i < ((height) * (width)); i++)
    {
        uint32_t al = (imageIn[i] & 0xFF000000) >>24;
        uint32_t r = (imageIn[i] & 0x000000FF);
        uint32_t g = (imageIn[i] & 0x0000FF00) >> 8;
        uint32_t b = (imageIn[i] & 0x00FF0000) >> 16;
        
        //uint32_t avg =(uint32_t) ((0.3*r) + (0.3*g) + (0.3*b))>>24;//AVERAGE
        uint32_t avg = (uint8_t)((r+g+b)/3);
        imageOut[i] = (al << 24) | (avg << 16) |(avg << 8) |(avg);
    }
}

uint8_t checkRGB(uint32_t pixel)
{
    uint32_t r = (pixel & 0x000000FF);
    uint32_t g = (pixel & 0x0000FF00) >> 8;
    uint32_t b = (pixel & 0x00FF0000) >> 16;
    
    uint8_t av = (r+g+b)/3;
    
    if(av > g_av)
    {
        g_av = av;
        cnt++;
        rc+= r;//return 'r';
        gc+=g;//return 'g';
        bc+=b;//return 'b';
    }
}




void CPU_Invert(uint32_t* imageIn, uint32_t * imageOut, int width, int height)
{
    printf("\nHeight:%d Width %d In invert Filter", height, width);
    for(int i = 0; i < ((height) * (width)); i++)
    {
        unsigned char al = ((imageIn[i]) >>24);
        unsigned char g = (((imageIn[i]) >> 16) & 0xFF);
        unsigned char r = (((imageIn[i]) >> 8)  & 0xFF);
        unsigned char b = ((imageIn[i]) & 0xFF);
        r = 255-r;
        g = 255-g;
        b = 255-b;
        uint32_t OutPixel = RGBA(b,r,g,al);
        imageOut[i] = OutPixel;
    }
    
}






// Host code
int main(int argc, char** argv)
{
    ParseArguments(argc, argv);
    
    struct bmp_header bmp;
    struct dib_header dib;
    
    // Initializing the timer //
    cutCreateTimer (& kernel_timer);
    cutCreateTimer (& timer_MemCpyHostToDevice);
    cutCreateTimer (& timer_MemCpyDeviceToHost);
    cutCreateTimer (& CPU_timer);
    cutCreateTimer (& total_timer);
    
    uint32_t *palete = NULL;
    uint32_t *data = NULL, *out = NULL;
    
    printf("Running %s filter\n", Filter);
    BitMapRead(BMPInFile, &bmp, &dib, &data, &palete);
    
    out = (uint32_t *)malloc(dib.image_size);
    
    printf("Computing the CPU output\n");
    printf("Image details: %d by %d = %d , imagesize = %d\n", dib.width, dib.height, dib.width * dib.height,dib.image_size);
    cutStartTimer (CPU_timer);
    // CPU_Sobel(data, out, dib.width, dib.height);
    
    switch(FilterMode)
    {
       
        case GRAYSCALE:
            CPU_Grayscale(data, out, dib.width, dib.height);
            break;
        case INVERT:
            CPU_Invert(data, out, dib.width, dib.height);
            break;
    }
    
    cutStopTimer (CPU_timer);
    BitMapWrite("CPU_output.bmp", &bmp, &dib, out, palete);
    printf("Done with CPU output\n");
    
    printf("Allocating %d bytes for image \n", dib.image_size);
    cutilSafeCall( cudaMalloc( (void **)&d_In, (dib.width * dib.height)*sizeof(unsigned int)) );
    printf("No Error in Input\n");
    cutilSafeCall( cudaMalloc( (void **)&d_Out, (dib.width * dib.height)*sizeof(unsigned int)) );
    printf("No Error in Ouput \n");
    cutStartTimer (timer_MemCpyHostToDevice);
    cudaMemcpy(d_In, data, (dib.width * dib.height)* sizeof(unsigned int), cudaMemcpyHostToDevice);
    cutStopTimer (timer_MemCpyHostToDevice);
    printf("No Error in memcpy from host to device\n");
    cutStartTimer (kernel_timer);
    FilterWrapper(data, dib.width, dib.height);
    cutStopTimer (kernel_timer);
    // Copy image back to host
    
    cutStartTimer (timer_MemCpyDeviceToHost);
    cudaMemcpy(out, d_Out, dib.image_size*sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cutStopTimer (timer_MemCpyDeviceToHost);
    
    printf("Kernel Execution time: %0.6f\n",cutGetTimerValue (kernel_timer));
    printf("Cpy Host to Device: %0.6f\n",cutGetTimerValue (timer_MemCpyDeviceToHost));
    printf("Cpy Device to Host time: %0.6f\n",cutGetTimerValue (timer_MemCpyHostToDevice));
    printf("CPU Time: %0.6f\n",cutGetTimerValue (CPU_timer));
    
    // Write output image
    BitMapWrite(BMPOutFile, &bmp, &dib, out, palete);
    
    Cleanup();
}

void Cleanup(void)
{
    cutDeleteTimer (kernel_timer);
    cutDeleteTimer (timer_MemCpyDeviceToHost);
    cutDeleteTimer (timer_MemCpyHostToDevice);
    cutDeleteTimer (CPU_timer);
    
    cutilSafeCall( cudaThreadExit() );
    exit(0);
}


void FilterWrapper(uint32_t* pImageIn, int Width, int Height)
{
    // Design grid disection around tile size
    int gridWidth  = (Width + TILE_WIDTH - 1) / TILE_WIDTH;
    int gridHeight = (Height + TILE_HEIGHT - 1) / TILE_HEIGHT;
    dim3 dimGrid(gridWidth, gridHeight);
    
    // But actually invoke larger blocks to take care of surrounding shared memory
    dim3 dimBlock(BLOCK_WIDTH, BLOCK_HEIGHT);
    
    switch(FilterMode) {
            
            
        case GRAYSCALE:
            printf("Grayscale Filter \n");
            GrayscaleFilter<<< dimGrid, dimBlock >>>(d_In, d_Out, Width, Height);
            cutilCheckMsg("kernel launch failure");
            break;
            
            
        case INVERT:
            printf("Invert Filter \n");
            InvertFilter<<< dimGrid, dimBlock >>>(d_In, d_Out, Width, Height);
            cutilCheckMsg("kernel launch failure");
            break;
            

    }
    cutilSafeCall( cudaThreadSynchronize() );
}



// Parse program arguments
void ParseArguments(int argc, char** argv)
{
    for (int i = 0; i < argc; ++i) {
        if (strcmp(argv[i], "--file") == 0 || strcmp(argv[i], "-file") == 0) {
            BMPInFile = argv[i+1];
            i = i + 1;
        }
        if (strcmp(argv[i], "--out") == 0 || strcmp(argv[i], "-out") == 0) {
            BMPOutFile = argv[i+1];
            i = i + 1;
        }
        if (strcmp(argv[i], "--filter") == 0 || strcmp(argv[i], "-filter") == 0) {
            Filter = argv[i+1];
            i = i + 1;
           
            if (strcmp(Filter, "grayscale") == 0)
                FilterMode = GRAYSCALE;
           
            else if (strcmp(Filter, "invert") == 0)
                FilterMode = INVERT;
  
            
            
        }
    }
}




