//Author:Sanjana Kalyanapagol
// Description: Image Procesing Algorithms
//Date: 06th May, 2017

#ifndef _FILTER_KERNEL_H_
#define _FILTER_KERNEL_H_
#include <stdio.h>

#define ALPHA(x)     ((x)>>24)
#define RED(x)       ((x) & 0xFF)
#define GREEN(x)     (((x)>>8) & 0xFF)
#define BLUE(x)      (((x)>>16) & 0xFF)




__global__ void GrayscaleFilter(uint32_t* g_DataIn, uint32_t* g_DataOut, int width, int height)
{
    __shared__ uint32_t sharedMem[BLOCK_HEIGHT*BLOCK_WIDTH];
    
    uint32_t avg = 0;
    
    int x = blockIdx.x * TILE_WIDTH + threadIdx.x ;//- FILTER_RADIUS;
    int y = blockIdx.y * TILE_HEIGHT + threadIdx.y ;//- FILTER_RADIUS;
    
    // Get the Global index into the original image
    int index = y * (width) + x;
    
    // Handle the extra thread case where the image width or height
    //
    if (x >= width || y >= height)
        return;
    
    // Handle the border cases of the global image
    if( x < FILTER_RADIUS || y < FILTER_RADIUS) {
        g_DataOut[index] = g_DataIn[index];
        return;
    }
    
    if ((x > width - FILTER_RADIUS - 1)&&(x <width)) {
        g_DataOut[index] = g_DataIn[index];
        return;
    }
    
    if ((y > height - FILTER_RADIUS - 1)&&(y < height)) {
        g_DataOut[index] = g_DataIn[index];
        return;
    }
    
    // Perform the first load of values into shared memory
    int sharedIndex = threadIdx.y * blockDim.y + threadIdx.x;
    sharedMem[sharedIndex] = g_DataIn[index];
    __syncthreads();
    
    avg = (uint8_t)(RED(g_DataIn[index]) + BLUE(g_DataIn[index]) + GREEN(g_DataIn[index]))/3;
    g_DataOut[index] = (ALPHA(g_DataIn[index]) << 24) | (avg << 16) |(avg << 8) |(avg);
    printf("\n In Kernel Grayscale");
    __syncthreads();
}




__global__ void InvertFilter(uint32_t* g_DataIn, uint32_t* g_DataOut, int width, int height)
{
    __shared__ uint32_t sharedMem[BLOCK_HEIGHT*BLOCK_WIDTH];
    
    uint32_t avg = 0;
    uint32_t r, g, b;
    
    int x = blockIdx.x * TILE_WIDTH + threadIdx.x ;//- FILTER_RADIUS;
    int y = blockIdx.y * TILE_HEIGHT + threadIdx.y ;//- FILTER_RADIUS;
    
    // Get the Global index into the original image
    int index = y * (width) + x;
    
    // Handle the extra thread case where the image width or height
    //
    if (x >= width || y >= height)
        return;
    
    // Handle the border cases of the global image
    if( x < FILTER_RADIUS || y < FILTER_RADIUS) {
        g_DataOut[index] = g_DataIn[index];
        return;
    }
    
    if ((x > width - FILTER_RADIUS - 1)&&(x <width)) {
        g_DataOut[index] = g_DataIn[index];
        return;
    }
    
    if ((y > height - FILTER_RADIUS - 1)&&(y < height)) {
        g_DataOut[index] = g_DataIn[index];
        return;
    }
    
    // Perform the first load of values into shared memory
    int sharedIndex = threadIdx.y * blockDim.y + threadIdx.x;
    sharedMem[sharedIndex] = g_DataIn[index];
    __syncthreads();
    
    r = RED(g_DataIn[index]);
    g = GREEN(g_DataIn[index]);
    b = BLUE(g_DataIn[index]);
    
    r = 255 - r;
    g = 255 - g;
    b = 255 - b;
    
    g_DataOut[index] = (ALPHA(g_DataIn[index]) << 24) | (b << 16) |(g << 8) |(r);
    //printf("\n In Kernel Invert");
    __syncthreads();
}





#endif // _FILTER_KERNEL_H_


