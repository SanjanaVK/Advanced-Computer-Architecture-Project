/*******************************************************************************************************************
Author: Sanjana Kalyanappagol 
Description: Matrix Multiplication
Date : 04th May, 2017 
********************************************************************************************************************/

/********************************************************** HEADERS ************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cutil_inline.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include "device_functions.h"

/**************************************************** INPUT ARRAY VARIABLES *****************************************/
void PrintArray(double*,int);

int in_size;
int out_size;
int WIDTH =300;

double* M1 = NULL;
double* N1 = NULL;
double* P1 = NULL;
double* Md = NULL;
double* Nd = NULL;
double* Pd = NULL;


 
unsigned char* d_In = NULL;
uint* d_Out = NULL;

struct Matrix{                                                                 /************ Matrix Structure ************/
	double elements[100000];
	int height;
	int width;
}M,N,P;


void RandomInit( int n)
{
    srand(1);	                                                              /************* Set rand() seed to 1 for repeatability *************/ 
    for(int i=0;i<n;i++) {	                                                  /************* Load array with digits *****************************/
        M.elements[i] = rand() % 64;                                          /************* Specify the number to be 0-63 **********************/
		M1[i] =  M.elements[i];
		N.elements[i] = rand() % 64;
		N1[i] = N.elements[i];
		//printf("value of K is %d,%G,%G \n",i,M.elements[i],N.elements[i]);
    }
} 


void MatrixMulOnHost()
{
	 for (int i = 0; i < M.height; ++i) {
	 for (int j = 0; j < N.width; ++j) {
		double sum = 0;
	 for (int k = 0; k < M.width; ++k) {
		double a = M.elements[i * M.width + k]; 
		double b = N.elements[k * N.width + j];
		sum += a * b;
	 }
	P.elements[i * N.width + j] = sum;
 }
 }
}
 
 
__global__ void MatrixMulKernel(double* Md ,double* Nd, double* Pd,int WIDTH)
{
/********************************************** 2D Thread Index. In the business of computing P[ty][tx] ****************************************/
int tx = threadIdx.x;
int ty = threadIdx.y;
                                /*** Pvalue will end up storing the value of P[ty][tx]. That is, P.elements[ty * P.WIDTH + tx] = Pvalue *******/

double Pvalue = 0;
int k;
for (k = 0; k < WIDTH; ++k) {
double Melement = Md[ty * WIDTH + k];
double Nelement = Nd[k * WIDTH + tx];
Pvalue += Melement * Nelement;
}
/************************************* Write the matrix to device memory each thread writes one element ***************************************/
k= ((ty * WIDTH) + tx);
Pd[k] = Pvalue;
} 


 

int main()
{
    
	int i;
	N.width=WIDTH;
	N.height=WIDTH;
	M.width=WIDTH;
	M.height=WIDTH;
	struct timeval start, end;
	struct timeval start2, end2;
	struct timezone tzp;

	dim3 dimGrid(1, 1);
	dim3 dimBlock(WIDTH,WIDTH);
	
/************************************************************ Sizes of the matrix are *********************************************************/
	in_size = M.height*M.width*sizeof(double); 
	out_size= M.height*M.width*sizeof(double);	
		
/****************************************************************** CPU malloc ****************************************************************/
	M1 = (double*)malloc(in_size);
	N1 = (double*)malloc(in_size);
	P1 = (double*)malloc(in_size);
	
	
	RandomInit(WIDTH*WIDTH);
	
	gettimeofday(&start, &tzp);
	MatrixMulOnHost();
	gettimeofday(&end, &tzp);
	printf("ARRAY M IS --->\n");
	for(i=0;i<(M.width*M.height);i++)
	{
		if(i%M.width==0)
		{
			printf("\n");
		}
		printf("%G\t",M.elements[i]);
	}
	printf("\n");
	printf("\n");
	printf("\n");
	printf("ARRAY N IS --->\n");
	for(i=0;i<(M.width*M.height);i++)
	{
		if(i%M.width==0)
		{
			printf("\n");
		}
		printf("%G\t",N.elements[i]);
	}
	printf("\n");
	printf("\n");
	printf("\n");
	
	printf("ARRAY P IS --->\n");
	for(i=0;i<(M.width*M.height);i++)
	{
		if(i%M.width==0)
		{
			printf("\n");
		}
		printf("%G\t",P.elements[i]);
	}
		
		
	cutilSafeCall( cudaMalloc((void**)&Md, in_size) );
    cutilSafeCall( cudaMalloc((void**)&Nd, in_size) );
	cutilSafeCall( cudaMalloc((void**)&Pd, out_size) );
	
	
	gettimeofday(&start2, &tzp);
	
	cutilSafeCall(cudaMemcpy(Md ,M1 ,in_size, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(Nd ,N1 ,in_size, cudaMemcpyHostToDevice));
	MatrixMulKernel<<<dimGrid, dimBlock>>>(Md, Nd, Pd, WIDTH);
	cutilSafeCall(cudaMemcpy(P1 ,Pd ,in_size, cudaMemcpyDeviceToHost));	
	
	gettimeofday(&end2, &tzp);
	
	printf("\n Size of matrix M- %d * %d",M.height,M.width);
	printf("\n Size of matrix N- %d * %d",N.height,N.width);
	printf("\nTime for cpu execution in usec: %lu\n\n", end.tv_usec - start.tv_usec);
	printf("\nTime for gpu execution in usec: %lu\n\n", end2.tv_usec - start2.tv_usec);
    int mul = WIDTH*WIDTH;
	
	
	return 0;
}


void PrintArray(double* data , int n)
{
    for (int i = 0; i < n; i++)
	{
        printf("[%d] => %G\n",i,data[i]);
	}
}






	
	
	

