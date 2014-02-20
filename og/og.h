#ifndef OG_H
#define OG_H
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <vector>
#include <string>
#include "helper_cuda.h"
#define _USE_MATH_DEFINES
#include "math.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/tuple.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/random.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/binary_search.h>
#include <thrust/gather.h>

#define NUM_PARTICLES 1000
#define SRR 0.1f
#define STR 0.1f
#define SRT 0.1f
#define STT 0.1f

typedef thrust::device_vector<float>::iterator floatIterator;
typedef thrust::tuple<floatIterator, floatIterator, floatIterator> floatIterTuple;
typedef thrust::zip_iterator<floatIterTuple> zipIteratorFloatTuple;

struct pseudorgnorm
{
	float a,b;
	__host__ __device__ pseudorgnorm(float _a=0.0f, float _b=1.0f): a(_a), b(_b) {};
	__host__ __device__ float operator()(const unsigned int n)const
	{
		thrust::default_random_engine rng;
		thrust::normal_distribution<float>dist(a,b);
		rng.discard(n);
		return dist(rng);
	}
};

struct pseudorg
{
	float a,b;
	__host__ __device__ pseudorg(float _a=0.0f, float _b=1.0f): a(_a), b(_b) {};
	__host__ __device__ float operator()(const unsigned int n)const
	{
		thrust::default_random_engine rng;
		thrust::uniform_real_distribution<float>dist(a,b);
		rng.discard(n);
		return dist(rng);
	}
};

template <typename T> 
struct cos_v : public thrust::unary_function<T, T>
{
	__host__ __device__ T operator()(T i)
	{
		return __cosf(i);
	}
};

struct correctAngle : public thrust::unary_function<float, float>
{
	float modulo;
	__host__ __device__ correctAngle(float _modulo): modulo(_modulo){}
	__host__ __device__ float operator()(float i)
	{
		i=fmodf(i, modulo);
		if(i>(modulo/2))
			i-=modulo;
		return i;
	}
};
template <typename T> 
struct sin_v : public thrust::unary_function<T, T>
{
	__host__ __device__ T operator()(T i)
	{
		return __sinf(i);
	}
};

struct score_to_weight : public thrust::unary_function<float, float>
{
	float max_val;
	__host__ __device__ score_to_weight(float _max_val): max_val(_max_val){}
	__host__ __device__ float operator()(float i)
	{
		return expf(1/(0.075*NUM_PARTICLES)*(i-max_val));
	}
};

__constant__ int mapW;
__constant__ int mapH;
__constant__ float resolution;
__constant__ float range_max;


float normalizeAngle(float a);
void drawFromMotion(thrust::device_vector<float>& x_part, thrust::device_vector<float>& y_part, thrust::device_vector<float>& theta_part, float x_h, float y_h, float theta_h, float x_o, float y_o, float theta_o, int seed);
bool readLine(std::ifstream& file, std::vector<int>& numScans, std::vector<std::vector<float>>& scans, std::vector<float>& x, std::vector<float>& y, std::vector<float>& theta);
bool loadLog(std::string filename, std::vector<int>& numScans, std::vector<std::vector<float>>& scans, std::vector<float>& x, std::vector<float>& y, std::vector<float>& theta);
void initMap(float * map, int w, int h, size_t pitch, int w_th, int h_th );
void computeMatchScores(thrust::device_vector<float> & x_part, thrust::device_vector<float> & y_part, thrust::device_vector<float> & theta_part, float * scan_gpu, float * map, size_t pitch, float * scores, int numScans);
void updateMapBresenham(float *map, size_t pitch, float * scan_gpu, float x, float y, float theta, int numBeams);
void init(float * map, size_t * pitch, int * width, int * height, thrust::device_vector<float> & x_part, thrust::device_vector<float> & y_part, thrust::device_vector<float> & theta_part, std::vector<int> & numScans, std::vector<std::vector<float>> & scans, std::vector<float> & xs, std::vector<float> & ys, std::vector<float> & thetas, float * x_old, float * y_old, float * theta_old);
float * get_map( float *map_gpu, int width, int height, size_t pitch);
void save_map(float * map_gpu, int width, int height, size_t pitch, char * filename);
int resample(thrust::device_vector<float> & x_part, thrust::device_vector<float> & y_part, thrust::device_vector<float> & theta_part, thrust::device_ptr<float> & weights, thrust::device_vector<float> & resampling_vector, thrust::device_vector<int> & resampled_indices);
void run();
inline void check_cuda_error(cudaError_t err)
{
	if(err!=cudaSuccess)
	{
		printf("cuda error number:%d\n%s\n", (int)err, cudaGetErrorString(err));
		exit(1);
	}

}
#endif

