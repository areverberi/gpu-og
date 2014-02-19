#include <stdlib.h>
#include <stdio.h>
#include "string.h"
#include <stddef.h>
#include <vector>
#define _USE_MATH_DEFINES
#include "math.h"
#include "helper_cuda.h"
#include<cuda.h>
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
#define PART_PER_THREAD 100
#define ALPHA1 0.2f
#define ALPHA2 0.2f
#define ALPHA3 0.2f
#define ALPHA4 0.2f
#define SRR 0.1f
#define STR 0.1f
#define SRT 0.1f
#define STT 0.1f
#define MAX_ICP_ITER 20

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

struct diff_from_zero : public thrust::unary_function<float, bool>
{
	__host__ __device__ bool operator()(float v)
	{
		return v!=0.0f;
	}
};

template <typename T> 
struct lin_to_row_index : public thrust::unary_function<T, T>
{
	T C;
	__host__ __device__ lin_to_row_index(T _C): C(_C) {}
	__host__ __device__ T operator()(T i)
	{
		return i/C;
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
		return exp(1/(0.075*NUM_PARTICLES)*(i-max_val));
	}
};

struct angleDiff_v : public thrust::binary_function<float, float, float>
{
	__host__ __device__ float operator()(float a, float b)
	{
		float d1, d2;
		a=atan2f(sinf(a), cosf(a));
		b=atan2f(sinf(b), cosf(b));
		d1=a-b;
		d2=M_PI*2-fabs(d1);
		if(d1>0)
			d2*=-1.0;
		if(fabs(d1)<fabs(d2))
			return d1;
		else 
			return d2;
	}
};

struct plusMod : public thrust::binary_function<float, float, float>
{

	float modulo;
	__host__ __device__ plusMod(float _modulo): modulo(_modulo){}
	__host__ __device__ float operator()(float a, float b)
	{
		float res=fmodf(a+b, modulo);
		if(res>M_PI)
			res-=2*M_PI;
		return res;
	}
};

//texture <float, 2, cudaReadModeElementType> radius;
//texture <float, 2, cudaReadModeElementType> angle;
//texture <float, 2, cudaReadModeElementType> sensor_model;

//__constant__ float x;
//__constant__ float y;
//__constant__ float theta;
__constant__ int mapW;
__constant__ int mapH;
__constant__ float resolution;
__constant__ float range_max;

float normalizeAngle(float a)
{
	return atan2f(sinf(a), cosf(a));
}

void drawFromMotion(thrust::device_vector<float>& x_part, thrust::device_vector<float>& y_part, thrust::device_vector<float>& theta_part, float x_h, float y_h, float theta_h, float x_o, float y_o, float theta_o, int seed)
{
	float sxy=0.3*SRR;
	float d_theta=normalizeAngle(theta_h-theta_o);
	float s=sinf(theta_o); float c=cosf(theta_o);
	float d_x=c*(x_h-x_o)+s*(y_h-y_o);
	if(d_x<0)
		printf("-----------------------------------------------------------------negative x\n");
	float d_y=-s*(x_h-x_o)+c*(y_h-y_o);
	float sigma_x=SRR*fabs(d_x)+STR*fabs(d_theta)+sxy*fabs(d_y);
	float sigma_y=SRR*fabs(d_y)+STR*fabs(d_theta)+sxy*fabs(d_x);
	float sigma_theta=STT*fabs(d_theta)+SRT*hypotf(d_x, d_y);
	thrust::counting_iterator<unsigned int> rndSeed(seed);
	thrust::device_vector<float> d_x_v(NUM_PARTICLES);
	thrust::device_vector<float> d_y_v(NUM_PARTICLES);
	thrust::device_vector<float> d_theta_v(NUM_PARTICLES);
	thrust::transform(rndSeed, rndSeed+NUM_PARTICLES, d_x_v.begin(), pseudorgnorm(d_x, sigma_x));
	thrust::transform(rndSeed+NUM_PARTICLES, rndSeed+NUM_PARTICLES*2, d_y_v.begin(), pseudorgnorm(d_y, sigma_y));
	thrust::transform(rndSeed+NUM_PARTICLES*2, rndSeed+NUM_PARTICLES*3, d_theta_v.begin(), pseudorgnorm(d_theta, sigma_theta));
	thrust::transform(d_theta_v.begin(), d_theta_v.end(), d_theta_v.begin(), correctAngle(2*M_PI));
	thrust::transform_iterator<sin_v<float>, thrust::device_vector<float>::iterator> sin_iter(theta_part.begin(), sin_v<float>());
	thrust::transform_iterator<cos_v<float>, thrust::device_vector<float>::iterator> cos_iter(theta_part.begin(), cos_v<float>());
	thrust::device_vector<float> cosx(NUM_PARTICLES), sinx(NUM_PARTICLES), cosy(NUM_PARTICLES), siny(NUM_PARTICLES);
	thrust::transform(d_x_v.begin(), d_x_v.end(), sin_iter, sinx.begin(), thrust::multiplies<float>());
	thrust::transform(d_x_v.begin(), d_x_v.end(), cos_iter, cosx.begin(), thrust::multiplies<float>());
	thrust::transform(d_y_v.begin(), d_y_v.end(), sin_iter, siny.begin(), thrust::multiplies<float>());
	thrust::transform(d_y_v.begin(), d_y_v.end(), cos_iter, cosy.begin(), thrust::multiplies<float>());
	thrust::transform(cosx.begin(), cosx.end(), siny.begin(), d_x_v.begin(), thrust::minus<float>());
	thrust::transform(sinx.begin(), sinx.end(), cosy.begin(), d_y_v.begin(), thrust::plus<float>());
	thrust::transform(x_part.begin(), x_part.end(), d_x_v.begin(), x_part.begin(), thrust::plus<float>());
	thrust::transform(y_part.begin(), y_part.end(), d_y_v.begin(), y_part.begin(), thrust::plus<float>());
	thrust::transform(theta_part.begin(), theta_part.end(), d_theta_v.begin(), theta_part.begin(), thrust::plus<float>());
}

struct gt: public thrust::binary_function<float, float, int>
{
	__host__ __device__ int operator()(float a, float b)
	{
		int res=a>b?0:1;
		return res;
	}
};
//void ICPStep(float * scores, int * scores_mask, thrust::device_vector<float> & new_x_part, thrust::device_vector<float> & new_y_part, thrust::device_vector<float> & new_theta_part, float * map, thrust::device_vector<float> & x_part, thrust::device_vector<float> & y_part, thrust::device_vector<float> & theta_part, float * scan_gpu, thrust::device_vector<float> & likelihood, bool score_only)
//{
//	thrust::device_vector<float> x_b(360*NUM_PARTICLES), y_b(360*NUM_PARTICLES), x_p_n(360*NUM_PARTICLES), y_p_n(360*NUM_PARTICLES);
//	thrust::device_vector<int> stencil(360*NUM_PARTICLES);
//	<<<360, NUM_PARTICLES>>> ICPStepKernel(scores, scores_mask, thrust::raw_pointer_cast(&x_p_n[0]), thrust::raw_pointer_cast(&y_p_n[0]), thrust::raw_pointer_cast(&stencil[0]), map, trust::raw_pointer_cast(&x_part[0]), trust::raw_pointer_cast(&y_part[0]), trust::raw_pointer_cast(&theta_part[0]), thrust::raw_pointer_cast(&x_b[0]), thrust::raw_pointer_cast(&y_b[0]), scan_gpu, trust::raw_pointer_cast(likelihood), scoreOnly);
//	if(!score_only)
//	{
//		thrust::counting_iterator<int> part_index(0);
//		thrust::device_vector<int> row_index(360*NUM_PARTICLES);
//		thrust::device_vector<int> r(NUM_PARTICLES);
//		thrust::sequence(row_index.begin(), row_index.end(), 0);
//		thrust::transform(row_index.begin(), row_index.end(), row_index.begin(), lin_to_row_index<int>(360));
//		thrust::zip_iterator<thrust::tuple<thrust::device_vector<int>::iterator, thrust::device_vector<float>::iterator, thrust::device_vector<float>::iterator, thrust::device_vector<float>::iterator, thrust::device_vector<float>::iterator>> part_zip_iter=thrust::make_zip_iterator(thrust::make_tuple(row_index.begin(), x_b.begin(), y_b.begin(), x_p_n.begin(), y_p_n.begin()));
//		thrust::zip_iterator<thrust::tuple<thrust::device_vector<int>::iterator, thrust::device_vector<float>::iterator, thrust::device_vector<float>::iterator, thrust::device_vector<float>::iterator, thrust::device_vector<float>::iterator>> part_zip_iter_end=part_zip_iter+360*NUM_PARTICLES;
//		part_zip_iter_end=thrust::remove_if(part_zip_iter, part_zip_iter_end, stencil.begin(), thrust::identity<int>());
//		int distance=part_zip_iter_end-part_zip_iter;
//		thrust::device_vector<float> x_b_avg(NUM_PARTICLES);
//		thrust::reduce_by_key(row_index.begin(), row_index.begin()+distance, x_b.begin(), r.begin(), x_b_avg.begin(), thrust::equal_to<int>(), thrust::plus<float>());
//		thrust::transform(x_b_avg.begin(), x_b_avg.end(), 
//		thrust::device_vector<float> y_b_avg(NUM_PARTICLES);
//		thrust::reduce_by_key(row_index.begin(), row_index.begin()+distance, y_b.begin(), r.begin(), y_b_avg.begin(), thrust::equal_to<int>(), thrust::plus<float>());
//		thrust::device_vector<float> x_p_n_avg(NUM_PARTICLES);
//		thrust::reduce_by_key(row_index.begin(), row_index.begin()+distance, x_p_n.begin(), r.begin(), x_p_n_avg.begin(), thrust::equal_to<int>(), thrust::plus<float>());
//		thrust::device_vector<float> y_p_n_avg(NUM_PARTICLES);
//		thrust::reduce_by_key(row_index.begin(), row_index.begin()+distance, y_p_n.begin(), r.begin(), y_p_n_avg.begin(), thrust::equal_to<int>(), thrust::plus<float>());
//
//
//
//	}
//}
//thrust::device_vector<float> & ICP(thrust::device_vector<float>& new_x_part,thrust::device_vector<float>& new_y_part,thrust::device_vector<float>& new_theta_part, float * map, thrust::device_vector<float>& x_part, thrust::device_vector<float>& y_part, thrust::device_vector<float>& theta_part, float * scan_gpu, thrust::device_vector<float>& likelihood)
//{
//	thrust::device_vector<float> sc(NUM_PARTICLES);
//	thrust::device_vector<float> cur_sc(NUM_PARTICLES);
//	thrust::fill(cur_sc.begin(), cur_sc.end(), 0.0f);
//	thrust::device_vector<int> sc_mask(NUM_PARTICLES);
//	thrust::device_vector<float> start_x(NUM_PARTICLES);
//	thrust::device_vector<float> start_y(NUM_PARTICLES);
//	thrust::device_vector<float> start_theta(NUM_PARTICLES);
//	thrust::fill(sc_mask.begin(), sc_mask.end(), 0);
//	ICPStep(thrust::raw_pointer_cast(&sc[0]), thrust::raw_pointer_cast(&sc_mask[0]),new_x_part,  new_y_part, new_theta_part, map, x_part, y_part, theta_part, scan_gpu, likelihood, true);
//	int iter=0;
//	thrust::copy(x_part.begin(), x_part.end(), start_x.begin());
//	thrust::copy(y_part.begin(), y_part.end(), start_y.begin());
//	thrust::copy(theta_part.begin(), theta_part.end(), start_theta.begin());
//	int num_finished=0;
//	do
//	{
//		thrust::copy(sc.begin(), sc.end(), cur_sc.begin());
//		thrust::fill(sc.begin(), sc.end(), 0.0f);
//		ICPStep(thrust::raw_pointer_cast(&sc[0]), thrust::raw_pointer_cast(&sc_mask[0]), new_x_part, new_y_part, new_theta_part, map, start_x, start_y, start_theta, scan_gpu, likelihood, false);
//		thrust::transform(sc.begin(), sc.end(), cur_sc.begin(), sc_mask.begin(), gt());
//		num_finished=thrust::reduce(sc_mask.begin(), sc_mask.end());
//		thrust::copy(new_x_part.begin(), new_x_part.end(), start_x.begin());
//		thrust::copy(new_y_part.begin(), new_y_part.end(), start_y.begin());
//		thrust::copy(new_theta_part.begin(), new_theta_part.end(), start_theta.begin());
//		++iter;
//	}while(num_finished<NUM_PARTICLES && iter<MAX_ICP_ITER);
//	return cur_sc;
//}

float angleDiff(float a, float b)
{
	float d1, d2;
	a=normalizeAngle(a);
	b=normalizeAngle(b);
	d1=a-b;
	d2=M_PI*2-fabs(d1);
	if(d1>0)
		d2*=-1.0;
	if(fabs(d1)<fabs(d2))
		return d1;
	else 
		return d2;
}

__device__ float fatomicMin(float *addr, float value)
{
	float old = *addr, assumed;
	if(old <= value) return old;
	do
	{
		assumed = old;
		old = atomicCAS((unsigned int*)addr, __float_as_int(assumed), __float_as_int(value));
	}while(old!=assumed);
	return old;
}
bool readLine(std::ifstream& file, std::vector<int>& numScans, std::vector<std::vector<float>>& scans, std::vector<float>& x, std::vector<float>& y, std::vector<float>& theta)
{
	std::string line_type;
	file>>line_type;
	if(line_type=="#" || line_type=="PARAM" || line_type=="ODOM" || line_type=="NEFF")
	{
		std::string skip;
		std::getline(file, skip);
		return true;
	}
	if(line_type=="FLASER")
	{
		int num;
		file>>num;
		numScans.push_back(num);
		std::vector<float> scan;
		for(unsigned int i=0; i<num; ++i)
		{
			float s;
			file>>s;
			scan.push_back(s);
		}
		scans.push_back(scan);
		float t;
		file>>t;
		x.push_back(t);
		file>>t;
		y.push_back(t);
		file>>t;
		theta.push_back(t);
		std::string rem;
		std::getline(file, rem);
		return true;
	}
	return false;
}
bool loadLog(std::string filename, std::vector<int>& numScans, std::vector<std::vector<float>>& scans, std::vector<float>& x, std::vector<float>& y, std::vector<float>& theta)
{
	std::ifstream file(filename.c_str());
	if(!file.is_open())
		return false;
	while(readLine(file, numScans, scans, x, y, theta));
	return true;
}
char* mystrsep(char** stringp, const char* delim)
{
  char* start = *stringp;
  char* p;

  p = (start != NULL) ? strpbrk(start, delim) : NULL;

  if (p == NULL)
  {
    *stringp = NULL;
  }
  else
  {
    *p = '\0';
    *stringp = p + 1;
  }

  return start;
}
__global__ void initMap(float* map, int w, int h, size_t pitch, int w_th, int h_th){
	unsigned int idx=blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int idy=blockIdx.y*blockDim.y+threadIdx.y;
	//unsigned int w_th=w/(blockDim.x*gridDim.x);
	//unsigned int h_th=h/(blockDim.y*gridDim.y);
	for(unsigned int i=0; i<h_th; ++i)
	{
		for(unsigned int j=0; j<w_th; ++j)
		{
			unsigned int x=idx*w_th+j;
			unsigned int y=idy*h_th+i;
			if(x<w && y<h)
			{
				map[x+y*pitch]=-1.0f;
			}
		}
	}
    __syncthreads();
}

__device__ void getCoordsBresenham(float *coords, float * range, float * x_o, float * y_o, float * theta_o, int coord_sys=0)
{
	__shared__ int x1, y1, x2, y2;
	__shared__ float delta_x, delta_y, m;
	__shared__ int sign_delta_x, sign_delta_y;
	__shared__ float theta_b;

	if(threadIdx.x==0)
	{
		theta_b=*theta_o+blockIdx.x*M_PI/359-M_PI_2;
		float s;
		float c;
		__sincosf(theta_b, &s, &c);
		//mapW/H is offset, 0.1f is resolution
		x1=(int)floorf(mapW/2+*x_o/resolution);
		y1=(int)floorf(mapH/2+*y_o/resolution);
		//0.1f for wall thickness, if needed, add to range before mul
		x2=(int)floorf(mapW/2+(*x_o +(*range+0.1f)*c)/resolution);
		y2=(int)floorf(mapH/2+(*y_o +(*range+0.1f)*s)/resolution);
		delta_x=(float)(x2-x1);
		delta_y=(float)(y2-y1);
		/*
		sign_delta_x=1;
		if(delta_x<0)sign_delta_x=-1;
		sign_delta_y=1;
		if(delta_y<0)sign_delta_y=-1;
		*/
		sign_delta_x=copysignf(1, delta_x);
		sign_delta_y=copysignf(1, delta_y);
	}
	__syncthreads();
	if(*range<range_max)
	{
		int current_x, current_y, pos;
		if(coord_sys==0)
			pos=threadIdx.x;
		else
			pos=blockIdx.y;
		if(fabs(delta_y)>fabs(delta_x))
		{
			m=delta_x/delta_y;
			current_y=y1+sign_delta_y*pos;
			current_x=x1+rintf(m*(current_y-y1));
            //current_x=x1+floorf(0.4999999f+m*(current_y-y1));
		}
		else
		{
			m=delta_y/delta_x;
			current_x=x1+sign_delta_x*pos;
			current_y=y1+rintf(m*(current_x-x1));
            //current_y=y1+floorf(0.4999999f+m*(current_x-x1));
		}
		coords[0]=current_x;
		coords[1]=current_y;
	}
	else
	{
		coords[0]=-1;
		coords[1]=-1;
	}
	if(coords[0]>=0 && coords[0]<mapW && coords[1]>=0 && coords[1]<mapH)
	{
		coords[2]=hypotf(coords[0]-x1, coords[1]-y1)*resolution;
	}
	else
	{
		coords[2]=-1;
	}
}

__global__ void computeMatchScores(float * x_part, float * y_part, float * theta_part, float * scan_gpu, float *map, size_t pitch, float * scores)
{
	//__shared__ float range;
	////__shared__ float x;
	////__shared__ float y;
	////__shared__ float theta;
	//__shared__ float true_range;
	////__shared__ float computed_ranges[256];
	//float coords[3];
	//if(threadIdx.x==0)
	//{
	//	range=range_max-0.0001f;
	//	//true_range=scan_gpu[blockIdx.x];
	//}
	//__syncthreads();
	//for(int i=threadIdx.x*PART_PER_THREAD; i<threadIdx.x*PART_PER_THREAD+PART_PER_THREAD; ++i)
	//{
	//	float x=x_part[i];
	//	float y=y_part[i];
	//	float theta=theta_part[i];
	//	getCoordsBresenham(coords, &range, &x, &y, &theta, 1);
	//	if(coords[2]>=0.0f && map[(int)coords[0]+(int)coords[1]*pitch]>0.5f)
	//	//computed_ranges[threadIdx.x]=coords[2];
	//		fatomicMin(&scores[i], coords[2]);
	//}
	//__syncthreads();
	//computed_ranges[threadIdx.x]=range_max;	
	/*int threadsInB=blockDim.x;
	while(threadsInB>1)
	{
		int halfTh=(threadsInB >> 1);
		if(threadIdx.x<halfTh)
		{
			int thread2=threadIdx.x+halfTh;
			float temp=computed_ranges[thread2];
			if(temp<computed_ranges[threadIdx.x])
				computed_ranges[threadIdx.x]=temp;
		}
		__syncthreads();
		threadsInB=halfTh;
	}
	__syncthreads();*/
	//if(threadIdx.x==0)
	//{
	//	//float score=(true_range-computed_ranges[0])*(true_range-computed_ranges[0])/(true_range*computed_ranges[0]);
	//	float score=(true_range-computed_range)*(true_range-computed_range)/(true_range*computed_range);
	//	scores[blockIdx.x+blockIdx.y*gridDim.y]=score;
	//}
	__shared__ float x,y,theta;
	__shared__ float score;
	if(threadIdx.x==0)
	{
		x=x_part[blockIdx.x];
		y=y_part[blockIdx.x];
		theta=theta_part[blockIdx.x];
		score=0;
	}
	__syncthreads();
	float range=scan_gpu[threadIdx.x];
	bool found=false;
	float minDist;
	if (range<6.4f)
	{
		float theta_t=theta+threadIdx.x*M_PI/360-M_PI_2;
		float s, c;
		__sincosf(theta_t, &s, &c);
		int x_hit=(int)floorf(mapW/2+(x+(range+0.1f)*c)/resolution);
		int y_hit=(int)floorf(mapH/2+(y+(range+0.1f)*s)/resolution);

		for(int i=-2; i<=2; ++i)
		{
			for(int j=-2; j<=2; ++j)
			{
				int x_hit_k=x_hit+i;
				int y_hit_k=y_hit+j;
				if(x_hit>=0 && x_hit<mapW && y_hit>=0 && y_hit<mapH && x_hit_k>=0 && x_hit_k<mapW && y_hit_k>=0 && y_hit_k<mapH)
					if(map[x_hit_k+y_hit_k*pitch]>0.5f)
					{
						if(!found)
						{
							minDist=i*i+j*j;
							found=true;
						}
						else
						{
							float dist=i*i+j*j;
							minDist=dist<minDist?dist:minDist;
						}
					}
			}
		} 
	}
	float pt_score=found?-minDist/0.075f:-0.5f/0.075f;
	/*if(minDist==0.0f && found)
		printf("mindist is 0\n");
	*/if(found)
		atomicAdd(&score, pt_score);
	else
		atomicAdd(&score, pt_score);
	/*
	if(x_hit>=0 && x_hit<mapW && y_hit>=0 && y_hit<mapH)
		if(map[x_hit+y_hit*pitch]>0.5f)
			atomicInc(&score, 0);
			*/
	__syncthreads();
	if(threadIdx.x==0)
	{
		scores[blockIdx.x]=(float)score;
	}
}

__global__ void updateMapBresenham(float *map, size_t pitch, float *scan_gpu, float x, float y, float theta){
	__shared__ float range;
	float coords[3];
	if(threadIdx.x==0)
	{
		range=scan_gpu[blockIdx.x];
	}
	getCoordsBresenham(coords, &range, &x, &y, &theta);
	//printf("coords:%d %d\n", coords[0], coords[1]);
	if(coords[2]>=0)
	{
		//0.1f because going from grid (10cm cell) to meters
		float d=coords[2];
		int current_x=(int)coords[0];
		int current_y=(int)coords[1];
		//divide by 100 because rmax is #of cells, ie 500->turn to meters
		//float k=1-(d/rmax)*(d/rmax)/100;
		//float k=1;
        //float k=0.6;
		//float s=0.00001425*range*range;
		//float s=0.4;
        //float s=0.6;
		//float expon=((d-range)/s)*((d-range)/s);
		float prob;

		if(d<range)
		{
			//sensor model
			//prob=0.3+(k/s*__frsqrt_rn(s)+0.2)*__expf(-0.5*expon);
			if(d<1.0f)
			prob=0.45f;
			else
			prob=0.45f+(d-1.0f)/6.4f*(0.5f-0.45f);
		}
		else
		{
			//sensor model
			//prob=0.5+k/s*__frsqrt_rn(s)*__expf(-0.5*expon);
					
			if(d<1.0f)
			prob=0.75f;
			else
			prob=0.75f+(d-1.0f)/6.4f*(0.5f-0.75f);
					
		}
		//map[current_x+current_y*pitch]+=__logf(prob/(1-prob));
				
		if (d<=range+0.1f && d<=6.4f)
		{
			float pr=map[current_x+current_y*pitch];
			if(pr==-1.0f)
				pr=0.5f;
			map[current_x+current_y*pitch]=1.0f-1.0f/(1.0f+prob/(1.0f-prob)*pr/(1.0f-pr));

			//printf("-------------------------------------------------------------------updating map\n");  
		}
	}
	//if(threadIdx.x==0)
	//{
	//	range=scan_gpu[blockIdx.x];
	//	theta_b=theta+blockIdx.x*M_PI/359-M_PI_2;
	//	float s;
	//	float c;
	//	__sincosf(theta_b, &s, &c);
	//	//mapW/H is offset, 0.1f is resolution
	//	x1=(int)floorf(mapW/2+x/resolution);
	//	y1=(int)floorf(mapH/2+y/resolution);
	//	//0.1f for wall thickness, if needed, add to range before mul
	//	x2=(int)floorf(mapW/2+(x+(range+0.1f)*c)/resolution);
	//	y2=(int)floorf(mapH/2+(y+(range+0.1f)*s)/resolution);
	//	delta_x=(float)(x2-x1);
	//	delta_y=(float)(y2-y1);
	//	/*
	//	sign_delta_x=1;
	//	if(delta_x<0)sign_delta_x=-1;
	//	sign_delta_y=1;
	//	if(delta_y<0)sign_delta_y=-1;
	//	*/
	//	sign_delta_x=copysignf(1, delta_x);
	//	sign_delta_y=copysignf(1, delta_y);
	//}
	//__syncthreads();
	//if(range<range_max)
	//{
	//	int current_x, current_y;
	//	if(fabs(delta_y)>fabs(delta_x))
	//	{
	//		m=delta_x/delta_y;
	//		current_y=y1+sign_delta_y*threadIdx.x;
	//		current_x=x1+rintf(m*(current_y-y1));
 //           //current_x=x1+floorf(0.4999999f+m*(current_y-y1));
	//	}
	//	else
	//	{
	//		m=delta_y/delta_x;
	//		current_x=x1+sign_delta_x*threadIdx.x;
	//		current_y=y1+rintf(m*(current_x-x1));
 //           //current_y=y1+floorf(0.4999999f+m*(current_x-x1));
	//	}
	//	if(current_x>=0 && current_x<mapW && current_y>=0 && current_y<mapH)
	//	{
	//		//0.1f because going from grid (10cm cell) to meters
	//		float d=hypotf(current_x-x1, current_y-y1)*resolution;
	//		//divide by 100 because rmax is #of cells, ie 500->turn to meters
	//		//float k=1-(d/rmax)*(d/rmax)/100;
	//		//float k=1;
 //           //float k=0.6;
	//		//float s=0.00001425*range*range;
	//		//float s=0.4;
 //           //float s=0.6;
	//		//float expon=((d-range)/s)*((d-range)/s);
	//		float prob;

	//		if(d<range)
	//		{
	//			//sensor model
	//			//prob=0.3+(k/s*__frsqrt_rn(s)+0.2)*__expf(-0.5*expon);
	//			if(d<1.0f)
	//			prob=0.45f;
	//			else
	//			prob=0.45f+(d-1.0f)/6.4f*(0.5f-0.45f);
	//		}
	//		else
	//		{
	//			//sensor model
	//			//prob=0.5+k/s*__frsqrt_rn(s)*__expf(-0.5*expon);
	//				
	//			if(d<1.0f)
	//			prob=0.75f;
	//			else
	//			prob=0.75f+(d-1.0f)/6.4f*(0.5f-0.75f);
	//				
	//		}
	//		//map[current_x+current_y*pitch]+=__logf(prob/(1-prob));
	//			
	//		if (d<=range+0.1f && d<=6.4f)
	//		{
	//			float pr=map[current_x+current_y*pitch];
	//			if(pr==-1.0f)
	//				pr=0.5f;
	//			map[current_x+current_y*pitch]=1.0f-1.0f/(1.0f+prob/(1.0f-prob)*pr/(1.0f-pr));

	//			//printf("-------------------------------------------------------------------updating map\n");  
	//		}
	//	}
	//	else
	//	{
	//		//printf("%d %d\n", current_x, current_y); 
	//	}
	//}
	//else
	//{
	//	//printf("range: %d \n", range);
	//}
}
//__global__ void __launch_bounds__(1024) updateMap(float x, float y, float theta, float* map, float* scan_gpu, size_t pitch, int mapW, int mapH, float rmax){
//    __shared__ float scan[360];
//	/*first 360 threads load scan*/
//	unsigned int scanperthread=360/(blockDim.x*blockDim.y);
//	if(scanperthread>1){
//		unsigned int ind=(threadIdx.x*blockDim.x+threadIdx.y)*scanperthread;
//		unsigned int off;
//		for(off=0; off<scanperthread; off++){
//			if(ind+off<360)
//				scan[ind+off]=scan_gpu[ind+off];
//		}
//	}
//	else{
//		unsigned int ind=threadIdx.x*blockDim.x+threadIdx.y;
//		if(ind<360){
//			scan[ind]=scan_gpu[ind];
//		}
//	}
//	//printf("scan loaded\n");
//	__syncthreads();
//	float x_local_lu=(blockIdx.x*blockDim.x+threadIdx.x)*1.0/(gridDim.x*blockDim.x)*rmax;
//    float y_local_lu=(blockIdx.y*blockDim.y+threadIdx.y)*1.0/(gridDim.y*blockDim.y)*rmax;
//	/*to fix: the 10.0 should be s_m_resolution*/
//	//float val=tex2D(sensor_model, tex2D(radius, x_local_lu, y_local_lu)*10.0, scan[(int)rint(tex2D(angle, x_local_lu, y_local_lu))]*10.0);
//	float val=tex2D(sensor_model, scan[(int)rint(tex2D(angle, x_local_lu, y_local_lu))]*10.0, tex2D(radius, x_local_lu, y_local_lu)*10.0);
//	//if(tex2D(radius, x_local_lu, y_local_lu)>scan[(int)rint(tex2D(angle, x_local_lu, y_local_lu))])
//	//	printf("val:%f\n", val);
//	//printf("angle:%d\n", (int)rint(tex2D(angle, x_local_lu, y_local_lu)));
//	//printf("val:%f\n", val);
//	if (val!=0.5f)
//	{
//		float x_local=x_local_lu-rmax/2;
//		float y_local=rmax/2-y_local_lu;
//		x_local=x_local*__cosf(theta)-y_local*__sinf(theta);
//		y_local=x_local*__sinf(theta)+y_local*__cosf(theta);
//		//int x_map=(int)rint(x_local*cosf(theta)+y_local*sinf(theta)-x+mapW/2.0);
//		//int y_map=(int)rint(-x_local*sinf(theta)+y_local*cosf(theta)-y+mapH/2.0);
//		int x_map_cell=(int)rint(x_local+x*10.0f+mapW/2.0);
//		int y_map_cell=(int)rint(-(y_local+y*10.0f-mapH/2.0));
//		/*if(x_map_cell<0 || y_map_cell<0)
//		printf("%f %f\n", x_map_cell, y_map_cell);
//		*/if(x_map_cell<mapH && y_map_cell<mapW ){
//			//no size difference between local and global cells, otherwise you'd need to divide by global cell size to get map cell
//			//int x_map_cell=(int)rint(x_map);
//			//int y_map_cell=(int)rint(y_map);
//			if(scan[(int)rint(tex2D(angle, x_local_lu, y_local_lu))]>0.0)
//			{
//				size_t index=x_map_cell*pitch+y_map_cell;
//				//map[index]=0.5f*val+0.5f*map[index];
//				map[index]=1-1/(1+map[index]/(1-map[index])*val/(1-val));
//			}
//			/*
//			if(scan[(int)rint(tex2D(angle, x_local_lu, y_local_lu))]>0.0){}
//			if(map[index]<0.0f)
//			map[index]=val;
//			else
//			map[index]=0.5f*val+0.5f*map[index];
//			*/
//		} 
//	}
//    __syncthreads();
//}

int main(int argc, char** argv){
	/*float *r;
	float *a;
	float *s_m;*/
	/*size of the matrix in cells*/
	int local_size=500;
	//int map_size=1000;
    int map_size=1600;
	float cell_dim=0.1;
	//int map_size_x=1600;
	//int map_size_y=880;
	//float s_m_resolution=10.0;
	//r=(float*)malloc(sizeof(float)*local_size*local_size);
	//a=(float*)malloc(sizeof(float)*local_size*local_size);
	//s_m=(float*)malloc(sizeof(float)*local_size*local_size*(int)(s_m_resolution*s_m_resolution));
	//int loopX=0;
	//int loopY=0;
	///*initialization of lookups for radius, angle and sensor model*/
	//for(loopY=0; loopY<local_size*s_m_resolution; loopY++){
	//	for(loopX=0; loopX<local_size*s_m_resolution; loopX++){
	//		if(loopX<local_size && loopY<local_size){
	//			float x_cell=loopX*cell_dim+cell_dim/2.0f-local_size*cell_dim/2.0f;
	//			float y_cell=-loopY*cell_dim+cell_dim/2.0f+local_size*cell_dim/2.0f;
	//			r[loopY*local_size+loopX]=hypotf(x_cell, y_cell);
	//			a[loopY*local_size+loopX]=(atan2(y_cell, x_cell)+M_PI)/M_PI*180.0f;
	//			//a[loopY*local_size+loopX]=atan2(y_cell, x_cell)/M_PI*180.0f;
	//		}
	//		if (abs(loopX-loopY)<s_m_resolution/2.0){
	//			s_m[loopY*local_size*((int)s_m_resolution)+loopX]=0.95f;
	//		}
	//		else{
	//			if (loopY<loopX){
	//				s_m[loopY*local_size*((int)s_m_resolution)+loopX]=0.05f;
	//			}
	//			else{ 
	//				/*
	//				float min=(local_size*10<loopX+s_m_resolution?local_size*10:loopX+s_m_resolution);
	//				if (loopY> min){
	//					s_m[loopX*local_size*10+loopY]=0.5f;
	//				}
	//				*/
	//				s_m[loopY*local_size*((int)s_m_resolution)+loopX]=0.5f;
	//			}
	//		}
	//	}
	//}
	///*setting filter mode for the textures. It's linear for radius and angle so I get interpolation "for free"*/
	////printf("radius 0:%f\n", r[0]);
	////printf("angle 0:%f\n", a[0]);
	////getchar();
	//sensor_model.filterMode=cudaFilterModePoint;
	//radius.filterMode=cudaFilterModeLinear;
	//angle.filterMode=cudaFilterModeLinear;	

	/*creating the cudaArrays that will contain the textures*/
	cudaChannelFormatDesc cf=cudaCreateChannelDesc<float>();
	/*cudaArray *r_gpu;
	checkCudaErrors(cudaMallocArray(&r_gpu, &cf, local_size, local_size));
	checkCudaErrors(cudaMemcpyToArray(r_gpu, 0, 0, r, sizeof(float)*local_size*local_size, cudaMemcpyHostToDevice));
	cudaArray *a_gpu;
	checkCudaErrors(cudaMallocArray(&a_gpu, &cf, local_size, local_size));
	checkCudaErrors(cudaMemcpyToArray(a_gpu, 0, 0, a, sizeof(float)*local_size*local_size, cudaMemcpyHostToDevice));
	float *s_m_gpu;
	size_t pitch_s;
	checkCudaErrors(cudaMallocPitch(&s_m_gpu, &pitch_s, local_size*((int)s_m_resolution)*sizeof(float), local_size*((int)s_m_resolution)));
	checkCudaErrors(cudaMemcpy2D(s_m_gpu, pitch_s, s_m, local_size*((int)s_m_resolution)*sizeof(float), local_size*((int)s_m_resolution)*sizeof(float), local_size*((int)s_m_resolution), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy2D(s_m, local_size*((int)s_m_resolution)*sizeof(float), s_m_gpu, pitch_s, local_size*((int)s_m_resolution)*sizeof(float), local_size*((int)s_m_resolution), cudaMemcpyDeviceToHost));
	FILE *s_m_ff;
	FILE *rad;
	FILE *ang;
	s_m_ff=fopen("sensor.dat", "w");
	rad=fopen("radius.dat", "w");
	ang=fopen("angle.dat", "w");
	if(s_m_ff!=NULL){
		fwrite(s_m, sizeof(float), local_size*((int)s_m_resolution)*local_size*((int)s_m_resolution), s_m_ff);
	}
	if(rad!=NULL)
	{
		fwrite(r, sizeof(float), local_size*local_size, rad);
	}
	if(ang!=NULL)
	{
		fwrite(a, sizeof(float), local_size*local_size, ang);
	}

	fclose(s_m_ff);
	fclose(rad);
	fclose(ang);
	*/
	/*map initialization and texture binding*/
    int width=map_size;
    int height=map_size;
	float res=0.05f;
	float rmax=50.0f;
    float* map;
    size_t pitch;
	checkCudaErrors(cudaMallocPitch(&map,&pitch,width*sizeof(float), height));
	checkCudaErrors(cudaMemcpyToSymbol(mapW, &width, sizeof(int)));
	checkCudaErrors(cudaMemcpyToSymbol(mapH, &height, sizeof(int)));
	checkCudaErrors(cudaMemcpyToSymbol(resolution, &res, sizeof(float)));
	checkCudaErrors(cudaMemcpyToSymbol(range_max, &rmax, sizeof(float)));
	dim3 numThr(32, 32);
	dim3 numBlocks(width/numThr.x, height/numThr.y);
    initMap <<<numBlocks, numThr>>> (map, width, height, pitch/sizeof(float), 1, 1);
	thrust::device_vector<float> delta_t_v(NUM_PARTICLES);
	thrust::device_vector<float> delta_r1_v(NUM_PARTICLES);
	thrust::device_vector<float> delta_r2_v(NUM_PARTICLES);
	thrust::device_vector<float> temp(NUM_PARTICLES);
	thrust::device_vector<float> x_part(NUM_PARTICLES);
	thrust::device_vector<float> y_part(NUM_PARTICLES);
	thrust::device_vector<float> theta_part(NUM_PARTICLES);
	float * scanScores;
	checkCudaErrors(cudaMalloc(&scanScores, NUM_PARTICLES*sizeof(float)));
	thrust::device_ptr<float> weights(scanScores);
	thrust::device_vector<float> resampling_vector(NUM_PARTICLES);
	thrust::device_vector<int> resampled_indices(NUM_PARTICLES);
	cudaError_t err=cudaGetLastError();
	if (err != cudaSuccess) 
		printf("Error: %s\n", cudaGetErrorString(err));
	checkCudaErrors(cudaDeviceSynchronize());
	float *mapsave;
	cudaError_t status=cudaMallocHost(&mapsave, width*height*sizeof(float));
	if(status!=cudaSuccess)
		printf("error allocating pinned memory\n");
	size_t pitchSave=sizeof(float)*width;
	checkCudaErrors(cudaMemcpy2D(mapsave, pitchSave, map, pitch, width*sizeof(float), height, cudaMemcpyDeviceToHost));
	FILE *img;
	img=fopen("mapinit.dat", "w");
	if(img!=NULL){
		fwrite(mapsave, sizeof(float), width*height, img);
		/*int ptrIndex=0;
		for(ptrIndex=0; ptrIndex<width*height; ptrIndex++){
			float elem=mapsave[ptrIndex];
			fprintf(img, "%f", elem);
			if(ptrIndex%width==0 && ptrIndex!=0)
				fprintf(img, "\n");
			else
				fprintf(img, " ");
		}*/
	}
	cudaFreeHost(mapsave);
	fclose(img);

	/*checkCudaErrors(cudaBindTexture2D(0,sensor_model, s_m_gpu, local_size*((int)s_m_resolution), local_size*((int)s_m_resolution), pitch_s));
	checkCudaErrors(cudaBindTextureToArray(radius, r_gpu));
	checkCudaErrors(cudaBindTextureToArray(angle, a_gpu));
	*/
	/*loading the range readings from file*/
	/*FILE *f;
	f=fopen("fr079.log", "r");*/
	float ares=2*M_PI/360.0f;
	int numReadings=(int)(M_PI*2/ares);
	//float amin=-M_PI;
	float amin=0;
	float areadmin=0.0f;
	int astart=(int)((areadmin-amin)/ares);
	std::vector<int> numScans;
	std::vector<std::vector<float>> scans;
	std::vector<float> xs;
	std::vector<float> ys;
	std::vector<float> thetas;
	bool open=loadLog("fr079.log", numScans, scans, xs, ys, thetas);
	/*float *xs=(float*)malloc(sizeof(float));
	float *ys=(float*)malloc(sizeof(float));
	float *thetas=(float*)malloc(sizeof(float));
	int *numScans=(int*)malloc(sizeof(float));
	float **scans=(float**)malloc(sizeof(float*));
	*/
	//int iter=0;
	//int len=0;
	//if (f!=NULL){
	//	char *buffer=(char*)malloc(4096*sizeof(char));
	//	int line=0;
	//	while(fgets(buffer, 4096, f)){
	//		line++;
	//		int numElem=-1;
	//		sscanf(buffer, "FLASER %d", &numElem);
	//		if (numElem==-1){
	//			continue;
	//		}
	//		numElem+=11;
	//		char **a;
	//		char **res;
	//		res=new char* [numElem];
	//		for(a=res; (*a=mystrsep(&buffer, " "))!=NULL;){
	//			if(**a!='\0')
	//				if(++a>=&res[numElem])
	//					break;
	//		}
	//		int i, j;
	//		numScans[iter]=atoi(res[1]);
	//		float *readings_f=(float*)malloc(numReadings*sizeof(float));
	//		/*for(j=0; j<astart; j++){
	//			readings_f[j]=-1.0;
	//		}*/
	//		for(i=2; i<2+atoi(res[1]); i++){
	//			sscanf(res[i], "%f", &readings_f[i-2]);
	//			//readings_f[astart+i-2]*=100;
	//		}
	//		float x=(float)atof(res[i]);
	//		//float x=(float)atof(res[i])*10;
	//		xs[iter]=x;
	//		//float y=(float)atof(res[i+1])*10;
	//		float y=(float)atof(res[i+1]);
	//		ys[iter]=y;
	//		float theta=(float)atof(res[i+2]);
	//		thetas[iter]=theta;
	//		scans[iter]=readings_f;
	//		iter++;
	//		float *xs_new=(float*)realloc(xs, (iter+1)*sizeof(float));
	//		float *ys_new=(float*)realloc(ys, (iter+1)*sizeof(float));
	//		float *thetas_new=(float*)realloc(thetas, (iter+1)*sizeof(float));
	//		int *numScans_new=(int*)realloc(numScans, (iter+1)*sizeof(int));
	//		float **scans_new=(float**)realloc(scans, (iter+1)*sizeof(float*));
	//		if (xs_new!=NULL)
	//			xs=xs_new;
	//		else
	//			printf("no xs");
	//		if (ys_new!=NULL)
	//			ys=ys_new;
	//		else
	//			printf("no ys");
	//		if (thetas_new!=NULL)
	//			thetas=thetas_new;
	//		else
	//			printf("no thetas");
	//		if (scans_new!=NULL)
	//			scans=scans_new;
	//		if(numScans_new!=NULL)
	//			numScans=numScans_new;
	//		else
	//			printf("no scans");
	//		
	//		buffer=(char*)malloc(4096*sizeof(char));
	//	}
	//	xs=(float*)realloc(xs, iter*sizeof(float));
	//	ys=(float*)realloc(ys, iter*sizeof(float));
	//	thetas=(float*)realloc(thetas, iter*sizeof(float));
	//	numScans=(int*)realloc(numScans, iter*sizeof(int));
	//	scans=(float**)realloc(scans, iter*sizeof(float*));
	//	len=iter;
	//	/*int j;
	//	for(j=0; j<iter; j++){
	//		printf("xs:%f\t", xs[j]);
	//		printf("ys:%f\t", ys[j]);
	//		printf("thetas:%f\n", thetas[j]);
	//		int k;
	//		for(k=0; k<numReadings; k++){
	//			float * s=scans[j];
	//			printf("%f\t", s[k]);
	//		}
	//		printf("\n");
	//	}
	//	printf("lines read:%d\n", line);
	//	*/
	//}
	int index;
	float tot_time=0.0f;
	float x_old=xs[0];
	float y_old=ys[0];
	float theta_old=thetas[0];
	thrust::fill(x_part.begin(), x_part.end(), x_old);
	thrust::fill(y_part.begin(), y_part.end(), y_old);
	thrust::fill(theta_part.begin(), theta_part.end(), theta_old);
	float x_old_c=0.0f;
	float y_old_c=0.0f;
	float theta_old_c=0.0f;
	printf("%d, %d\n",numScans.size(), open==0);
    for(index=0; index<numScans.size(); index++){
		/*taking one range reading at a time*/
		cudaEvent_t start, stop, resample_time;
		cudaEvent_t startScores, stopScores;
		float time;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventCreate(&resample_time);
		cudaEventCreate(&startScores);
		cudaEventCreate(&stopScores);
		cudaEventRecord(start, 0);
        std::vector<float> scan=scans[index];
		float x_h;
		float y_h;
		float theta_h;
		/*checkCudaErrors(cudaMallocHost(&x_h, sizeof(float)));
		checkCudaErrors(cudaMallocHost(&y_h, sizeof(float)));
		checkCudaErrors(cudaMallocHost(&theta_h, sizeof(float)));*/
        x_h=xs[index];
        y_h=ys[index];
        theta_h=thetas[index];
		printf("position:%f %f %f\n", x_h, y_h, theta_h);
		float *scan_gpu;
		checkCudaErrors(cudaMalloc(&scan_gpu, sizeof(float)*numReadings));
		checkCudaErrors(cudaMemcpy(scan_gpu, &scan[0], numReadings*sizeof(float), cudaMemcpyHostToDevice));
		drawFromMotion(x_part, y_part, theta_part, x_h, y_h, theta_h, x_old, y_old, theta_old, (int)start);
		/*int numTU=32;
		int numBU=(int)ceil((float)local_size/numTU);
		printf("num blocks:%d\n", numBU);
		dim3 numThrU(numTU, numTU);
		dim3 numBlU(numBU, numBU);
		*/
		/*checkCudaErrors(cudaMemcpyToSymbol(x, &x_h, sizeof(float)));
		checkCudaErrors(cudaMemcpyToSymbol(y, &y_h, sizeof(float)));
		checkCudaErrors(cudaMemcpyToSymbol(theta, &theta_h, sizeof(float)));
		*/
		//updateMap<<<numBlU, numThrU>>>(x, y, theta*M_PI/180.0f, map, scan_gpu, pitch/sizeof(float), width, height, local_size);
		/*float delta_theta=atan2(sinf(y_h-y_old), cos(x_h-x_old));
		float s=sinf(theta_old);
		float c=cosf(theta_old);
		float delta_x=c*(x_h-x_old)+s*(y_h-y_old);
		float delta_y=-s*(x_h-x_old)+c*(y_h-y_old);
		float srr=0.01f;
		float sxy=0.3f*srr;
		float str=0.01;
		float srt=0.01;
		float stt=0.01;
		float sigma_x=srr*fabs(delta_x)+str*fabs(delta_theta)+sxy*fabs(delta_y);
		float sigma_y=srr*fabs(delta_y)+str*fabs(delta_theta)+sxy*fabs(delta_x);
		float sigma_theta=stt*fabs(delta_theta)+srt*hypot(delta_x, delta_y);
		*/
		//float delta_r1=angleDiff(atan2(y_h-y_old, x_h-x_old),theta_old);
		//float delta_r1_s=std::min(fabs(angleDiff(delta_r1, 0.0)), fabs(angleDiff(delta_r1, M_PI)));
		//float delta_t=hypot(x_h-x_old, y_h-y_old);
		//float delta_r2=angleDiff(angleDiff(theta_h,theta_old),delta_r1);
		//float delta_r2_s=std::min(fabs(angleDiff(delta_r2, 0.0)), fabs(angleDiff(delta_r2, M_PI)));
		//float sigma_t=ALPHA3*delta_t*delta_t+ALPHA4*(delta_r1_s*delta_r1_s+delta_r2_s*delta_r2_s);
		//float sigma_r1=ALPHA1*delta_r1_s*delta_r1_s+ALPHA2*delta_t*delta_t;
		//float sigma_r2=ALPHA1*delta_r2_s*delta_r2_s+ALPHA2*delta_t*delta_t;
		//thrust::counting_iterator<unsigned int> rndSeed((int)start);
		//thrust::transform(rndSeed, rndSeed+NUM_PARTICLES, delta_t_v.begin(), pseudorgnorm(0.0f, sigma_t));
		//thrust::transform(rndSeed+NUM_PARTICLES, rndSeed+NUM_PARTICLES*2, delta_r1_v.begin(), pseudorgnorm(0.0f, sigma_r1));
		//thrust::transform(rndSeed+NUM_PARTICLES*2, rndSeed+NUM_PARTICLES*3, delta_r2_v.begin(), pseudorgnorm(0.0f, sigma_r2));
		//thrust::constant_iterator<float> delta_t_const(delta_t);
		//thrust::constant_iterator<float> delta_r1_const(delta_r1);
		//thrust::constant_iterator<float> delta_r2_const(delta_r2);
		//thrust::transform(delta_t_const, delta_t_const+NUM_PARTICLES, delta_t_v.begin(), delta_t_v.begin(), thrust::minus<float>());
		//thrust::transform(delta_r1_const, delta_r1_const+NUM_PARTICLES, delta_r1_v.begin(), delta_r1_v.begin(), angleDiff_v());
		//thrust::transform(delta_r2_const, delta_r2_const+NUM_PARTICLES, delta_r2_v.begin(), delta_r2_v.begin(), angleDiff_v());
		////theta+delta_r1
		//thrust::transform(theta_part.begin(), theta_part.end(), delta_r1_v.begin(), delta_r1_v.begin(), thrust::plus<float>());
		////thrust::transform(delta_r1_v.begin(), delta_r1_v.end(), x_part.begin(), cos_v());
		////delta_t*cos(delta_r1+theta)
		//thrust::transform(delta_t_v.begin(), delta_t_v.end(), make_transform_iterator(delta_r1_v.begin(), cos_v<float>()), temp.begin(), thrust::multiplies<float>());
		////x+delta_t*cos(delta_r1+theta)
		//thrust::transform(temp.begin(), temp.end(), x_part.begin(), x_part.begin(), thrust::plus<float>());
		////thrust::constant_iterator<float> x_const(x_old_c);
		////thrust::transform(x_const, x_const+NUM_PARTICLES, x_part.begin(), x_part.begin(), thrust::plus<float>());
		////thrust::transform(delta_r1_v.begin(), delta_r1_v.end(), y_part.begin(), sin_v());
		////delta_t*sin(delta_r1+theta)
		//thrust::transform(delta_t_v.begin(), delta_t_v.end(), make_transform_iterator(delta_r1_v.begin(), sin_v<float>()), temp.begin(), thrust::multiplies<float>());
		////y+delta_t*sin(delta_r1+theta)
		//thrust::transform(temp.begin(), temp.end(), y_part.begin(), y_part.begin(), thrust::plus<float>());
		///*thrust::constant_iterator<float> y_const(y_old_c);
		//thrust::transform(y_const, y_const+NUM_PARTICLES, y_part.begin(), y_part.begin(), thrust::plus<float>());
		//*/
		////(theta+delta_r1)+delta_r2
		//thrust::transform(delta_r1_v.begin(), delta_r1_v.end(), delta_r2_v.begin(), theta_part.begin(), plusMod(M_PI*2));
		//
		//thrust::transform(delta_r1_v.begin(), delta_r1_v.end(), delta_r2_v.begin(), theta_part.begin(), thrust::plus<float>());
		/*thrust::constant_iterator<float> theta_const(theta_old_c);
		thrust::transform(theta_const, theta_const+NUM_PARTICLES, theta_part.begin(), theta_part.begin(), thrust::plus<float>());*/
		//thrust::transform(theta_const, theta_const+NUM_PARTICLES, delta_r1_v.begin(), delta_r1_v.begin(), thrust::plus<float>());
		/*
		thrust::device_vector<float> delta_x_v(NUM_PARTICLES);
		thrust::device_vector<float> delta_y_v(NUM_PARTICLES);
		thrust::device_vector<float> delta_theta_v(NUM_PARTICLES);
		thrust::transform(rndSeed, rndSeed+NUM_PARTICLES, delta_x_v.begin(), pseudorgnorm(delta_x, sigma_x));
		thrust::transform(rndSeed+NUM_PARTICLES, rndSeed+NUM_PARTICLES*2, delta_y_v.begin(), pseudorgnorm(delta_y, sigma_y));
		thrust::transform(rndSeed+NUM_PARTICLES*2, rndSeed+NUM_PARTICLES*3, delta_theta_v.begin(), pseudorgnorm(delta_theta, sigma_theta));
		thrust::transform(delta_theta_v.begin(), delta_theta_v.end(), delta_theta_v.begin(), correctAngle(2*M_PI));
		thrust::transform(x_part.begin(), x_part.end(), delta_x_v.begin(), x_part.begin(), thrust::plus<float>());
		thrust::transform(y_part.begin(), y_part.end(), delta_y_v.begin(), y_part.begin(), thrust::plus<float>());
		thrust::transform(theta_part.begin(), theta_part.end(), delta_theta_v.begin(), theta_part.begin(), thrust::plus<float>());
		*/
		//float delta_t=hypot(x_h-x_old, y_h-y_old);
		//float delta_r1;
		//if(delta_t<0.01f)
		//	delta_r1=0.0f;
		//else
		//	delta_r1=angleDiff(atan2(y_h-y_old, x_h-x_old),theta_old);
		//delta_r1=std::min(fabs(angleDiff(delta_r1, 0.0)), fabs(angleDiff(delta_r1, M_PI)));
		//float delta_r2=angleDiff(angleDiff(theta_h,theta_old),delta_r1);
		//delta_r2=std::min(fabs(angleDiff(delta_r2, 0.0)), fabs(angleDiff(delta_r2, M_PI)));
		//float sigma_t=ALPHA3*delta_t*delta_t+ALPHA4*(delta_r1*delta_r1+delta_r2*delta_r2);
		//float sigma_r1=ALPHA1*delta_r1*delta_r1+ALPHA2*delta_t*delta_t;
		//float sigma_r2=ALPHA1*delta_r2*delta_r2+ALPHA2*delta_t*delta_t;
		//printf("deltas: %f %f %f\n sigmas:%f %f %f\n", delta_t, delta_r1, delta_r2, sigma_t, sigma_r1, sigma_r2);
		//thrust::counting_iterator<unsigned int> rndSeed((int)start);
		//thrust::transform(rndSeed, rndSeed+NUM_PARTICLES, delta_t_v.begin(), pseudorgnorm(0.0f, sigma_t));
		//thrust::transform(rndSeed+NUM_PARTICLES, rndSeed+NUM_PARTICLES*2, delta_r1_v.begin(), pseudorgnorm(0.0f, sigma_r1));
		//thrust::transform(rndSeed+NUM_PARTICLES*2, rndSeed+NUM_PARTICLES*3, delta_r2_v.begin(), pseudorgnorm(0.0f, sigma_r2));
		//thrust::constant_iterator<float> delta_t_const(delta_t);
		//thrust::constant_iterator<float> delta_r1_const(delta_r1);
		//thrust::constant_iterator<float> delta_r2_const(delta_r2);
		//thrust::transform(delta_t_const, delta_t_const+NUM_PARTICLES, delta_t_v.begin(), delta_t_v.begin(), thrust::minus<float>());
		//thrust::transform(delta_r1_const, delta_r1_const+NUM_PARTICLES, delta_r1_v.begin(), delta_r1_v.begin(), angleDiff_v());
		//thrust::transform(delta_r2_const, delta_r2_const+NUM_PARTICLES, delta_r2_v.begin(), delta_r2_v.begin(), angleDiff_v());
		////theta+delta_r1
		//thrust::transform(theta_part.begin(), theta_part.end(), delta_r1_v.begin(), delta_r1_v.begin(), thrust::plus<float>());
		////thrust::transform(delta_r1_v.begin(), delta_r1_v.end(), x_part.begin(), cos_v());
		////delta_t*cos(delta_r1+theta)
		//thrust::transform(delta_t_v.begin(), delta_t_v.end(), make_transform_iterator(delta_r1_v.begin(), cos_v<float>()), temp.begin(), thrust::multiplies<float>());
		////x+delta_t*cos(delta_r1+theta)
		//thrust::transform(temp.begin(), temp.end(), x_part.begin(), x_part.begin(), thrust::plus<float>());
		////thrust::constant_iterator<float> x_const(x_old_c);
		////thrust::transform(x_const, x_const+NUM_PARTICLES, x_part.begin(), x_part.begin(), thrust::plus<float>());
		////thrust::transform(delta_r1_v.begin(), delta_r1_v.end(), y_part.begin(), sin_v());
		////delta_t*sin(delta_r1+theta)
		//thrust::transform(delta_t_v.begin(), delta_t_v.end(), make_transform_iterator(delta_r1_v.begin(), sin_v<float>()), temp.begin(), thrust::multiplies<float>());
		////y+delta_t*sin(delta_r1+theta)
		//thrust::transform(temp.begin(), temp.end(), y_part.begin(), y_part.begin(), thrust::plus<float>());
		///*thrust::constant_iterator<float> y_const(y_old_c);
		//thrust::transform(y_const, y_const+NUM_PARTICLES, y_part.begin(), y_part.begin(), thrust::plus<float>());
		//*/
		////(theta+delta_r1)+delta_r2
		//thrust::transform(delta_r1_v.begin(), delta_r1_v.end(), delta_r2_v.begin(), theta_part.begin(), plusMod(M_PI*2));
		///*thrust::constant_iterator<float> theta_const(theta_old_c);
		//thrust::transform(theta_const, theta_const+NUM_PARTICLES, theta_part.begin(), theta_part.begin(), thrust::plus<float>());*/
		////thrust::transform(theta_const, theta_const+NUM_PARTICLES, delta_r1_v.begin(), delta_r1_v.begin(), thrust::plus<float>());

		float * x_part_kernel=thrust::raw_pointer_cast(&x_part[0]);
		float * y_part_kernel=thrust::raw_pointer_cast(&y_part[0]);
		float * theta_part_kernel=thrust::raw_pointer_cast(&theta_part[0]);
		
		dim3 blocksScores(numScans[index], 256);
		//computeMatchScores<<<blocksScores, NUM_PARTICLES/PART_PER_THREAD >>>(x_part_kernel, y_part_kernel, theta_part_kernel, scan_gpu, map, pitch/sizeof(float), scanScores);
		computeMatchScores<<<NUM_PARTICLES, numScans[index] >>>(x_part_kernel, y_part_kernel, theta_part_kernel, scan_gpu, map, pitch/sizeof(float), scanScores);
		/*thrust::device_vector<float> weights(NUM_PARTICLES);
		thrust::device_vector<float> scoreIndexes(NUM_PARTICLES);
		thrust::reduce_by_key(thrust::make_transform_iterator(thrust::counting_iterator<int>(0), lin_to_row_index<int>(numScans[index])), thrust::make_transform_iterator(thrust::counting_iterator<int>(0)+numScans[index]*NUM_PARTICLES, lin_to_row_index<int>(numScans[index])), allScores, scoreIndexes.begin(), weights.begin(), thrust::equal_to<int>(), thrust::plus<float>());
		checkCudaErrors(cudaFree(scanScores));
		*/
		thrust::device_vector<float> nnzweights(NUM_PARTICLES);
		//thrust::transform(weights, weights+NUM_PARTICLES, stencil.begin(), diff_from_zero());
		thrust::copy_if(weights, weights+NUM_PARTICLES, nnzweights.begin(), diff_from_zero());
		float max_w=*(thrust::max_element(weights, weights+NUM_PARTICLES));
		printf("max val:%f\n", max_w);
		thrust::transform(weights, weights+NUM_PARTICLES, weights, score_to_weight(max_w));
		//thrust::constant_iterator<float> max_w_const(max_w);
		//thrust::transform(weights, weights+NUM_PARTICLES, max_w_const, weights, thrust::divides<float>());
		/*thrust::constant_iterator<float> one_const(1.0f);
		thrust::transform(one_const, one_const+NUM_PARTICLES, weights.begin(), weights.begin(), thrust::minus<float>());
		*/
		zipIteratorFloatTuple zipIter=thrust::make_zip_iterator(make_tuple(x_part.begin(), y_part.begin(), theta_part.begin()));
		//thrust::sort_by_key(weights, weights+NUM_PARTICLES, zipIter);
		thrust::inclusive_scan(weights, weights+NUM_PARTICLES, weights);
		thrust::transform(weights, weights+NUM_PARTICLES, thrust::make_constant_iterator(weights[NUM_PARTICLES-1]), weights, thrust::divides<float>());
		cudaEventRecord(resample_time, 0);
		thrust::counting_iterator<unsigned int> resampleSeed((unsigned int)resample_time);
		thrust::transform(resampleSeed, resampleSeed+NUM_PARTICLES, resampling_vector.begin(), pseudorg(0.0f, 1.0f));
		thrust::lower_bound(weights, weights+NUM_PARTICLES, resampling_vector.begin(), resampling_vector.end(), resampled_indices.begin());
		thrust::gather(resampled_indices.begin(), resampled_indices.end(), zipIter, zipIter);
		thrust::gather(resampled_indices.begin(), resampled_indices.end(), weights, weights);
		/*float x_avg=thrust::reduce(x_part.begin(), x_part.end());
		float y_avg=thrust::reduce(y_part.begin(), y_part.end());
		float theta_avg=thrust::reduce(theta_part.begin(), theta_part.end());
		x_avg/=NUM_PARTICLES;
		y_avg/=NUM_PARTICLES;
		theta_avg/=NUM_PARTICLES;
		*/
		thrust::device_ptr<float> max_ptr=thrust::max_element(weights, weights+NUM_PARTICLES);
		float x_avg=x_part[max_ptr-weights];
		float y_avg=y_part[max_ptr-weights];
		float theta_avg=theta_part[max_ptr-weights];
		printf("computed position: %f %f %f\n", x_avg, y_avg, theta_avg);
		printf("max position:%d\n", max_ptr-weights);
		updateMapBresenham<<<360, 256>>>(map, pitch/sizeof(float),scan_gpu, x_avg, y_avg, theta_avg);
		checkCudaErrors(cudaFree(scan_gpu));
		//checkCudaErrors(cudaFree(scanScores));
		/*checkCudaErrors(cudaFreeHost(x_h));
		checkCudaErrors(cudaFreeHost(y_h));
		checkCudaErrors(cudaFreeHost(theta_h));*/
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		tot_time+=time;
		cudaError_t err=cudaGetLastError();
		if (err != cudaSuccess){ 
			printf("Error: %s\n", cudaGetErrorString(err));
			return -1;
		}
		x_old=x_h;
		y_old=y_h;
		theta_old=theta_h;
		x_old_c=x_avg;
		y_old_c=y_avg;
		theta_old_c=theta_avg;
		if(index%100==0){
			float *mapsave;
			/*saving map at every iteration, just for testing purposes*/
			cudaMallocHost(&mapsave, width*height*sizeof(float));
			size_t pitchSave=sizeof(float)*width;
			checkCudaErrors(cudaMemcpy2D(mapsave, pitchSave, map, pitch, width*sizeof(float), height, cudaMemcpyDeviceToHost));
			FILE *img;
			char filename[40];
			sprintf(filename, "map%d.dat", index);
			img=fopen(filename, "wb");
			if(img!=NULL){
				fwrite(mapsave, sizeof(float), width*height, img);
				/*int ptrIndex=0;
				for(ptrIndex=0; ptrIndex<width*height; ptrIndex++){
					float elem=mapsave[ptrIndex];
					fprintf(img, "%f ", elem);
					if(ptrIndex%width==0 && ptrIndex!=0)
						fprintf(img, "\n");
					else
						fprintf(img, " ");
				}*/
			}
			cudaFreeHost(mapsave);
			fclose(img);
		}
    }
	/*unbinding textures and cleanup*/
	/*checkCudaErrors(cudaUnbindTexture(radius));
	checkCudaErrors(cudaUnbindTexture(angle));
	checkCudaErrors(cudaUnbindTexture(sensor_model));
	checkCudaErrors(cudaFreeArray(r_gpu));
	checkCudaErrors(cudaFreeArray(a_gpu));
	checkCudaErrors(cudaFree(s_m_gpu));
	free(r);
	free(a);
	free(s_m);
	*/
	float avg_time=tot_time/numScans.size();
	printf("avg time:%f\n", avg_time);
	getchar();
}
