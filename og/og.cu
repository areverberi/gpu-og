#include "og.h"

__global__ void initMapK(float* map, int w, int h, size_t pitch, int w_th, int h_th);
__global__ void computeMatchScoresK(float * x_part, float * y_part, float * theta_part, float * scan_gpu, float *map, size_t pitch, float * scores);
__global__ void updateMapBresenhamK(float *map, size_t pitch, float *scan_gpu, float x, float y, float theta);

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
void initMap(float * map, int w, int h, size_t pitch, int w_th, int h_th )
{
	dim3 numThr(32, 32);
	dim3 numBlocks(w/numThr.x, h/numThr.y);
    initMapK <<<numBlocks, numThr>>> (map, w, h, pitch/sizeof(float), 1, 1);
	
}
__global__ void initMapK(float* map, int w, int h, size_t pitch, int w_th, int h_th){
	unsigned int idx=blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int idy=blockIdx.y*blockDim.y+threadIdx.y;
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
		x1=(int)floorf(mapW/2+*x_o/resolution);
		y1=(int)floorf(mapH/2+*y_o/resolution);
		//0.1f for wall thickness, if needed, add to range before mul
		x2=(int)floorf(mapW/2+(*x_o +(*range+0.1f)*c)/resolution);
		y2=(int)floorf(mapH/2+(*y_o +(*range+0.1f)*s)/resolution);
		delta_x=(float)(x2-x1);
		delta_y=(float)(y2-y1);
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
		}
		else
		{
			m=delta_y/delta_x;
			current_x=x1+sign_delta_x*pos;
			current_y=y1+rintf(m*(current_x-x1));
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
void computeMatchScores(thrust::device_vector<float> & x_part, thrust::device_vector<float> & y_part, thrust::device_vector<float> & theta_part, float * scan_gpu, float * map, size_t pitch, float * scores, int numScans)
{
	float * x_part_kernel=thrust::raw_pointer_cast(&x_part[0]);
	float * y_part_kernel=thrust::raw_pointer_cast(&y_part[0]);
	float * theta_part_kernel=thrust::raw_pointer_cast(&theta_part[0]);
	computeMatchScoresK<<<NUM_PARTICLES, numScans >>>(x_part_kernel, y_part_kernel, theta_part_kernel, scan_gpu, map, pitch, scores);
}
__global__ void computeMatchScoresK(float * x_part, float * y_part, float * theta_part, float * scan_gpu, float *map, size_t pitch, float * scores)
{
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
	atomicAdd(&score, pt_score);
	__syncthreads();
	if(threadIdx.x==0)
	{
		scores[blockIdx.x]=(float)score;
	}
}
void updateMapBresenham(float *map, size_t pitch, float * scan_gpu, float x, float y, float theta, int numBeams)
{
	updateMapBresenhamK<<<numBeams, 256>>>(map, pitch, scan_gpu, x, y, theta);
}
__global__ void updateMapBresenhamK(float *map, size_t pitch, float *scan_gpu, float x, float y, float theta){
	__shared__ float range;
	float coords[3];
	if(threadIdx.x==0)
	{
		range=scan_gpu[blockIdx.x];
	}
	getCoordsBresenham(coords, &range, &x, &y, &theta);
	if(coords[2]>=0)
	{
		float d=coords[2];
		int current_x=(int)coords[0];
		int current_y=(int)coords[1];
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
				
		if (d<=range+0.1f && d<=6.4f)
		{
			float pr=map[current_x+current_y*pitch];
			if(pr==-1.0f)
				pr=0.5f;
			map[current_x+current_y*pitch]=1.0f-1.0f/(1.0f+prob/(1.0f-prob)*pr/(1.0f-pr));

		}
	}
}
void init(float * map, size_t * pitch, int * width, int * height, thrust::device_vector<float> & x_part, thrust::device_vector<float> & y_part, thrust::device_vector<float> & theta_part, std::vector<int> & numScans, std::vector<std::vector<float>> & scans, std::vector<float> & xs, std::vector<float> & ys, std::vector<float> & thetas, float * x_old, float * y_old, float * theta_old)
{
	float res=0.05f;
	float rmax=50.0f;
	check_cuda_error(cudaMemcpyToSymbol(mapW, width, sizeof(int)));
	check_cuda_error(cudaMemcpyToSymbol(mapH, height, sizeof(int)));
	check_cuda_error(cudaMemcpyToSymbol(resolution, &res, sizeof(float)));
	check_cuda_error(cudaMemcpyToSymbol(range_max, &rmax, sizeof(float)));
	dim3 numThr(32, 32);
	dim3 numBlocks(*width/numThr.x, *height/numThr.y);
    initMap(map, *width, *height, *pitch, 1, 1);
	bool open=loadLog("fr079.log", numScans, scans, xs, ys, thetas);
	*x_old=xs[0];
	*y_old=ys[0];
	*theta_old=thetas[0];
	thrust::fill(x_part.begin(), x_part.end(), *x_old);
	thrust::fill(y_part.begin(), y_part.end(), *y_old);
	thrust::fill(theta_part.begin(), theta_part.end(), *theta_old);
}
float * get_map( float *map_gpu, int width, int height, size_t pitch)
{
	float *mapsave;
	check_cuda_error(cudaMallocHost(&mapsave, width*height*sizeof(float)));
	size_t pitchSave=sizeof(float)*width;
	check_cuda_error(cudaMemcpy2D(mapsave, pitchSave, map_gpu, pitch, width*sizeof(float), height, cudaMemcpyDeviceToHost));
	return mapsave;
}
void save_map(float * map_gpu, int width, int height, size_t pitch, char * filename)
{
	float *mapsave;
	mapsave=get_map(map_gpu, width, height, pitch);
	FILE *img;
	img=fopen(filename, "wb");
	if(img!=NULL){
		fwrite(mapsave, sizeof(float), width*height, img);
	}
	cudaFreeHost(mapsave);
	fclose(img);
}
int resample(thrust::device_vector<float> & x_part, thrust::device_vector<float> & y_part, thrust::device_vector<float> & theta_part, thrust::device_ptr<float> & weights, thrust::device_vector<float> & resampling_vector, thrust::device_vector<int> & resampled_indices)
{
	cudaEvent_t resample_time;
	check_cuda_error(cudaEventCreate(&resample_time));
	float max_w=*(thrust::max_element(weights, weights+NUM_PARTICLES));
	thrust::transform(weights, weights+NUM_PARTICLES, weights, score_to_weight(max_w));
	zipIteratorFloatTuple zipIter=thrust::make_zip_iterator(make_tuple(x_part.begin(), y_part.begin(), theta_part.begin()));
	thrust::inclusive_scan(weights, weights+NUM_PARTICLES, weights);
	thrust::transform(weights, weights+NUM_PARTICLES, thrust::make_constant_iterator(weights[NUM_PARTICLES-1]), weights, thrust::divides<float>());
	check_cuda_error(cudaEventRecord(resample_time, 0));
	thrust::counting_iterator<unsigned int> resampleSeed((unsigned int)resample_time);
	thrust::transform(resampleSeed, resampleSeed+NUM_PARTICLES, resampling_vector.begin(), pseudorg(0.0f, 1.0f));
	thrust::lower_bound(weights, weights+NUM_PARTICLES, resampling_vector.begin(), resampling_vector.end(), resampled_indices.begin());
	thrust::gather(resampled_indices.begin(), resampled_indices.end(), zipIter, zipIter);
	thrust::gather(resampled_indices.begin(), resampled_indices.end(), weights, weights);
	thrust::device_ptr<float> max_ptr=thrust::max_element(weights, weights+NUM_PARTICLES);
	return max_ptr-weights;
}

void run()
{
	int map_size=1600;
	int width=map_size;
    int height=map_size;
    float* map;
    size_t pitch;
	check_cuda_error(cudaMallocPitch(&map,&pitch,width*sizeof(float), height));
	thrust::device_vector<float> x_part(NUM_PARTICLES);
	thrust::device_vector<float> y_part(NUM_PARTICLES);
	thrust::device_vector<float> theta_part(NUM_PARTICLES);
	float * scanScores;
	check_cuda_error(cudaMalloc(&scanScores, NUM_PARTICLES*sizeof(float)));
	thrust::device_ptr<float> weights(scanScores);
	thrust::device_vector<float> resampling_vector(NUM_PARTICLES);
	thrust::device_vector<int> resampled_indices(NUM_PARTICLES);
	std::vector<int> numScans;
	std::vector<std::vector<float>> scans;
	std::vector<float> xs, ys, thetas;
	float x_old, y_old, theta_old;
	init(map, &pitch, &width, &height, x_part, y_part, theta_part, numScans, scans, xs, ys, thetas, &x_old, &y_old, &theta_old);
	check_cuda_error(cudaGetLastError());
	save_map(map, width, height, pitch, "mapinit.dat");
	float ares=2*M_PI/360.0f;
	int numReadings=(int)(M_PI*2/ares);
	
	int index;
	float tot_time=0.0f;
	for(index=0; index<numScans.size(); index++){
		/*taking one range reading at a time*/
		cudaEvent_t start, stop;
		cudaEvent_t startScores, stopScores;
		float time;
		check_cuda_error(cudaEventCreate(&start));
		check_cuda_error(cudaEventCreate(&stop));
		check_cuda_error(cudaEventCreate(&startScores));
		check_cuda_error(cudaEventCreate(&stopScores));
		check_cuda_error(cudaEventRecord(start, 0));
        std::vector<float> scan=scans[index];
		float x_h;
		float y_h;
		float theta_h;
		printf("%d\n", index);
		x_h=xs[index];
        y_h=ys[index];
        theta_h=thetas[index];
		printf("position:%f %f %f\n", x_h, y_h, theta_h);
		float *scan_gpu;
		check_cuda_error(cudaMalloc(&scan_gpu, sizeof(float)*numReadings));
		check_cuda_error(cudaMemcpy(scan_gpu, &scan[0], numReadings*sizeof(float), cudaMemcpyHostToDevice));
		drawFromMotion(x_part, y_part, theta_part, x_h, y_h, theta_h, x_old, y_old, theta_old, (int)start);
		computeMatchScores(x_part, y_part, theta_part, scan_gpu, map, pitch/sizeof(float), scanScores, numScans[index]);
		check_cuda_error(cudaGetLastError());
		int best_pos=resample(x_part, y_part, theta_part, weights, resampling_vector, resampled_indices);
		float x_avg=x_part[best_pos];
		float y_avg=y_part[best_pos];
		float theta_avg=theta_part[best_pos];
		printf("computed position: %f %f %f\n", x_avg, y_avg, theta_avg);
		updateMapBresenham(map, pitch/sizeof(float),scan_gpu, x_avg, y_avg, theta_avg, numScans[index]);
		check_cuda_error(cudaGetLastError());
		check_cuda_error(cudaFree(scan_gpu));
		check_cuda_error(cudaEventRecord(stop, 0));
		check_cuda_error(cudaEventSynchronize(stop));
		check_cuda_error(cudaEventElapsedTime(&time, start, stop));
		check_cuda_error(cudaEventDestroy(start));
		check_cuda_error(cudaEventDestroy(stop));
		tot_time+=time;
		check_cuda_error(cudaGetLastError());
		x_old=x_h;
		y_old=y_h;
		theta_old=theta_h;
		if(index%100==0){
			char filename[40];
			sprintf(filename, "map%d.dat", index);
			save_map(map, width, height, pitch, filename);
		}
    }
	
	float avg_time=tot_time/numScans.size();
	printf("avg time:%f\n", avg_time);
	check_cuda_error(cudaFree(map));
	thrust::device_free(weights);
	getchar();
}