#include <stdlib.h>
#include <stdio.h>
#include "string.h"
#include <stddef.h>
#define _USE_MATH_DEFINES
#include "math.h"
#include "helper_cuda.h"

texture <float, 2, cudaReadModeElementType> radius;
texture <float, 2, cudaReadModeElementType> angle;
texture <float, 2, cudaReadModeElementType> sensor_model;

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
__global__ void __launch_bounds__(1024) initMap(float* map, int w, int h, size_t pitch, int numX, int numY){
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	int idy=blockIdx.y*blockDim.y+threadIdx.y;
	if(idx<h && idy<w){
		map[idx*pitch+idy]=-1.0f;
	}
    __syncthreads();
}
__global__ void updateMapBresenham(float x, float y, float theta, float *map, size_t pitch, float *scan_gpu, int mapW, int mapH, float rmax){
	__shared__ float range;
	__shared__ int x1, y1, x2, y2;
	__shared__ float delta_x, delta_y, m;
	__shared__ int sign_delta_x, sign_delta_y;

	if(threadIdx.x==0)
	{
		range=scan_gpu[blockIdx.x];
		theta+=blockIdx.x*M_PI/359-M_PI_2;
		//mapW/H is offset, 0.1f is resolution
		x1=(int)floor(+mapW/2+x/0.025f);
		y1=(int)floor(+mapH/2+y/0.025f);
		//0.1f for wall thickness, if needed, add to range before mul
		x2=(int)floor(+mapW/2+x+((range+0.1f)*cos(theta))/0.025f);
		y2=(int)floor(+mapH/2+y+((range+0.1f)*sin(theta))/0.025f);
		delta_x=(float)(x2-x1);
		delta_y=(float)(y2-y1);
		sign_delta_x=1;
		if(delta_x<0)sign_delta_x=-1;
		sign_delta_y=1;
		if(delta_y<0)sign_delta_y=-1;
		//sign_delta_x=copysignf(1, delta_x);
		//sign_delta_y=copysignf(1, delta_y);
	}
	__syncthreads();
	if(range<50.0)
	{
		int current_x, current_y;
		if(fabs(delta_y)>fabs(delta_x))
		{
			m=delta_x/delta_y;
			current_y=y1+sign_delta_y*threadIdx.x;
			//current_x=x1+rintf(m*(current_y-y1));
            current_x=x1+floorf(0.4999999f+m*(current_y-y1));
		}
		else
		{
			m=delta_y/delta_x;
			current_x=x1+sign_delta_x*threadIdx.x;
			//current_y=y1+rintf(m*(current_x-x1));
            current_y=y1+floorf(0.4999999f+m*(current_x-x1));
		}
		if(current_x>=0 && current_x<mapW && current_y>=0 && current_y<mapH)
		{
			//0.1f because going from grid (10cm cell) to meters
			float d=hypotf(current_x-x1, current_y-y1)*0.025f;
			//divide by 100 because rmax is #of cells, ie 500->turn to meters
			//float k=1-(d/rmax)*(d/rmax)/100;
			//float k=1;
            float k=0.6;
			//float s=0.00001425*range*range;
			//float s=0.4;
            float s=0.6;
			float expon=((d-range)/s)*((d-range)/s);
			float prob;

			if (d<0.025f*256||true)
			{
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
		}
		else
		{
			//printf("%d %d\n", current_x, current_y); 
		}
	}
	else
	{
		//printf("range: %d \n", range);
	}
}
__global__ void __launch_bounds__(1024) updateMap(float x, float y, float theta, float* map, float* scan_gpu, size_t pitch, int mapW, int mapH, float rmax){
    __shared__ float scan[360];
	/*first 360 threads load scan*/
	unsigned int scanperthread=360/(blockDim.x*blockDim.y);
	if(scanperthread>1){
		unsigned int ind=(threadIdx.x*blockDim.x+threadIdx.y)*scanperthread;
		unsigned int off;
		for(off=0; off<scanperthread; off++){
			if(ind+off<360)
				scan[ind+off]=scan_gpu[ind+off];
		}
	}
	else{
		unsigned int ind=threadIdx.x*blockDim.x+threadIdx.y;
		if(ind<360){
			scan[ind]=scan_gpu[ind];
		}
	}
	//printf("scan loaded\n");
	__syncthreads();
	float x_local_lu=(blockIdx.x*blockDim.x+threadIdx.x)*1.0/(gridDim.x*blockDim.x)*rmax;
    float y_local_lu=(blockIdx.y*blockDim.y+threadIdx.y)*1.0/(gridDim.y*blockDim.y)*rmax;
	/*to fix: the 10.0 should be s_m_resolution*/
	//float val=tex2D(sensor_model, tex2D(radius, x_local_lu, y_local_lu)*10.0, scan[(int)rint(tex2D(angle, x_local_lu, y_local_lu))]*10.0);
	float val=tex2D(sensor_model, scan[(int)rint(tex2D(angle, x_local_lu, y_local_lu))]*10.0, tex2D(radius, x_local_lu, y_local_lu)*10.0);
	//if(tex2D(radius, x_local_lu, y_local_lu)>scan[(int)rint(tex2D(angle, x_local_lu, y_local_lu))])
	//	printf("val:%f\n", val);
	//printf("angle:%d\n", (int)rint(tex2D(angle, x_local_lu, y_local_lu)));
	//printf("val:%f\n", val);
	if (val!=0.5f)
	{
		float x_local=x_local_lu-rmax/2;
		float y_local=rmax/2-y_local_lu;
		x_local=x_local*__cosf(theta)-y_local*__sinf(theta);
		y_local=x_local*__sinf(theta)+y_local*__cosf(theta);
		//int x_map=(int)rint(x_local*cosf(theta)+y_local*sinf(theta)-x+mapW/2.0);
		//int y_map=(int)rint(-x_local*sinf(theta)+y_local*cosf(theta)-y+mapH/2.0);
		int x_map_cell=(int)rint(x_local+x*10.0f+mapW/2.0);
		int y_map_cell=(int)rint(-(y_local+y*10.0f-mapH/2.0));
		/*if(x_map_cell<0 || y_map_cell<0)
		printf("%f %f\n", x_map_cell, y_map_cell);
		*/if(x_map_cell<mapH && y_map_cell<mapW ){
			//no size difference between local and global cells, otherwise you'd need to divide by global cell size to get map cell
			//int x_map_cell=(int)rint(x_map);
			//int y_map_cell=(int)rint(y_map);
			if(scan[(int)rint(tex2D(angle, x_local_lu, y_local_lu))]>0.0)
			{
				size_t index=x_map_cell*pitch+y_map_cell;
				//map[index]=0.5f*val+0.5f*map[index];
				map[index]=1-1/(1+map[index]/(1-map[index])*val/(1-val));
			}
			/*
			if(scan[(int)rint(tex2D(angle, x_local_lu, y_local_lu))]>0.0){}
			if(map[index]<0.0f)
			map[index]=val;
			else
			map[index]=0.5f*val+0.5f*map[index];
			*/
		} 
	}
    __syncthreads();
}

int main(int argc, char** argv){
	float *r;
	float *a;
	float *s_m;
	/*size of the matrix in cells*/
	int local_size=500;
	//int map_size=1000;
    int map_size=1600;
	float cell_dim=0.1;
	//int map_size_x=1600;
	//int map_size_y=880;
	float s_m_resolution=10.0;
	r=(float*)malloc(sizeof(float)*local_size*local_size);
	a=(float*)malloc(sizeof(float)*local_size*local_size);
	s_m=(float*)malloc(sizeof(float)*local_size*local_size*(int)(s_m_resolution*s_m_resolution));
	int loopX=0;
	int loopY=0;
	/*initialization of lookups for radius, angle and sensor model*/
	for(loopY=0; loopY<local_size*s_m_resolution; loopY++){
		for(loopX=0; loopX<local_size*s_m_resolution; loopX++){
			if(loopX<local_size && loopY<local_size){
				float x_cell=loopX*cell_dim+cell_dim/2.0f-local_size*cell_dim/2.0f;
				float y_cell=-loopY*cell_dim+cell_dim/2.0f+local_size*cell_dim/2.0f;
				r[loopY*local_size+loopX]=hypotf(x_cell, y_cell);
				a[loopY*local_size+loopX]=(atan2(y_cell, x_cell)+M_PI)/M_PI*180.0f;
				//a[loopY*local_size+loopX]=atan2(y_cell, x_cell)/M_PI*180.0f;
			}
			if (abs(loopX-loopY)<s_m_resolution/2.0){
				s_m[loopY*local_size*((int)s_m_resolution)+loopX]=0.95f;
			}
			else{
				if (loopY<loopX){
					s_m[loopY*local_size*((int)s_m_resolution)+loopX]=0.05f;
				}
				else{ 
					/*
					float min=(local_size*10<loopX+s_m_resolution?local_size*10:loopX+s_m_resolution);
					if (loopY> min){
						s_m[loopX*local_size*10+loopY]=0.5f;
					}
					*/
					s_m[loopY*local_size*((int)s_m_resolution)+loopX]=0.5f;
				}
			}
		}
	}
	/*setting filter mode for the textures. It's linear for radius and angle so I get interpolation "for free"*/
	//printf("radius 0:%f\n", r[0]);
	//printf("angle 0:%f\n", a[0]);
	//getchar();
	sensor_model.filterMode=cudaFilterModePoint;
	radius.filterMode=cudaFilterModeLinear;
	angle.filterMode=cudaFilterModeLinear;	

	/*creating the cudaArrays that will contain the textures*/
	cudaChannelFormatDesc cf=cudaCreateChannelDesc<float>();
	cudaArray *r_gpu;
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
	/*map initialization and texture binding*/
    int width=map_size;
    int height=map_size;
    float* map;
    size_t pitch;
	checkCudaErrors(cudaMallocPitch(&map,&pitch,width*sizeof(float), height));
	int numT=32;
	int numBX=(int)ceil((float)width/numT);
	int numBY=(int)ceil((float)height/numT);
	dim3 numBlocks(numBX, numBY);
    dim3 numThr(numT, numT);
    initMap <<<numBlocks, numThr>>> (map, width, height, pitch/sizeof(float), 1, 1);
	cudaError_t err=cudaGetLastError();
	if (err != cudaSuccess) 
		printf("Error: %s\n", cudaGetErrorString(err));
	checkCudaErrors(cudaDeviceSynchronize());
	float *mapsave;
	mapsave=(float*)calloc(width*height,sizeof(float));
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
	free(mapsave);
	fclose(img);

	checkCudaErrors(cudaBindTexture2D(0,sensor_model, s_m_gpu, local_size*((int)s_m_resolution), local_size*((int)s_m_resolution), pitch_s));
	checkCudaErrors(cudaBindTextureToArray(radius, r_gpu));
	checkCudaErrors(cudaBindTextureToArray(angle, a_gpu));
	
	/*loading the range readings from file*/
	FILE *f;
	f=fopen("fr079-sm.log", "r");
	float ares=2*M_PI/360.0f;
	int numReadings=(int)(M_PI*2/ares);
	//float amin=-M_PI;
	float amin=0;
	float areadmin=0.0f;
	int astart=(int)((areadmin-amin)/ares);
	float *xs=(float*)malloc(sizeof(float));
	float *ys=(float*)malloc(sizeof(float));
	float *thetas=(float*)malloc(sizeof(float));
	int *numScans=(int*)malloc(sizeof(float));
	float **scans=(float**)malloc(sizeof(float*));
	int iter=0;
	int len=0;
	if (f!=NULL){
		char *buffer=(char*)malloc(4096*sizeof(char));
		int line=0;
		while(fgets(buffer, 4096, f)){
			line++;
			int numElem=-1;
			sscanf(buffer, "FLASER %d", &numElem);
			if (numElem==-1){
				continue;
			}
			numElem+=11;
			char **a;
			char **res;
			res=new char* [numElem];
			for(a=res; (*a=mystrsep(&buffer, " "))!=NULL;){
				if(**a!='\0')
					if(++a>=&res[numElem])
						break;
			}
			int i, j;
			numScans[iter]=atoi(res[1]);
			float *readings_f=(float*)malloc(numReadings*sizeof(float));
			for(j=0; j<astart; j++){
				readings_f[j]=-1.0;
			}
			for(i=2; i<2+atoi(res[1]); i++){
				sscanf(res[i], "%f", &readings_f[astart+i-2]);
				//readings_f[astart+i-2]*=100;
			}
			float x=(float)atof(res[i]);
			//float x=(float)atof(res[i])*10;
			xs[iter]=x;
			//float y=(float)atof(res[i+1])*10;
			float y=(float)atof(res[i+1]);
			ys[iter]=y;
			float theta=(float)atof(res[i+2]);
			thetas[iter]=theta;
			scans[iter]=readings_f;
			iter++;
			float *xs_new=(float*)realloc(xs, (iter+1)*sizeof(float));
			float *ys_new=(float*)realloc(ys, (iter+1)*sizeof(float));
			float *thetas_new=(float*)realloc(thetas, (iter+1)*sizeof(float));
			int *numScans_new=(int*)realloc(numScans, (iter+1)*sizeof(int));
			float **scans_new=(float**)realloc(scans, (iter+1)*sizeof(float*));
			if (xs_new!=NULL)
				xs=xs_new;
			else
				printf("no xs");
			if (ys_new!=NULL)
				ys=ys_new;
			else
				printf("no ys");
			if (thetas_new!=NULL)
				thetas=thetas_new;
			else
				printf("no thetas");
			if (scans_new!=NULL)
				scans=scans_new;
			if(numScans_new!=NULL)
				numScans=numScans_new;
			else
				printf("no scans");
			
			buffer=(char*)malloc(4096*sizeof(char));
		}
		xs=(float*)realloc(xs, iter*sizeof(float));
		ys=(float*)realloc(ys, iter*sizeof(float));
		thetas=(float*)realloc(thetas, iter*sizeof(float));
		numScans=(int*)realloc(numScans, iter*sizeof(int));
		scans=(float**)realloc(scans, iter*sizeof(float*));
		len=iter;
		/*int j;
		for(j=0; j<iter; j++){
			printf("xs:%f\t", xs[j]);
			printf("ys:%f\t", ys[j]);
			printf("thetas:%f\n", thetas[j]);
			int k;
			for(k=0; k<numReadings; k++){
				float * s=scans[j];
				printf("%f\t", s[k]);
			}
			printf("\n");
		}
		printf("lines read:%d\n", line);
		*/
	}
	int index;
	float tot_time=0.0f;
    for(index=0; index<len; index++){
		/*taking one range reading at a time*/
        float* scan=scans[index];
        float x=xs[index];
        float y=ys[index];
        float theta=thetas[index];
		printf("position:%f %f %f\n", x, y, theta);
		float *scan_gpu;
		checkCudaErrors(cudaMalloc(&scan_gpu, sizeof(float)*numReadings));
		checkCudaErrors(cudaMemcpy(scan_gpu, scan, numReadings*sizeof(float), cudaMemcpyHostToDevice));
		int numTU=32;
		int numBU=(int)ceil((float)local_size/numTU);
		printf("num blocks:%d\n", numBU);
		dim3 numThrU(numTU, numTU);
		dim3 numBlU(numBU, numBU);
		cudaEvent_t start, stop;
		float time;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
		//updateMap<<<numBlU, numThrU>>>(x, y, theta*M_PI/180.0f, map, scan_gpu, pitch/sizeof(float), width, height, local_size);
		updateMapBresenham<<<360, 256>>>(x, y, theta, map, pitch/sizeof(float),scan_gpu, width, height, local_size);
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
		checkCudaErrors(cudaFree(scan_gpu));
		
		if(index%100==0||index%10==0){
			float *mapsave;
			/*saving map at every iteration, just for testing purposes*/
			mapsave=(float*)calloc(width*height,sizeof(float));
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
			free(mapsave);
			fclose(img);
		}
    }
	/*unbinding textures and cleanup*/
	checkCudaErrors(cudaUnbindTexture(radius));
	checkCudaErrors(cudaUnbindTexture(angle));
	checkCudaErrors(cudaUnbindTexture(sensor_model));
	checkCudaErrors(cudaFreeArray(r_gpu));
	checkCudaErrors(cudaFreeArray(a_gpu));
	checkCudaErrors(cudaFree(s_m_gpu));
	free(r);
	free(a);
	free(s_m);
	float avg_time=tot_time/len;
	printf("avg time:%f\n", avg_time);
	getchar();
}
