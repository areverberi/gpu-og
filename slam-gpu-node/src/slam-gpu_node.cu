#include <ros/ros.h>
#include "og.h"
#include <sensor_msgs/LaserScan.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/OccupancyGrid.h>
#include <tf/transform_broadcaster.h>

class FilterState{
public:
    FilterState(float * scanScores): x_part(NUM_PARTICLES), y_part(NUM_PARTICLES), theta_part(NUM_PARTICLES), resampling_vector(NUM_PARTICLES), resampled_indices(NUM_PARTICLES), weights(scanScores){}
    void drawFromMotion(float x_new, float y_new, float theta_new, float x_old, float y_old, float theta_old, int seed)
    {
        slamgpu::drawFromMotion(x_part, y_part, theta_part, x_new, y_new, theta_new, x_old, y_old, theta_old, seed);
    }
    void computeMatchScores(float *scan_gpu, float *map, size_t pitch, int numScans)
    {
        slamgpu::computeMatchScores(x_part, y_part, theta_part, scan_gpu, map, pitch/sizeof(float), thrust::raw_pointer_cast(&weights[0]), numScans);
    }

    void resample(float * best_x, float * best_y, float * best_theta)
    {
        int pos=slamgpu::resample(x_part, y_part, theta_part, weights, resampling_vector, resampled_indices);
        *best_x=x_part[pos];
        *best_y=y_part[pos];
        *best_theta=theta_part[pos];
    }
    void initParticles(float x, float y, float theta)
    {
        thrust::fill(x_part.begin(), x_part.end(), x);
        thrust::fill(y_part.begin(), y_part.end(), y);
        thrust::fill(theta_part.begin(), theta_part.end(), theta);
    }

protected:
    thrust::device_vector<float> x_part;
    thrust::device_vector<float> y_part;
    thrust::device_vector<float> theta_part;
    thrust::device_vector<float> resampling_vector;
    thrust::device_vector<int> resampled_indices;
    thrust::device_ptr<float> weights;
};

class RobotState{
public:
    RobotState(ros::NodeHandle & _nh):nh(_nh), x_old(0.0), y_old(0.0), theta_old(0.0), x_best(0.0), y_best(0.0), theta_best(0.0), index(0), m_to_o(tf::createQuaternionFromYaw(0), tf::Point(0,0,0)){
        laserS=nh.subscribe("/scan", 1, &RobotState::laserCB, this);
        odomS=nh.subscribe("/odom", 1, &RobotState::odoCB, this);
        //odom_pub=nh.advertise<nav_msgs::Odometry>("/slam_gpu/odom", 1);
	map_pub=nh.advertise<nav_msgs::OccupancyGrid>("/slam_gpu/map", 50);
        res=0.05f;
        map_size=1600;
        width=map_size;
        height=map_size;
        check_cuda_error(cudaMallocPitch(&map,&pitch,width*sizeof(float), height));
	//printf("malloced map\n");
        slamgpu::setMapParams(width, height, res);
	//printf("memcpyd map params\n");
        slamgpu::initMap(map, width, height, pitch, 1, 1);
	//printf("init map\n");
        check_cuda_error(cudaMalloc(&scanScores, NUM_PARTICLES*sizeof(float)));
	//printf("malloced scores\n");
        first_o=true;
        first_s=true;
        f=new FilterState(scanScores);
    }
    void laserCB(const sensor_msgs::LaserScan::ConstPtr & s)
    {
	ros::Time start=ros::Time::now();
        float a_min=s->angle_min;
        float a_max=s->angle_max;
        float a_incr=s->angle_increment;
	printf("a_incr:%f\n", a_incr);
        int numScans=(int)((a_max-a_min)/a_incr);
        float a_range=a_max-a_min;
        float r_max=s->range_max;
        if(first_s)
        {
            slamgpu::setScanParams(a_min, a_range, numScans, r_max);
        }
        float *scans=new float[numScans];
	
        //score computation and resampling, also broadcast of odom msg and tf transform
        float *scan_gpu;
        check_cuda_error(cudaMalloc(&scan_gpu, sizeof(float)*numScans));
	printf("malloced scan\n");
        check_cuda_error(cudaMemcpy(scan_gpu, &(s->ranges[0]), numScans*sizeof(float), cudaMemcpyHostToDevice));
	printf("memcpyd scores\n");
        f->computeMatchScores(scan_gpu, map, pitch, numScans);
        check_cuda_error(cudaGetLastError());
	printf("computed scores\n");
        f->resample(&x_best, &y_best, &theta_best);
	printf("resampled\n");
        slamgpu::updateMapBresenham(map, pitch/sizeof(float),scan_gpu, x_best, y_best, theta_best, numScans);
	printf("updated map\n");
	int8_t * mapout_d;
	int8_t * mapout_h;
	size_t pitch_o;
	check_cuda_error(cudaMallocPitch(&mapout_d, &pitch_o, width*sizeof(float), height));
	printf("malloced outmap\n");
	slamgpu::postProcMap(map, mapout_d, width, height, pitch, pitch_o, 1, 1);
	printf("postprocced outmap\n");
	mapout_h=slamgpu::get_map(mapout_d, width, height, pitch_o);
	printf("memcpyd outmap\n");
	/*
	if(index%100==0)
	{
		char filename[40];
		sprintf(filename, "~/maps/map%d.dat", index);
		slamgpu::save_map(map, width, height, pitch, filename);
	}
	*/
	check_cuda_error(cudaFree(mapout_d));
	printf("freed outmap\n");
	check_cuda_error(cudaFree(scan_gpu));
	printf("freed scan\n");
        ros::Time current_time=ros::Time::now();
	nav_msgs::OccupancyGrid map_m;
	map_m.header.stamp=current_time;
	map_m.header.frame_id="/slam_gpu/map";
	map_m.data.insert(map_m.data.begin(), mapout_h, mapout_h+width*height);
	map_m.info.width=width;
	map_m.info.height=height;
	map_m.info.resolution=res;
	if(first_s)
	{
		x_center=x_old;
		y_center=-y_old;
		theta_center=theta_old;
		first_s=false;
	}
	map_m.info.origin.position.x=-width/2*res+x_center;
	map_m.info.origin.position.y=-height/2*res+y_center;
	map_m.info.origin.orientation=tf::createQuaternionMsgFromYaw(theta_center);
	map_pub.publish(map_m);
	check_cuda_error(cudaFreeHost(mapout_h));
	printf("freed outmap h\n");
	tf::Transform l_to_m=tf::Transform(tf::createQuaternionFromYaw(theta_best), tf::Vector3(x_best, y_best, 0.0)).inverse();
	//odom_broadcaster.sendTransform(tf::StampedTransform(l_to_o, current_time, "/base_link", "/slam_gpu/odom"));
	tf::Transform o_to_l=tf::Transform(tf::createQuaternionFromYaw(theta_old), tf::Vector3(x_old, y_old, 0.0));
	m_to_o=(o_to_l*l_to_m).inverse();
	/*        
	nav_msgs::Odometry odom;
        odom.header.stamp=current_time;
        odom.header.frame_id="/slam_gpu/odom";
        odom.pose.pose.position.x=x_best;
        odom.pose.pose.position.y=y_best;
        odom.pose.pose.orientation=tf::createQuaternionMsgFromYaw(theta_best);
        odom.child_frame_id="/base_link";
        odom.twist.twist.linear.x=vx;
        odom.twist.twist.linear.y=vy;
        odom.twist.twist.angular.z=vt;
        odom_pub.publish(odom);
	*/
	++index;
	ros::Time stop=ros::Time::now();
	printf("elapsed time for the laser cb:%f s\n", (stop-start).toSec());
    }

    void odoCB(const nav_msgs::Odometry::ConstPtr & o)
    {
	ros::Time start=ros::Time::now();
        float x_new=o->pose.pose.position.x;
        float y_new=o->pose.pose.position.y;
        float theta_new=tf::getYaw(o->pose.pose.orientation);
        vx=o->twist.twist.linear.x;
        vy=o->twist.twist.linear.y;
        vt=o->twist.twist.angular.z;
        if(first_o)
        {
            x_old=x_new;
            y_old=y_new;
            theta_old=theta_new;
            f->initParticles(x_new, y_new, theta_new);
	    //printf("init particles\n");
            first_o=false;
        }
        //then do draw from motion
        f->drawFromMotion(x_new, y_new, theta_new, x_old, y_old, theta_old, (int)((ros::Time::now()).toSec()));
        //printf("drawn new particles\n");
	x_old=x_new;
	y_old=y_new;
	theta_old=theta_new;
	ros::Time stop=ros::Time::now();
	printf("elapsed time for the odom cb:%f s\n", (stop-start).toSec());
    }
    void spin()
    {
        ros::Rate rate(30);
        while(ros::ok())
        {
            ros::spinOnce();
	    odom_broadcaster.sendTransform(tf::StampedTransform(m_to_o, ros::Time::now()+ros::Duration(0.03), "/slam_gpu/map", "/odom"));
            rate.sleep();
        }
    }


protected:
    ros::NodeHandle nh;
    ros::Subscriber laserS;
    ros::Subscriber odomS;
    tf::TransformBroadcaster odom_broadcaster;
    tf::Transform m_to_o;
    ros::Publisher odom_pub;
    ros::Publisher map_pub;
    FilterState * f;
    float x_old, y_old, theta_old, x_best, y_best, theta_best, vx, vy, vt, x_center, y_center, theta_center;
    float res;
    float rmax;
    int map_size;
    int width;
    int height;
    int index;
    size_t pitch;
    float * map;
    float * scanScores;
    bool first_o, first_s;
};


int main(int argc, char **argv)
{
    ros::init(argc, argv, "slam-gpu_node");
    ros::NodeHandle nh("~");
    RobotState rs(nh);
    rs.spin();
    return 0;
}

