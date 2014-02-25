gpu-og
======

occupancy grid and slam on gpu. based on several papers:
- Rodriguez-Losada, Diego, et al. "GPU-Mapping: Robotic Map Building with Graphical Multiprocessors." Robotics & Automation Magazine, IEEE 20.2 (2013): 40-51.
- Thrun, Sebastian, and John J. Leonard. "Simultaneous localization and mapping." Springer handbook of robotics (2008): 871-889.
- Yguel, Manuel, Olivier Aycard, and Christian Laugier. "Efficient GPU-based construction of occupancy grids using several laser range-finders." International Journal of Vehicle Autonomous Systems 6.1 (2008): 48-83.
- Zhang, Haiyang, and Fred Martin. "CUDA accelerated robot localization and mapping." Technologies for Practical Robot Applications (TePRA), 2013 IEEE International Conference on. IEEE, 2013.



Important branches:
- slam: SLAM implementation, based on the CARMEN logfile format
- ros-slam: SLAM implementation, integrated in a ROS node
- master: Occupancy grid implementation, based on the CARMEN logfile format
