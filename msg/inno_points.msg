# Innovusion publish pointcloud msg format.

Header header             # ROS standard message header
string frame_id
bool is_dense
inno_point[] point
float64 measurement_time
uint64 measurement_time_ms
uint64 send_time
uint32 width
uint32 height
string model
uint32 source_id
uint64 idx   # index of frame
uint64 frame_ns_start
uint64 frame_ns_end