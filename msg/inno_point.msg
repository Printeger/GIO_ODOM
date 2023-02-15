# Innovusion costom pointcloud format.

uint32 offset_time      # offset time relative to the base time
float32 x               # X axis, unit:m
float32 y               # Y axis, unit:m
float32 z               # Z axis, unit:m
float32 intensity # intensity, 0~255
float32 elongation  # [default = 0];
uint64 timestamp
uint32 label
uint32 sub_id  # default = -1  belong to box
uint32 box_lut_id   # [default = -1]; // start from 0
uint32 flags
uint32 scan_id
uint32 scan_idx