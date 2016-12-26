# Flow toolbox
Optical flow extraction tool using OpenCV. The code is simple and mostly from documentation of OpenCV, but it makes it easy to use for pre-processing of videos with several options. It is easy to customize and change the algorithms used. Currently Brox algorithm is used for GPU and Farneback is used for CPU implementations.

## Compilation
TO-DO

## Help
Type `./flow_video -h` to see the help message below.  


```shell
USAGE:
	[-h] [-p <proc_type>] [-o <out_dir>] [-b <interval_beg>] [-e <interval_end>] [-v <visualize>] [-m <output_mm>] <input_name>

INPUT:
	<input_name>	 	    : Path to video file or image directory (e.g. img_%04d.jpg)
OPTIONS:
-h 	 	 	 	            : Display this help message
-p 	<proc_type>  	[gpu] 	: Processor type (gpu or cpu)
-o 	<out_dir> 	    [./] 	: Output folder containing flow images and minmax.txt
-b 	<interval_beg> 	[1] 	: Frame index to start (one-based indexing)
-e 	<interval_end> 	[last] 	: Frame index to stop
-v 	<visualize> 	[0] 	: Boolean for visualization of the optical flow
-m 	<output_mm> 	[<out_dir>/<basename(input_name)>_minmax.txt] 	: Name of the minmax file.

Notes:
*GPU method: Brox, CPU method: Farneback.
*Only <imagename>_%0xd.jpg (x any digit) is supported for image sequence input.
```

## Example usage
Extract flow from the frame interval 5-10.

```shell
./flow_video -b 5 -e 10 -o samples/out samples/video/video.avi
./flow_video -b 5 -e 10 -o samples/out samples/images/img1/img1_%05d.jpg
```
