#include <iostream>
#include <fstream>
#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/cudacodec.hpp"
#include <getopt.h>
#include <stdio.h>
#include "opencv2/video/tracking.hpp"
#include <dirent.h>

using namespace cv;
using namespace cv::cuda;
using namespace std;


//Mostly from https://github.com/opencv/opencv/blob/master/samples/gpu/optical_flow.cpp


//http://stackoverflow.com/questions/24221605/find-all-files-in-a-directory-and-its-subdirectory
int getdir(string dir, vector<string> &files)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL)
    {
        cout << "Error opening " << dir << endl;
        return -1;
    }

    while ((dirp = readdir(dp)) != NULL)
    {
        string s = string(dirp->d_name);
        if(strcmp(s.c_str(), ".") && strcmp(s.c_str(), ".."))
            files.push_back(s);
    }
    closedir(dp);
    return 1;
}

inline bool isFlowCorrect(Point2f u)
{
    return !cvIsNaN(u.x) && !cvIsNaN(u.y) && fabs(u.x) < 1e9 && fabs(u.y) < 1e9;
}

static Vec3b computeColor(float fx, float fy)
{
    static bool first = true;

    // relative lengths of color transitions:
    // these are chosen based on perceptual similarity
    // (e.g. one can distinguish more shades between red and yellow
    //  than between yellow and green)
    const int RY = 15;
    const int YG = 6;
    const int GC = 4;
    const int CB = 11;
    const int BM = 13;
    const int MR = 6;
    const int NCOLS = RY + YG + GC + CB + BM + MR;
    static Vec3i colorWheel[NCOLS];

    if (first)
    {
        int k = 0;

        for (int i = 0; i < RY; ++i, ++k)
            colorWheel[k] = Vec3i(255, 255 * i / RY, 0);

        for (int i = 0; i < YG; ++i, ++k)
            colorWheel[k] = Vec3i(255 - 255 * i / YG, 255, 0);

        for (int i = 0; i < GC; ++i, ++k)
            colorWheel[k] = Vec3i(0, 255, 255 * i / GC);

        for (int i = 0; i < CB; ++i, ++k)
            colorWheel[k] = Vec3i(0, 255 - 255 * i / CB, 255);

        for (int i = 0; i < BM; ++i, ++k)
            colorWheel[k] = Vec3i(255 * i / BM, 0, 255);

        for (int i = 0; i < MR; ++i, ++k)
            colorWheel[k] = Vec3i(255, 0, 255 - 255 * i / MR);

        first = false;
    }

    const float rad = sqrt(fx * fx + fy * fy);
    const float a = atan2(-fy, -fx) / (float) CV_PI;

    const float fk = (a + 1.0f) / 2.0f * (NCOLS - 1);
    const int k0 = static_cast<int>(fk);
    const int k1 = (k0 + 1) % NCOLS;
    const float f = fk - k0;

    Vec3b pix;

    for (int b = 0; b < 3; b++)
    {
        const float col0 = colorWheel[k0][b] / 255.0f;
        const float col1 = colorWheel[k1][b] / 255.0f;

        float col = (1 - f) * col0 + f * col1;

        if (rad <= 1)
            col = 1 - rad * (1 - col); // increase saturation with radius
        else
            col *= .75; // out of range

        pix[2 - b] = static_cast<uchar>(255.0 * col);
    }

    return pix;
}

static void drawOpticalFlow(const Mat_<float>& flowx, const Mat_<float>& flowy, Mat& dst, float maxmotion = -1)
{
    dst.create(flowx.size(), CV_8UC3);
    dst.setTo(Scalar::all(0));

    // determine motion range:
    float maxrad = maxmotion;

    if (maxmotion <= 0)
    {
        maxrad = 1;
        for (int y = 0; y < flowx.rows; ++y)
        {
            for (int x = 0; x < flowx.cols; ++x)
            {
                Point2f u(flowx(y, x), flowy(y, x));

                if (!isFlowCorrect(u))
                    continue;

                maxrad = max(maxrad, sqrt(u.x * u.x + u.y * u.y));
            }
        }
    }

    for (int y = 0; y < flowx.rows; ++y)
    {
        for (int x = 0; x < flowx.cols; ++x)
        {
            Point2f u(flowx(y, x), flowy(y, x));

            if (isFlowCorrect(u))
                dst.at<Vec3b>(y, x) = computeColor(u.x / maxrad, u.y / maxrad);
        }
    }
}

static void showFlow(const char* name, const GpuMat& d_flow)
{
    GpuMat planes[2];
    cuda::split(d_flow, planes);

    Mat flowx(planes[0]);
    Mat flowy(planes[1]);

    Mat out;
    drawOpticalFlow(flowx, flowy, out, 10);

    //imwrite("deneme.jpg", out);
    imshow(name, out);
}

static void showFlow(const char* name, const Mat& d_flow)
{
	vector<Mat> planes;
    split(d_flow, planes);
    Mat flowx(planes[0]);
    Mat flowy(planes[1]);

	Mat out;
    drawOpticalFlow(flowx, flowy, out, 10);

    //imwrite("deneme.jpg", out);
    imshow(name, out);
}

/* Compute the magnitude of flow given x and y components */
static void computeFlowMagnitude(const Mat_<float>& flowx, const Mat_<float>& flowy, Mat& dst)
{
    dst.create(flowx.size(), CV_32FC1);
    for (int y = 0; y < flowx.rows; ++y)
    {
        for (int x = 0; x < flowx.cols; ++x)
        {
            Point2f u(flowx(y, x), flowy(y, x));

            if (!isFlowCorrect(u))
                continue;

            dst.at<float>(y, x) = sqrt(u.x * u.x + u.y * u.y);
        }
    }
}

/* Write raw optical flow values into txt file
Example usage:
    writeFlowRaw<float>(name+"_x_raw.txt", flowx);
    writeFlowRaw<int>(name+"_x_raw_n.txt", flowx_n);
*/
template <typename T>
static void writeFlowRaw(string name, const Mat& flow)
{
    ofstream file;
    file.open(name.c_str());
    for(int y=0; y<flow.rows; ++y)
    {
        for(int x=0; x<flow.cols; ++x)
        {
            file << flow.at<T>(y, x) << " ";
        }
        file << endl;
    }
    file.close();
}

//min_x max_x min_y max_y
static void writeMM(string name, vector<double> mm)
{
    ofstream file;
    file.open(name.c_str());
    for(int i=0; i<mm.size(); i++)
    {
        file << mm[i] << " ";
    }
    file.close();
}

//min_x max_x min_y max_y (one line per frame)
static void writeMM(string name, vector<vector<double> > mm)
{
    ofstream file;
    file.open(name.c_str());
    for(int i=0; i<mm.size(); i++)
    {
        for(int j=0; j<mm[i].size(); j++)
        {
            file << mm[i][j] << " ";
        }
        file << endl;
    }
    file.close();
}

static vector<double> getMM(const Mat& flow)
{
    double min, max;
    cv::minMaxLoc(flow, &min, &max);
    vector<double> mm;
    mm.push_back(min);
    mm.push_back(max);
    return mm;
}

/* Write a 3-channel jpg image (flow_x, flow_y, flow_magnitude) in 0-255 range */
static void writeFlowMergedJpg(string name, const GpuMat& d_flow)
{
    GpuMat planes[2];
    cuda::split(d_flow, planes);

    Mat flowx(planes[0]);
    Mat flowy(planes[1]);

    Mat flowmag;
    computeFlowMagnitude(flowx, flowy, flowmag);

    Mat flowx_n, flowy_n, flowmag_n;
    cv::normalize(flowx, flowx_n, 0, 255, NORM_MINMAX, CV_8UC1);
    cv::normalize(flowy, flowy_n, 0, 255, NORM_MINMAX, CV_8UC1);
    cv::normalize(flowmag, flowmag_n, 0, 255, NORM_MINMAX, CV_8UC1);

    vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
    compression_params.push_back(95);

    Mat flow;
    vector<Mat> array_to_merge;
    array_to_merge.push_back(flowx_n);
    array_to_merge.push_back(flowy_n);
    array_to_merge.push_back(flowmag_n);
    cv::merge(array_to_merge, flow);

    imwrite(name+".jpg", flow, compression_params);
}

/* Write two 1-channel jpg images (flow_x and flow_y) in 0-255 range (input flow is gpumat)*/
static vector<double> writeFlowJpg(string name, const GpuMat& d_flow)
{
    // Split flow into x and y components in CPU
    GpuMat planes[2];
    cuda::split(d_flow, planes);
    Mat flowx(planes[0]);
    Mat flowy(planes[1]);

    // Normalize optical flows in range [0, 255]
    Mat flowx_n, flowy_n;
    cv::normalize(flowx, flowx_n, 0, 255, NORM_MINMAX, CV_8UC1);
    cv::normalize(flowy, flowy_n, 0, 255, NORM_MINMAX, CV_8UC1);

    // Save optical flows (x, y) as jpg images
    vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
    compression_params.push_back(95);

    imwrite(name+"_x.jpg", flowx_n, compression_params);
    imwrite(name+"_y.jpg", flowy_n, compression_params);

    // Return normalization elements
    vector<double> mm_frame;
    vector<double> temp = getMM(flowx);
    mm_frame.insert(mm_frame.end(), temp.begin(), temp.end());
    temp = getMM(flowy);
    mm_frame.insert(mm_frame.end(), temp.begin(), temp.end());

    return mm_frame;
}

/* Write two 1-channel jpg images (flow_x and flow_y) in 0-255 range (input flow is cpu mat)*/
static vector<double> writeFlowJpg(string name, const Mat& d_flow)
{
    vector<Mat> planes;
    split(d_flow, planes);
    Mat flowx(planes[0]);
    Mat flowy(planes[1]);
    // Normalize optical flows in range [0, 255]
    Mat flowx_n, flowy_n;
    cv::normalize(flowx, flowx_n, 0, 255, NORM_MINMAX, CV_8UC1); //TO-DO
    cv::normalize(flowy, flowy_n, 0, 255, NORM_MINMAX, CV_8UC1); //TO-DO

    // Save optical flows (x, y) as jpg images
    vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
    compression_params.push_back(95);

    imwrite(name+"_x.jpg", flowx_n, compression_params);
    imwrite(name+"_y.jpg", flowy_n, compression_params);

    // Return normalization elements
    vector<double> mm_frame;
    vector<double> temp = getMM(flowx); //TO-DO
    mm_frame.insert(mm_frame.end(), temp.begin(), temp.end());
    temp = getMM(flowy); //TO-DO
    mm_frame.insert(mm_frame.end(), temp.begin(), temp.end());

    return mm_frame;
}

int main(int argc, const char* argv[])
{
    // Parse parameters and options
    string input_name;              // name of the video file or directory of the image sequence
    string proc_type    = "gpu";    // "gpu" or "cpu"
    string out_dir   	= "./";     // directory for the output files
    int interval_beg  	= 1;       	// 1 for the beginning
    int interval_end    = -1;       // End (-1 for uninitialized, default set below)
    bool visualize      = 0;    	// boolean for flow visualization
    string output_mm    = "";       // name of the minmax.txt file (default set below)

    const char* usage = "[-h] [-p <proc_type>] [-o <out_dir>] [-b <interval_beg>] [-e <interval_end>] [-v <visualize>] [-m <output_mm>] <input_name>";
    string help  = "\n\n\nUSAGE:\n\t"+ string(usage) +"\n\n"
        "INPUT:\n"
        "\t<input_name>\t \t: Path to video file or image directory (e.g. img_%04d.jpg)\n"
        "OPTIONS:\n"
        "-h \t \t \t \t: Display this help message\n"
        "-p \t<proc_type>  \t[gpu] \t: Processor type (gpu or cpu)\n"
        "-o \t<out_dir> \t[./] \t: Output folder containing flow images and minmax.txt\n"
        "-b \t<interval_beg> \t[1] \t: Frame index to start (one-based indexing)\n"
        "-e \t<interval_end> \t[last] \t: Frame index to stop\n"
        "-v \t<visualize> \t[0] \t: Boolean for visualization of the optical flow\n"
        "-m \t<output_mm> \t[<out_dir>/<basename(input_name)>_minmax.txt] \t: Name of the minmax file.\n"
        "\n"
        "Notes:\n*GPU method: Brox, CPU method: Farneback.\n"
        "*Only <imagename>_%0xd.jpg (x any digit) is supported for image sequence input.\n\n\n";
// brox cpu
//fourcc check for image sequence detection, but not sure
    int option_char;
    while ((option_char = getopt(argc, (char **)argv, "hp:o:b:e:m:v:?")) != EOF)
    {
        switch (option_char)
        {  
            case 'p': proc_type      = optarg;       break;
            case 'o': out_dir        = optarg;       break;
            case 'b': interval_beg   = atoi(optarg); break;
            case 'e': interval_end   = atoi(optarg); break;
            case 'v': visualize      = atoi(optarg); break;
            case 'm': output_mm      = optarg;       break;
            case 'h': cout << help; return 0;        break;
            case '?': fprintf(stderr, "Unknown option.\nUSAGE: %s %s\n", argv[0], usage); return -1; break;
        }
    }

    // Retrieve the (non-option) argument
    if ( (argc <= 1) || (argv[argc-1] == NULL) || (argv[argc-1][0] == '-') )
    {
        fprintf(stderr, "No input name provided.\nUSAGE: %s %s\n", argv[0], usage);
        return -1;
    }
    else
    {
        input_name = argv[argc-1];
    }

    if(out_dir.compare("") != 0) 
    {
    	if(out_dir[out_dir.length()-1]!= '/') { out_dir = out_dir + "/"; } //and if last char not /
        char cmd[200];
        sprintf(cmd, "mkdir -p %s", out_dir.c_str());
        system(cmd);
    }
    if(output_mm.compare("") == 0)
    {
        output_mm = out_dir+basename(input_name.c_str())+"_minmax.txt";
    }

    // Declare useful variables
    Mat frame0, frame1;
    char name[200];
    vector<vector<double> > mm;
    
    // VIDEO INPUT
    if(proc_type.compare("gpu") == 0)
    {
    	cout << "Extracting flow from [" << input_name << "] using GPU." << endl;
    	// Solve the bug of allocating gpu memory after cap.read
    	cout << "Initialization (this may take awhile)..." << endl;
    	GpuMat temp = GpuMat(3, 3, CV_32FC1);
    	// Declare gpu mats
    	GpuMat g_frame0, g_frame1;
    	GpuMat gf_frame0, gf_frame1;
    	GpuMat g_flow;
   		// Create optical flow object
    	Ptr<cuda::BroxOpticalFlow> brox = cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 10, 77, 10);
                 
    	// Open video file
    	VideoCapture cap(input_name);
    	double cap_fourcc = cap.get(CV_CAP_PROP_FOURCC);
    	if(cap.isOpened())
    	{
        	int noFrames = cap.get(CV_CAP_PROP_FRAME_COUNT); //get the frame count
        	if(interval_end == -1)
        	{
            	interval_end = noFrames;
        	}
        	string outname = basename(input_name.c_str());
        	if(cap_fourcc == 0) // image sequence
        	{
        		outname = string(outname).substr(0, outname.length()-9);
        		output_mm = out_dir+outname+"_minmax.txt";
        	}
        	cout << "Total number of frames: " << noFrames << endl;
        	cout << "Extracting interval [" << interval_beg  << "-" << interval_end << "]" << endl;
        	cap.set(CV_CAP_PROP_POS_FRAMES, interval_beg-1); // causes problem for image sequence!
 
        	// Read first frame
        	if(noFrames>0)
        	{
            	bool bSuccess = cap.read(frame0);
            	if(!bSuccess)   { cout << "Cannot read frame!" << endl;   }
            	else            { cvtColor(frame0, frame0, CV_BGR2GRAY);  }
        	}
        	// For each frame in video (starting from the 2nd)
        	for(int k=1; k<interval_end-interval_beg+1; k++)
        	{
				sprintf(name, "%s%s_%05d", out_dir.c_str(), outname.c_str(), k+interval_beg-1);
            	
            	bool bSuccess = cap.read(frame1);
            	//imshow("Frame", frame1);
            	//waitKey();
            	if(!bSuccess)   { cout << "Cannot read frame " << name << "!" << endl; }
            	else
            	{
            		cout << "Outputting " << name << endl;
                	cvtColor(frame1, frame1, CV_BGR2GRAY);

                	// Upload images to GPU
                	g_frame0 = GpuMat(frame0); // Has an image in format CV_32FC1
                	g_frame1 = GpuMat(frame1); // Has an image in format CV_32FC1

                	// Convert to float
                	g_frame0.convertTo(gf_frame0, CV_32F, 1.0/255.0);
                	g_frame1.convertTo(gf_frame1, CV_32F, 1.0/255.0);

                	// Prepare receiving variable
                	g_flow = GpuMat(frame0.size(), CV_32FC2);
 
                	// Perform Brox optical flow
                	brox->calc(gf_frame0, gf_frame1, g_flow);
                	vector<double> mm_frame = writeFlowJpg(name, g_flow);
                	if(visualize)
                	{
                		showFlow("Flow", g_flow);
                		waitKey(30);
                	}
                
                	mm.push_back(mm_frame);
                	frame1.copyTo(frame0);
            	}
        	}
        	cout << "Outputting " << output_mm << endl;
        	writeMM(output_mm, mm);
        	cap.release();
    	}
    	else
    	{
        	cout << "Video " << input_name << " cannot be opened." << endl;
    	}
    }
    else if(proc_type.compare("cpu") == 0)
    {
    	cout << "Extracting flow from [" << input_name << "] using CPU." << endl;
    	
	Ptr<DualTVL1OpticalFlow> tvl1 = createOptFlow_DualTVL1();
    	VideoCapture cap(input_name);
    	if(cap.isOpened())
    	{
        	int noFrames = cap.get(CV_CAP_PROP_FRAME_COUNT);
        	if(interval_end == -1)
	        {
	            interval_end = noFrames;
	        }
	        cout << "Total number of frames: " << noFrames << endl;
	        cout << "Extracting interval [" << interval_beg  << "-" << interval_end << "]" << endl;
	        cap.set(CV_CAP_PROP_POS_FRAMES, interval_beg-1);
	        // Read first frame
	        if(noFrames>0)
	        {
	            bool bSuccess = cap.read(frame0);
	            if(!bSuccess)   { cout << "Cannot read frame!" << endl;   }
	            else            { cvtColor(frame0, frame0, CV_BGR2GRAY);        }
	        }
	        // For each frame in video (starting from the 2nd)
	        for(int k=1; k<interval_end-interval_beg+1; k++)
	        {
	            sprintf(name, "%s%s_%05d", out_dir.c_str(), basename(input_name.c_str()), k+interval_beg-1);
	            bool bSuccess = cap.read(frame1);
	            if(!bSuccess)   { cout << "Cannot read frame " << name << "!" << endl; }
	            else
	            {
	            	cout << "Outputting " << name << endl;
	                cvtColor(frame1, frame1, CV_BGR2GRAY);

	                // Convert to float
	                frame0.convertTo(frame0, CV_32F, 1.0/255.0);
	                frame1.convertTo(frame1, CV_32F, 1.0/255.0);

	                // Prepare receiving variable
	                Mat flow = Mat(frame0.size(), CV_32FC2);

	                // Perform optical flow
	//                tvl1->calc(frame0, frame1, flow);

	                calcOpticalFlowFarneback(frame0, frame1, flow, 0.5, 3, 3, 3, 5, 1.1, 0);  
	                vector<double> mm_frame = writeFlowJpg(name, flow);
	                
	                if(visualize)
	                {
	                	showFlow("Flow", flow);
	                	waitKey(30);
	                }

	                mm.push_back(mm_frame);
	                frame1.copyTo(frame0);

	            }
	        }
	        cout << "Outputting " << output_mm << endl;
	        writeMM(output_mm, mm);
	        cap.release();
	    }
	    else
	    {
	        cout << "Video " << input_name << " cannot be opened." << endl;
	    }
    }
    return 0;
}
