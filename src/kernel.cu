/*
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                             *
 * MIT License                                                                 *
 *                                                                             *
 * Copyright (c) 2023, Davide Stocco                                           *
 *                                                                             *
 * Permission is hereby granted, free of charge, to any person obtaining a     *
 * copy of this software and associated documentation files (the "Software"),  *
 * to deal in the Software without restriction, including without limitation   *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,    *
 * and/or sell copies of the Software, and to permit persons to whom the       *
 * Software is furnished to do so, subject to the following conditions:        *
 *                                                                             *
 * The above copyright notice and this permission notice shall be included in  *
 * all copies or substantial portions of the Software.                         *
 *                                                                             *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,    *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE *
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER      *
 * FROM, LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,       *
 * ARISING OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER       *
 * DEALINGS IN THE SOFTWARE.                                                   *
 *                                                                             *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
*/

// CUDA libraries
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// C/C++ libraries
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <iostream>

// OpenCV library
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// Define CUDA error check
#ifndef GPU_ERROR
#define GPU_ERROR(MSG)                   \
{                                        \
  GPU_Assert((MSG), __FILE__, __LINE__); \
}
#endif

// CUDA error check
inline
void
GPU_Assert(
  cudaError_t code,         // Error code
  const char *file,         // File name
  int         line,         // Line number
  bool        abort = false // Abort program flag
)
{
  if (code != cudaSuccess)
  {
    fprintf(
      stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line
    );
    if (abort == true)
    {
      exit(code);
    }
  }
}

// Define namespaces
using namespace cv;
using namespace std;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

// CUDA kernel for undistortion
__global__
void
undistort(
  uchar *image_in,     // Input image
  uchar *image_out_CUDA, // Output image
  double k1,        // Radial distortion coefficient k1
  double k2,        // Radial distortion coefficient k2
  double k3,        // Radial distortion coefficient k3
  double k4,        // Radial distortion coefficient k4
  int    width,     // Image width
  int    height     // Image height
)
{
  // Setting texture memory
  uint i = blockIdx.x * blockDim.x + threadIdx.x ;
  uint j = blockIdx.y * blockDim.y + threadIdx.y;

  // Define distortion center and polar coordinates
  double xc = width / 2.0;
  double yc = height / 2.0;
  double r  = std::sqrt((i-xc)*(i-xc) + (j-yc)*(j-yc));

  double k1_2 = k1*k1;
  double r_2  = r*r;
  double r_4  = r_2*r_2;

  // Calculate bn coefficents and Inverse transformation Q
  // Notice that b0 = 1
  double b1 = -k1;
  double b2 = 3.0*k1_2 - k2;
  double b3 = 8.0*k1*k2 - 12.0*k1_2*k1 - k3;
  double b4 = 55.0*k1_2*k1_2 + 10.0*k1*k3 - 55.0*k1_2*k2 + 5*k2*k2 - k4;
  double Q  = 1.0 + b1*r_2 + b2*r_4 + b3*r_2*r_4 + b4*r_4*r_4;

  // Final (x,y) coordinates
  double x = (i-xc) * Q;
  double y = (j-yc) * Q;
  int    o = xc + yc * width;

  // Bilinear Interpolation Algorithm
  // Ensure that the undistorted point sits in the image
  if (i <= width && j <= height)
  {
    // Define intermediate points
    int x1 = std::floor(x);
    int y1 = std::floor(y);
    int x2 = x1 + 1;
    int y2 = y1 + 1;

    // Ensure that the mean color value is computable
    if(x1 >= -xc && y1 >= -yc && x2 < (width/2.0) && y2 < (height/2.0))
    {
      double val1 = image_in[x1 + y1 * width + o];
      double val2 = image_in[x2 + y1 * width + o];
      double val3 = image_in[x1 + y2 * width + o];
      double val4 = image_in[x2 + y2 * width + o];

      double color1 = (x2 - x) * val1 + (x - x1) * val2;
      double color2 = (x2 - x) * val3 + (x - x1) * val4;

      image_out_CUDA[i + j * width] = (y2 - y) * color1 + (y - y1) * color2;
    }
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

// Main Function
int
main(void)
{
  // Print instructions
  std::cout
    << "CUDA KERNEL FOR RADIAL UNDISTORTION" << std::endl
    << "Press key:" << std::endl
    << "'a'=+k1 'z'=-k1 's'=+k2 'x'=+k2" << std::endl
    << "'d'=+k3 'c'=-k3 'f'=+k4 'v'=+k4" << std::endl
    << "'w' = See current distortion values" << std::endl
    << "'r' = Reset distortion values" << std::endl
    << "'q' = Quit program" << std::endl
    << std::endl;

  // Allocate distortion parameters
  double k1  = 0.0;
  double k2  = 0.0;
  double k3  = 0.0;
  double k4  = 0.0;
  double inc = 0.0000001;

  // Read input key
  char key;
  while (1) {
    std::cout << "Key:\t";
    std::cin >> key;

    // Switch case for key
    switch (key) {
      // Coefficient k1
      case 'a': {
        // increment k1
        k1 += inc;
        break;
      }
      case 'z': {
        // Decrement k1
        k1 -= inc;
        break;
      }
      // Coefficient k2
      case 's': {
      // Increment k2
        k2 += inc*inc;;
        break;
      }
      case 'x': {
      // Decrement k2
        k2 -= inc*inc;;
        break;
      }
      // Coefficient p1
      case 'd': {
        // increment k3
        k3 += inc*inc*inc;
        break;
      }
      case 'c': {
        // Decrement k3
        k3 -= inc*inc*inc;
        break;
      }
      // Coefficient p2
      case 'f': {
      // increment k4
        k4 += inc*inc*inc*inc;
        break;
      }
      case 'v': {
      // Decrement k4
        k4 -= inc*inc*inc*inc;
        break;
      }
      case 'q': {
        // Terminate program
        return 0;
      }
      case 'w': {
        // Print current distortion values
        std::cout
          << std::endl
          << "Current distortion values are:" << std::endl
          << "k1 = " << k1 << std::endl
          << "k2 = " << k2 << std::endl
          << "k3 = " << k3 << std::endl
          << "k4 = " << k4 << std::endl
          << std::endl;
        continue;
      }
      case 'r': {
        // Reset distortion values
        std::cout
          << std::endl
          << "Reset distortion values!" << std::endl
          << std::endl;
        k1 = 0.0;
        k2 = 0.0;
        k3 = 0.0;
        k4 = 0.0;
        continue;
      }
      default: {
        // If input is wrong, ask again
        std::cout
          << "Wrong key. Try again!" << std::endl;
        continue;
      }
    }

    // Read input image
    Mat image_in = imread(".lena.png", 0);
    uchar *img_in;

    // Allocate output image
    Mat image_out_CUDA = Mat(image_in.rows, image_in.cols, CV_8UC1, double(0.0));
    uchar *img_out;

    // Get image size
    int N = image_in.rows * image_in.cols;

    // Initialize CUDA stream and error
    cudaStream_t CUDA_stream;
    cudaError_t  CUDA_error;
    CUDA_error = cudaStreamCreate(&CUDA_stream);

    // To GPU memory
    GPU_ERROR(cudaFree(0));
    GPU_ERROR(cudaMalloc(&img_in, N*sizeof(uchar)));
    GPU_ERROR(cudaMemcpyAsync(img_in, image_in.data, N*sizeof(uchar), cudaMemcpyHostToDevice, CUDA_stream));
    GPU_ERROR(cudaMalloc(&img_out, N*sizeof(uchar)));
    GPU_ERROR(cudaMemcpyAsync(img_out, image_out_CUDA.data, N*sizeof(uchar), cudaMemcpyHostToDevice, CUDA_stream));

    // Kernel settings
    // NOTE1: Change this to your GPU's max block size, or the kernel will fail.
    // NOTE2: This part shold be optimized for a generic GPU architecture.
    dim3 dimBlock(256, 540);
    dim3 numBlocks(image_in.cols/dimBlock.x, image_in.rows/dimBlock.y);

    // Apply kernel on input image
    clock_t CUDA_start_clock = clock();
    undistort<<<dimBlock, numBlocks, 0, CUDA_stream>>>(
      img_in, img_out, k1, k2, k3, k4, image_in.cols, image_in.rows
      );
    cudaDeviceSynchronize();
    GPU_ERROR(cudaPeekAtLastError());
    clock_t CUDA_stop_clock = clock();
    double CUDA_elapsed_time =
      (CUDA_stop_clock - CUDA_start_clock) * 1000.0 / (double)CLOCKS_PER_SEC;

    std::cout
      << "Custom CUDA kernel - Elapsed time:"
      << CUDA_elapsed_time << " (ms)" << std::endl;
    GPU_ERROR(cudaPeekAtLastError());

    // Check for CUDA errors
    cudaError_t CUDA_past_error = cudaGetLastError();
    if (CUDA_past_error != cudaSuccess)
    {
      fprintf(stderr, "CUDA ERROR: %s \n", cudaGetErrorString(CUDA_past_error));
    }

    // To CPU
    uchar * out = (uchar*)malloc(N * sizeof(uchar));
    GPU_ERROR(cudaMemcpyAsync(out, img_out, N * sizeof(uchar), cudaMemcpyDeviceToHost, CUDA_stream));
    Mat out_Mat = Mat(image_in.rows, image_in.cols, CV_8UC1, out);

    // Display undistorted image
    if (!out_Mat.data)
    {
      std::cout
        << "Could not open or find the output image." << std::endl;
        return -1;
    }
    else
    {
      namedWindow("Undistorted image - Custom CUDA kernel", WINDOW_NORMAL);
      resizeWindow("Undistorted image - Custom CUDA kernel", 1600, 900);
      imshow("Undistorted image - Custom CUDA kernel", out_Mat);
      waitKey(0);
      imwrite("./lena.jpg" , out_Mat);
    }

    // Undistortion with OpenCV
    Mat image_out_CV  = Mat(image_in.rows, image_in.cols, CV_8UC1, double(0.0));
    Mat intrinsic    = Mat(3, 3, CV_8UC1);
    Mat distCoeffs   = (Mat1d(1, 4) << -k1, -k2, 0, 0, -k3, -k4, 0, 0);
    Mat cameraMatrix = (Mat1d(3, 3) << 1, 0, image_in.rows/2, 0, 1, image_in.cols/2, 0, 0, 1);

    clock_t CV_start_clock = clock();
    undistort(image_in, image_out_CV, cameraMatrix, distCoeffs);
    clock_t CV_stop_clock = clock();
    double CV_elapsed_time =
      (CV_stop_clock - CV_start_clock) * 1000.0 / (double)CLOCKS_PER_SEC;

    std::cout
      << "OpenCV function - Elapsed time:"
      << CV_elapsed_time << " (ms)" << std::endl;

    // Display undistorted image
    imshow("Undistorted image - OpenCV", image_out_CV);
    waitKey(0);

    // close the windows
    cvDestroyWindow("Undistorted image - Custom CUDA kernel");
    cvDestroyWindow("Undistorted image - OpenCV");
    out_Mat.release();
    image_out_CUDA.release();
    image_out_CV.release();
    image.release();
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

// That's all folks!
