/*! 
 @file rx_cuda_utils.h

 @brief CUDAópÇÃä÷êîÇ»Ç«
 
 @author Makoto Fujisawa
 @date 2012-12
*/


#ifndef _RX_CUDA_UTILS_H_
#define _RX_CUDA_UTILS_H_


#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_cuda.h>
#include <helper_cuda_gl.h>


#define RX_CUCHECK(val) RxCudaCheckFunc(val, #val, __FILE__, __LINE__)

template<typename T> 
inline bool RxCudaCheckFunc(T rtrn_val, const char* func, const char* file, const int line)
{
	if(rtrn_val){
		fprintf(stderr, "CUDA error at %s line %d : %s (error code = %d), function=%s\n",
				file, line, cudaGetErrorString(rtrn_val), (int)(rtrn_val), func);
		return true;
	}
	else
	{
		return false;
	}
}



#define RX_CUERROR(msg) RxCudaLastError(msg, __FILE__, __LINE__)

inline bool RxCudaLastError(const char *msg, const char *file, const int line)
{
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess){
		fprintf(stderr, "CUDA last error at %s line %d : %s (error code = %d)\n",
				file, line, cudaGetErrorString(err), (int)(err));
		return true;
	}
	else
	{
		return false;
	}
}


#endif // #ifndef _RX_CUDA_UTILS_H_