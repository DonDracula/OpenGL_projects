/*! 
  @file rx_cu_funcs.cu
	
  @brief CUDA関数 - メモリ関係など

  @author Makoto Fujisawa
  @date 2009-08, 2011-06
*/
// FILE --rx_cu_funcs.cu--


//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
#include <cstdio>
#include <GL/glew.h>
#include <GL/glut.h>

#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include <GL/freeglut.h>

#include "rx_cu_common.cu"



//-----------------------------------------------------------------------------
// CUDA関数
//-----------------------------------------------------------------------------
extern "C"
{
/*!
 * CUDAデバイスの設定
 *  - コマンドライン引数に基づきCUDAデバイスを設定((例)-device 0)
 * @param[in] argc コマンドライン引数の数
 * @param[in] argv コマンドライン引数リスト(argv[0]は実行ファイル名)
 */
void CuInit(int argc, char **argv)
{   
	if(checkCmdLineFlag(argc, (const char**)argv, "device")){
		int id = getCmdLineArgumentInt(argc, (const char**)argv, "device=");
		if(id < 0){
			id = gpuGetMaxGflopsDeviceId();
			cudaSetDevice(id);
		}
		else{
			cudaSetDevice(id);
		}
	}
	else{
		cudaSetDevice( gpuGetMaxGflopsDeviceId() );
	}
}

/*!
 * CUDAデバイスの設定 - idを直接指定
 * @param[in] id デバイスID
 */
void CuSetDevice(int id)
{ 
	int device_count = 0;
	cudaGetDeviceCount(&device_count);
	if(id < 0 || id >= device_count){
		id = gpuGetMaxGflopsDeviceId();
		cudaSetDevice(0);
	}
	else{
		cudaSetDevice(id);
	}
}

/*!
 * デバイスメモリの確保
 * @param[out] dPtr デバイスメモリへのポインタ
 * @param[in] size 確保サイズ(メモリ上のサイズ)
 */
void CuAllocateArray(void **dPtr, size_t size)
{
	RX_CUCHECK(cudaMalloc(dPtr, size));
}

/*!
 * デバイスメモリの解放
 * @param[in] devPtr デバイスメモリへのポインタ
 */
void CuFreeArray(void *dPtr)
{
	RX_CUCHECK(cudaFree(dPtr));
}

/*!
 * デバイスメモリ領域の初期化
 * @param[in] dPtr デバイスメモリへのポインタ
 * @param[in] val 初期値
 * @param[in] size 初期化する領域のサイズ(メモリ上のサイズ)
 */
void CuSetArrayValue(void *dPtr, int val, size_t size)
{
	RX_CUCHECK(cudaMemset(dPtr, val, size));
}

/*!
 * デバイスメモリ間コピー
 * @param[in] dDst コピー先
 * @param[in] dSrc コピー元
 * @param[in] size コピーサイズ(メモリ上のサイズ)
 */
void CuCopyArrayD2D(void *dDst, void *dSrc, int size)
{
	RX_CUCHECK(cudaMemcpy(dDst, dSrc, size, cudaMemcpyDeviceToDevice));
}


/*!
 * VBOをマッピング
 * @param[in] vbo VBO,PBO名
 */
void *CuMapGLBufferObject(cudaGraphicsResource **resource)
{
	void *ptr;
	RX_CUCHECK(cudaGraphicsMapResources(1, resource, 0));
	size_t num_bytes;
	RX_CUCHECK(cudaGraphicsResourceGetMappedPointer((void**)&ptr, &num_bytes, *resource));
	return ptr;
}

/*!
 * VBOをアンマップ
 * @param[in] vbo VBO,PBO名
 */
void CuUnmapGLBufferObject(cudaGraphicsResource *resource)
{
	RX_CUCHECK(cudaGraphicsUnmapResources(1, &resource, 0));
}

/*!
 * PBO,VBOバッファをCUDAに登録
 * @param[in] vbo VBO,PBO名
 */
void CuRegisterGLBufferObject(uint vbo, cudaGraphicsResource **resource)
{
	RX_CUCHECK(cudaGraphicsGLRegisterBuffer(resource, vbo, cudaGraphicsMapFlagsNone));
}

/*!
 * PBO,VBOバッファをCUDAから削除
 * @param[in] vbo VBO,PBO名
 */
void CuUnregisterGLBufferObject(cudaGraphicsResource *resource)
{
	RX_CUCHECK(cudaGraphicsUnregisterResource(resource));
}

/*!
 * デバイスからホストメモリへのコピー
 * @param[in] hDst コピー先ホストメモリ(最低size分確保されていること)
 * @param[in] dSrc コピー元デバイスメモリ
 * @param[in] vbo dSrcがVBOの場合，VBOのID．そうでない場合は0を指定
 * @param[in] size コピーサイズ(メモリ上のサイズ)
 */
void CuCopyArrayFromDevice(void* hDst, const void* dSrc, cudaGraphicsResource **resource, int offset, int size)
{   
	if(resource) dSrc = CuMapGLBufferObject(resource);

	RX_CUCHECK(cudaMemcpy(hDst, (char*)dSrc+offset, size, cudaMemcpyDeviceToHost));
	
	if(resource) CuUnmapGLBufferObject(*resource);
}

/*!
 * ホストからデバイスメモリへのコピー
 * @param[in] dDst コピー先デバイスメモリ(最低size分確保されていること)
 * @param[in] hSrc コピー元ホストメモリ
 * @param[in] offset コピー先オフセット
 * @param[in] size コピーサイズ(メモリ上のサイズ)
 */
void CuCopyArrayToDevice(void* dDst, const void* hSrc, int offset, int size)
{
	RX_CUCHECK(cudaMemcpy((char*)dDst+offset, hSrc, size, cudaMemcpyHostToDevice));
}

/*!
 * スレッド同期
 */
void CuThreadSync(void)
{
	RX_CUCHECK(cudaThreadSynchronize());
}

/*!
 * デバイスプロパティの表示
 */
void CuDeviceProp(void)
{
	int n;	//デバイス数
	RX_CUCHECK(cudaGetDeviceCount(&n));

	for(int i = 0; i < n; ++i){
		cudaDeviceProp dev;

		// デバイスプロパティ取得
		RX_CUCHECK(cudaGetDeviceProperties(&dev, i));

		printf("device %d\n", i);
		printf(" device name : %s\n", dev.name);
		printf(" total global memory : %d (MB)\n", dev.totalGlobalMem/1024/1024);
		printf(" shared memory / block : %d (KB)\n", dev.sharedMemPerBlock/1024);
		printf(" register / block : %d\n", dev.regsPerBlock);
		printf(" warp size : %d\n", dev.warpSize);
		printf(" max pitch : %d (B)\n", dev.memPitch);
		printf(" max threads / block : %d\n", dev.maxThreadsPerBlock);
		printf(" max size of each dim. of block : (%d, %d, %d)\n", dev.maxThreadsDim[0], dev.maxThreadsDim[1], dev.maxThreadsDim[2]);
		printf(" max size of each dim. of grid  : (%d, %d, %d)\n", dev.maxGridSize[0], dev.maxGridSize[1], dev.maxGridSize[2]);
		printf(" clock rate : %d (MHz)\n", dev.clockRate/1000);
		printf(" total constant memory : %d (KB)\n", dev.totalConstMem/1024);
		printf(" compute capability : %d.%d\n", dev.major, dev.minor);
		printf(" alignment requirement for texture : %d\n", dev.textureAlignment);
		printf(" device overlap : %s\n", (dev.deviceOverlap ? "ok" : "not"));
		printf(" num. of multiprocessors : %d\n", dev.multiProcessorCount);
		printf(" kernel execution timeout : %s\n", (dev.kernelExecTimeoutEnabled ? "on" : "off"));
		printf(" integrated : %s\n", (dev.integrated ? "on" : "off"));
		printf(" host memory mapping : %s\n", (dev.canMapHostMemory ? "on" : "off"));

		printf(" compute mode : ");
		if(dev.computeMode == cudaComputeModeDefault) printf("default mode (multiple threads can use) \n");
		else if(dev.computeMode == cudaComputeModeExclusive) printf("exclusive mode (only one thread will be able to use)\n");
		else if(dev.computeMode == cudaComputeModeProhibited) printf("prohibited mode (no threads can use)\n");
		
	}

	printf("Device with Maximum GFLOPS : %d\n", gpuGetMaxGflopsDeviceId());
}

/*!
 * thrust::exclusive_scanの呼び出し
 * @param[out] dScanData scan後のデータ
 * @param[in] dData 元データ
 * @param[in] num データ数
 */
void CuScan(unsigned int* dScanData, unsigned int* dData, unsigned int num)
{
	thrust::exclusive_scan(thrust::device_ptr<unsigned int>(dData), 
						   thrust::device_ptr<unsigned int>(dData+num),
						   thrust::device_ptr<unsigned int>(dScanData));
}

/*!
 * thrust::sort_by_keyによるハッシュ値に基づくソート
 * @param[in] dHash ハッシュ値
 * @param[in] dIndex インデックス(パーティクル，ポリゴンなど)
 * @param[in] num データ数
 */
void CuSort(unsigned int *dHash, uint *dIndex, uint num)
{
	thrust::sort_by_key(thrust::device_ptr<unsigned int>(dHash),
						thrust::device_ptr<unsigned int>(dHash+num),
						thrust::device_ptr<unsigned int>(dIndex));
}




}   // extern "C"
