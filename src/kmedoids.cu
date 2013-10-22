#include <iostream>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"Actual error, GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void kernel(int *data, int size)
{   
    int index = blockIdx.x*gridDim.y + blockIdx.y;
    
    if(index < size)
        data[index] = index;
}

/* Works up to 65536 words */
__global__ void distance_to_medoids(const float *data_in, const float *medoids_in, float *distance_matrix, int nD)
{
//    int index_y = blockIdx.y*blockDim.x; 
 //   int index_x = blockIdx.x; 

    // map the two 2D indices to a single linear, 1D index
    int index = blockIdx.x*gridDim.y + blockIdx.y; //index_x + index_y;

    int data_index = blockIdx.x*nD + threadIdx.x;
    int medoid_index = blockIdx.y*nD + threadIdx.x;

    // map the two 2D block indices to a single linear, 1D block index
    float diff = data_in[data_index]-medoids_in[medoid_index];
    float result = diff*diff;
    //float result = medoids[medoid_index];

    // write out the result
    atomicAdd( &(distance_matrix[index]), result);
    //distance_matrix[0] = 99.0;


}

/* Must be a power of 2 */
/* Called multiple times for each row of the distance matrix */
__global__ void min_element_reduce(const float *data_in, float *d_dist_out, int *d_index_out)
{
    // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
    extern __shared__ float sdata[];
    extern __shared__ int idata[];

    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid  = threadIdx.x;

    // load shared mem from global mem
    sdata[tid] = data_in[myId];
    idata[tid] = myId;

    __syncthreads();            // make sure entire block is loaded!

    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            if(sdata[tid] > sdata[tid + s])
            {
                sdata[tid] = sdata[tid + s];
                idata[tid] = idata[tid + s];
            }
        }
        __syncthreads();        // make sure all adds at one stage are done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    {
        d_dist_out[blockIdx.x] = sdata[0];
        d_index_out[blockIdx.x] = idata[0];
    }
}

void min_element(const float *d_in, float *d_intermediate, int *d_index_out, int nCols)
{
    // assumes that size is not greater than maxThreadsPerBlock^2
    // and that size is a multiple of maxThreadsPerBlock
    const int maxThreadsPerBlock = 1024;
    int threads = maxThreadsPerBlock;
    int blocks = nCols / maxThreadsPerBlock;

    float *d_dist_intermediate=0;
    gpuErrchk(cudaMalloc((void **)d_dist_intermediate,nCols*sizeof(float)));

    int *d_index_intermediate=0;
    gpuErrchk(cudaMalloc((void **)d_index_intermediate,nCols*sizeof(float)));
    
    min_element_reduce<<<blocks, threads, threads * sizeof(float)>>>
        (d_in, d_dist_intermediate, d_index_intermediate);

    threads = blocks;
    blocks = 1;

    float *d_min_dist=0;

    min_element_reduce<<<blocks, threads, threads * sizeof(float)>>>
                            (d_intermediate, d_min_dist, d_index_out);

}

/* Input is a matrix of n x D */
/* Assume medoids index in is already a random selection */
void kmedoids_wrapper(float *h_data_in, int *h_medoids_index_in, int *h_assignment_out, int data_size, int medoids_size, int Dim)
//void kmedoids_wrapper()
{
    float *d_data_in=0;
    gpuErrchk( cudaMalloc( (void **)&d_data_in, data_size*Dim*sizeof(float) ) );
    gpuErrchk(cudaMemcpy(d_data_in, h_data_in, data_size*Dim*sizeof(float), cudaMemcpyHostToDevice));
   
    float *d_distance_matrix=0;
    gpuErrchk(cudaMalloc((void **)&d_distance_matrix,data_size*medoids_size*sizeof(float)));
    gpuErrchk(cudaMemset(d_distance_matrix,0x00,data_size*medoids_size*sizeof(float)));

    float medoids[medoids_size*Dim];
    for(int i=0; i < medoids_size; i++)
    {
        int m = h_medoids_index_in[i];

        for(int j=0; j < Dim; j++)
        {
            medoids[i*Dim+j] = h_data_in[m*Dim+j];
        }
    }

    std::cout << "Data:\n";
    for(int i=0; i < data_size; i++)
    {
        for(int j=0; j < Dim; j++)
        {
            std::cout << h_data_in[i*Dim + j] << " ";
        }
        std::cout << "\n";
    }

    std::cout << "Medoids:\n";
    for(int i=0; i < medoids_size; i++)
    {
        for(int j = 0; j < Dim; j++)
        {
            std::cout << medoids[i*Dim + j] << " ";
        }
        std::cout << "\n";
    }

    float *d_medoids=0;
    gpuErrchk(cudaMalloc((void **)&d_medoids,medoids_size*Dim*sizeof(float)));
    gpuErrchk(cudaMemcpy(d_medoids, medoids, medoids_size*Dim*sizeof(float), cudaMemcpyHostToDevice));

    dim3 grid_dim;
    grid_dim.x = data_size;
    grid_dim.y = medoids_size;

    std::cout << "Grid is " << grid_dim.x << "x" << grid_dim.y << " with " << Dim << " threads\n";
    distance_to_medoids<<<grid_dim,Dim>>>(d_data_in, d_medoids, d_distance_matrix, Dim);

    // intermediate debugging check
    float *h_distance_matrix=0;
    h_distance_matrix = (float *)malloc(data_size*medoids_size*sizeof(float));
    gpuErrchk(cudaMemcpy(h_distance_matrix, d_distance_matrix, data_size*medoids_size*sizeof(float), cudaMemcpyDeviceToHost));


    for(int i=0; i < data_size; i++)
    {
        for(int j=0; j < medoids_size; j++)
        {
            std::cout << h_distance_matrix[i*medoids_size+j] << "  ";
        }
        std::cout << "\n";
    }

/*
    // Sanity check
    float *h_distance_check = 0;
    h_distance_check = (float *)malloc(medoids_size*data_size*sizeof(float));
    gpuErrchk(cudaMemcpy(h_distance_check, d_distance_matrix, medoids_size*data_size*sizeof(float), cudaMemcpyDeviceToHost));
    for(int i=0; i < data_size*medoids_size; i++)
    {
        std::cout << h_distance_check[i] << "  ";
    }*/
    gpuErrchk(cudaFree(d_medoids));
    gpuErrchk(cudaFree(d_data_in));
    gpuErrchk(cudaFree(d_distance_matrix));

  /*  int rows=2; 
    int cols=3;
    
    int *d_data = 0; 
    cudaMalloc((void **)&d_data,rows*cols*sizeof(int));
    
    dim3 grid_dim;
    grid_dim.x = rows;
    grid_dim.y = cols;
    kernel<<<grid_dim,1>>>(d_data, rows*cols);
    
    int *h_data=0;
    h_data = (int *)malloc(rows*cols);
    cudaMemcpy(h_data,d_data,rows*cols*sizeof(int),cudaMemcpyDeviceToHost);
    for(int i=0; i < rows; i++)
    {   
        for(int j=0; j < cols; j++)
        {   
            std::cout << h_data[i*cols+j] << "    ";
        }
        std::cout << "\n";
    }
*/

}
/*
int main(void)
{
 kmedoids_wrapper();
}*/
