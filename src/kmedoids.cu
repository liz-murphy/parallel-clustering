#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>

typedef thrust::tuple<int,float> argMinType;

struct medoid_unary_op : public thrust::unary_function<argMinType, int>
{
    int nC;
    __host__ __device__
        medoid_unary_op(int _nC) : nC(_nC) {}

    __host__ __device__
        int operator()(argMinType &a) const
        {
            //argMinType tmp = thrust::get<0>(a);
            int raw_index = thrust::get<0>(a);
            //int row_index = thrust::get<1>(a);
            int row_index = raw_index/nC;
            return raw_index - row_index*nC;
        }
};

struct argMin : public thrust::binary_function
                <argMinType,argMinType,argMinType>
{
    __host__ __device__
        argMinType operator()(const argMinType& a, const argMinType &b) const
        {
            if (thrust::get<1>(a) < thrust::get<1>(b)){
                return a;
            } else {
                return b;
            }
        }

};

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"Actual error, GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// convert a linear index to a row index
template <typename T>
struct linear_index_to_row_index : public thrust::unary_function<T,T>
{
    T C; // number of columns

    __host__ __device__
        linear_index_to_row_index(T C) : C(C) {}

    __host__ __device__
        T operator()(T i)
        {
            return i / C;
        }
};

/* Works up to 65536 words */
__global__ void distance_to_medoids(const float *data_in, const float *medoids_in, float *distance_matrix, int nD)
{
    // map the two 2D indices to a single linear, 1D index
    int index = blockIdx.x*gridDim.y + blockIdx.y; //index_x + index_y;

    int data_index = blockIdx.x*nD + threadIdx.x;
    int medoid_index = blockIdx.y*nD + threadIdx.x;

    // map the two 2D block indices to a single linear, 1D block index
    float diff = data_in[data_index]-medoids_in[medoid_index];
    float result = diff*diff;

    // write out the result
    atomicAdd( &(distance_matrix[index]), result);
}

__global__ void all_element_distance(const float *data_in, const int *cluster_id, float *distance_matrix, int nD)
{
    // map the two 2D indices to a single linear, 1D index
    int index = blockIdx.x*gridDim.y + blockIdx.y; //index_x + index_y;

    int data1_index = cluster_id[blockIdx.x]*nD + threadIdx.x;
    int data2_index = cluster_id[blockIdx.y]*nD + threadIdx.x;

    // map the two 2D block indices to a single linear, 1D block index
    float diff = data_in[data1_index]-data_in[data2_index];
    float result = diff*diff;

    // write out the result
    atomicAdd( &(distance_matrix[index]), result);
}

thrust::device_vector<argMinType> minsOfRowSpace(thrust::device_ptr<float> A, int nRows, int nColumns) {
    // allocate storage for row argmins and indices
    thrust::device_vector<int> row_indices(nRows);          
    thrust::device_vector<argMinType> row_argmins(nRows);          

    // compute row argmins by finding argmin values with equal row indices
    thrust::reduce_by_key
        (thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(nColumns)),  // InputIterator keys_first
         thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(nColumns)) + (nRows*nColumns),  // InputIterator keys_last
         thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<int>(0),&A[0])),    // InputIterator values_first
         row_indices.begin(),   // OutputIterator keys_output
         row_argmins.begin(),   // OutputIterator values_output
         thrust::equal_to<int>(),   // binary predicate
         argMin());             // binary operation
    return row_argmins;
}

thrust::device_vector<float> sumOfRowSpace(thrust::device_ptr<float> A, int nRows, int nColumns) {
    // allocate storage for row argmins and indices
    thrust::device_vector<int> row_indices(nRows);          
    thrust::device_vector<float> row_sum(nRows);          

    // compute row argmins by finding argmin values with equal row indices
    thrust::reduce_by_key
        (thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(nColumns)),  // InputIterator keys_first
         thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(nColumns)) + (nRows*nColumns),  // InputIterator keys_last
         &A[0],
         row_indices.begin(),
         row_sum.begin(),
         thrust::equal_to<int>(),
         thrust::plus<float>());
    return row_sum;
}

/* Input is a matrix of n x D */
/* Assume medoids index in is already a random selection */
void kmedoids_wrapper(float *h_data_in, int *h_medoids_index_in, int *h_assignment_out, int data_size, int medoids_size, int Dim)
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

    // Compute distance of ith binary code to medoids in parallel
    distance_to_medoids<<<grid_dim,Dim>>>(d_data_in, d_medoids, d_distance_matrix, Dim);

    cudaThreadSynchronize();

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

    thrust::device_ptr<float> thrust_distance_matrix_ptr(d_distance_matrix);

    // Do parallel min-reduce to assign binary code i to closest medoid
    thrust::device_vector<argMinType> result= minsOfRowSpace(thrust_distance_matrix_ptr, data_size, medoids_size);

    cudaThreadSynchronize();


    std::cout << "Closest medoid by row:\n";
    for(int i=0; i < data_size; i++)
    {
        std::cout << "[";
        for(int j=0; j < medoids_size; j++)
            std::cout << thrust_distance_matrix_ptr[i*medoids_size + j] << "  ";
        argMinType temp = result[i];
        std::cout << "]   " << thrust::get<0>(temp)-i*medoids_size << "\n";
    }

    // Now have a vector of argMins, row index is into data, 0 col is closest medoid
    // Need to sort the data to bring equal elements together

    // Extract out the closest mediod indices
    thrust::device_vector<int> indices(data_size);
    thrust::transform(
            result.begin(),       
            result.end(),
            indices.begin(), 
            medoid_unary_op(medoids_size));

    thrust::device_vector<int> raw_indices(data_size);
    thrust::sequence(raw_indices.begin(), raw_indices.end());
    thrust::sort_by_key(indices.begin(), indices.end(), raw_indices.begin());

    std::cout << "Ordered indices:\n";
    for(int i=0; i < data_size; i++)
    {
        std::cout << indices[i] << "    " << raw_indices[i] << "\n";
    }

    int num_bins = medoids_size;
    thrust::device_vector<int> cumulative_histogram(num_bins);
    thrust::device_vector<int> histogram(num_bins);
    thrust::counting_iterator<int> search_begin(0);
    thrust::upper_bound(indices.begin(), indices.end(), search_begin, search_begin+num_bins, cumulative_histogram.begin());

    std::cout << "Cumulative histogram: \n";
    for(int i = 0; i < num_bins; i++)
    {
        std::cout << "Bin: " << i << "    Count:  " << cumulative_histogram[i] << "\n";
    }

    thrust::adjacent_difference(cumulative_histogram.begin(), cumulative_histogram.end(), histogram.begin());

    std::cout << "Histogram: \n";
    for(int i = 0; i < num_bins; i++)
    {
        std::cout << "Bin: " << i << "    Count:  " << histogram[i] << "\n";
    }

    int max_cluster_size = thrust::reduce(histogram.begin(), histogram.end(), -1, thrust::maximum<int>());


    int last_offset = 0;

    int h_new_medoids[medoids_size];

    // Allocate memory, compute distance matrix, do sum-reduce and then min-reduce
    for(int i=0; i< medoids_size; i++)
    {

        cudaThreadSynchronize();
        // Allocate the new distance matrix
        float *d_distance_matrix_cluster=0;
        gpuErrchk( cudaMalloc( (void **)&d_distance_matrix_cluster, histogram[i]*histogram[i]*sizeof(float) ) );
        gpuErrchk( cudaMemset( d_distance_matrix_cluster, 0x00, histogram[i]*histogram[i]*sizeof(float) ) );
        dim3 cluster_grid_dim;
        cluster_grid_dim.x = histogram[i];
        cluster_grid_dim.y = histogram[i];

        thrust::device_vector<int> cluster_indices(histogram[i]);
        thrust::copy(raw_indices.begin()+last_offset, raw_indices.begin()+cumulative_histogram[i], cluster_indices.begin());
        last_offset = cumulative_histogram[i];

        std::cout << "Raw indices for cluster: " << i << "\n";
        for(int z = 0; z < histogram[i]; z++)
            std::cout << cluster_indices[z] << "\n";

        all_element_distance<<<cluster_grid_dim,Dim>>>(d_data_in, cluster_indices.data().get(), d_distance_matrix_cluster, Dim);

        cudaThreadSynchronize();


        // sum reduce on the rows
        thrust::device_ptr<float> dptr(d_distance_matrix_cluster);
        thrust::device_vector<float> sum_result=sumOfRowSpace(dptr, histogram[i], histogram[i]);    // sum_result is a vector
        thrust::host_vector<float> H(dptr, dptr+histogram[i]*histogram[i]);
        std::cout << "Distance matrix for cluster " << i << ":\n";
        for(int j=0; j < histogram[i]; j++)
        {
            std::cout << "[";
            for(int k=0; k < histogram[i]; k++)
            {
                std::cout << H[j*histogram[i]+k] << "  ";
            }
            std::cout << "]    " << sum_result[j] << "\n";
        }

        std::cout << "Sum result:\n";
        for(int ii=0; ii < histogram[i]; ii++)
            std::cout << sum_result[ii] << "   ";

        thrust::device_vector<float>::iterator iter = thrust::min_element(sum_result.begin(), sum_result.end());
        unsigned int pos = iter - sum_result.begin();
        float min_val = *iter;
        std::cout << "\nMin index cluster " << i << " is " << pos << " value " << min_val << "\n";
        cudaThreadSynchronize();

        h_new_medoids[i] = cluster_indices[pos];

        std::cout << "New medoid for cluster " << i << " is " << h_new_medoids[i] << "\n";

        gpuErrchk(cudaFree(d_distance_matrix_cluster));
    }

    std::cout << "I am here\n";
    cudaThreadSynchronize();

    gpuErrchk(cudaFree(d_distance_matrix));
    gpuErrchk(cudaFree(d_data_in));
    gpuErrchk(cudaFree(d_medoids));

    std::cout << "At end\n";
}
