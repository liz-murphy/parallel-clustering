#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <thrust/transform.h>
#include <set>

#define DEBUG 0

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

// convert a linear index to a column index
struct linear_index_to_col_index : public thrust::unary_function<int,int>
{
    int R; // number of columns

    __host__ __device__
        linear_index_to_col_index(int R) : R(R) {}

    __host__ __device__
        int operator()(int i)
        {
            return (i % R);
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

void print2Darray(float *array, int nrows, int ncols)
{
    for(int i=0; i < nrows; i++)
    {
        for(int j=0; j < ncols; j++)
        {
            std::cout << array[i*ncols+j] << "  ";
        }
        std::cout << "\n";
    }
}

void print2Darray(const float *array, int nrows, int ncols)
{
    for(int i=0; i < nrows; i++)
    {
        for(int j=0; j < ncols; j++)
        {
            std::cout << array[i*ncols+j] << "  ";
        }
        std::cout << "\n";
    }
}

void check_mem()
{
    size_t free_byte ;
    size_t total_byte ;

    cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;
    if ( cudaSuccess != cuda_status ){
        printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
        exit(1);
    }

    double free_db = (double)free_byte ;
    double total_db = (double)total_byte ;
    double used_db = total_db - free_db ;
    printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
            used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
}

/* Input is a matrix of n x D */
/* Assume medoids index in is already a random selection */
void kmedoids_wrapper(const float *const h_data_in, int * const h_medoids_index, int * const h_assignment_out, int data_size, int medoids_size, int Dim)
{
    float *d_data_in=0;
    gpuErrchk( cudaMalloc( (void **)&d_data_in, data_size*Dim*sizeof(float) ) );
    gpuErrchk(cudaMemcpy(d_data_in, h_data_in, data_size*Dim*sizeof(float), cudaMemcpyHostToDevice));

    float *d_distance_matrix=0;
    gpuErrchk(cudaMalloc((void **)&d_distance_matrix,data_size*medoids_size*sizeof(float)));
    gpuErrchk(cudaMemset(d_distance_matrix,0x00,data_size*medoids_size*sizeof(float)));

    float h_medoids[medoids_size*Dim];
    for(int i=0; i < medoids_size; i++)
    {
        int m = h_medoids_index[i];

        for(int j=0; j < Dim; j++)
        {
            h_medoids[i*Dim+j] = h_data_in[m*Dim+j];
        }
    }

    if(DEBUG)
    {
        std::cout << "Data:\n";
        print2Darray(h_data_in, data_size, Dim);
        std::cout << "Medoids:\n";
        print2Darray(h_medoids, medoids_size, Dim);
    }
   
    float *d_medoids=0;
    gpuErrchk(cudaMalloc((void **)&d_medoids,medoids_size*Dim*sizeof(float)));

    bool medoids_changed = true;

    std::set<int> h_old_medoids_index(h_medoids_index, h_medoids_index + medoids_size);
    std::set<int> h_new_medoids_index(h_medoids_index, h_medoids_index + medoids_size);

    int iterations=0;

    if(DEBUG)
    {
        std::cout << "Initial medoid indices: ";
        for(std::set<int>::iterator it=h_new_medoids_index.begin(); it != h_new_medoids_index.end(); ++it)
            std::cout <<  *it << ", ";
        std::cout << "\n";
    }

    thrust::device_vector<int> raw_indices(data_size);
    thrust::device_vector<int> indices(data_size);
    
    while(medoids_changed)
    {
        check_mem();

        iterations++;

        // Can do this on the device
        int i=0;
        for(std::set<int>::iterator it=h_new_medoids_index.begin(); it != h_new_medoids_index.end(); ++it)
        {
            int m = *it;

            for(int j=0; j < Dim; j++)
            {
                h_medoids[i*Dim+j] = h_data_in[m*Dim+j];
            }
            i++;
        }

        if(DEBUG)
        {
            std::cout << "New medoids: \n";
            print2Darray(h_medoids, medoids_size, Dim);
        }
        
        gpuErrchk(cudaMemcpy(d_medoids, h_medoids, medoids_size*Dim*sizeof(float), cudaMemcpyHostToDevice));

        check_mem();

        cudaThreadSynchronize();

        
        dim3 grid_dim;
        grid_dim.x = data_size;
        grid_dim.y = medoids_size;

        if(DEBUG)
            std::cout << "Grid is " << grid_dim.x << "x" << grid_dim.y << " with " << Dim << " threads\n";

        // Compute distance of ith binary code to medoids in parallel
        gpuErrchk(cudaMemset(d_distance_matrix,0x00,data_size*medoids_size*sizeof(float)));
        distance_to_medoids<<<grid_dim,Dim>>>(d_data_in, d_medoids, d_distance_matrix, Dim);

        check_mem();

        cudaThreadSynchronize();

        // intermediate debugging check

        thrust::device_ptr<float> thrust_distance_matrix_ptr(d_distance_matrix);

        check_mem();

        // Do parallel min-reduce to assign binary code i to closest medoid
        thrust::device_vector<argMinType> result= minsOfRowSpace(thrust_distance_matrix_ptr, data_size, medoids_size);

        check_mem();
        cudaThreadSynchronize();

        if(DEBUG)
        {
            std::cout << "Closest medoid by row:\n";
            for(int i=0; i < data_size; i++)
            {
                std::cout << "[";
                for(int j=0; j < medoids_size; j++)
                    std::cout << thrust_distance_matrix_ptr[i*medoids_size + j] << "  ";
                argMinType temp = result[i];
                std::cout << "]   " << thrust::get<0>(temp)-i*medoids_size << "\n";
            }
        }

        // Now have a vector of argMins, row index is into data, 0 col is closest medoid
        // Need to sort the data to bring equal elements together

        // Extract out the closest mediod indices
        //thrust::device_vector<int> indices(data_size);
        thrust::transform(
                result.begin(),       
                result.end(),
                indices.begin(), 
                medoid_unary_op(medoids_size));

        check_mem();
        //thrust::device_vector<int> raw_indices(data_size);
        thrust::sequence(raw_indices.begin(), raw_indices.end());
        try
        {
            thrust::sort_by_key(indices.begin(), indices.end(), raw_indices.begin());
        }
        catch(...)
        {
            std::cerr << "Cuda error after sort by key\n";
        }
        
        check_mem();

        if(DEBUG)
        {
            std::cout << "Ordered indices:\n";
            for(int i=0; i < data_size; i++)
            {
                std::cout << indices[i] << "    " << raw_indices[i] << "\n";
            }
        }

        int num_bins = medoids_size;
        thrust::device_vector<int> cumulative_histogram(num_bins);
        thrust::device_vector<int> histogram(num_bins);
        thrust::counting_iterator<int> search_begin(0);
        thrust::upper_bound(indices.begin(), indices.end(), search_begin, search_begin+num_bins, cumulative_histogram.begin());

        check_mem();
        
        if(DEBUG)
        {
            std::cout << "Cumulative histogram: \n";
            for(int i = 0; i < num_bins; i++)
            {
                std::cout << "Bin: " << i << "    Count:  " << cumulative_histogram[i] << "\n";
            }
        }

        thrust::adjacent_difference(cumulative_histogram.begin(), cumulative_histogram.end(), histogram.begin());

        check_mem();
        
        if(DEBUG)
        {
            std::cout << "Histogram: \n";
            for(int i = 0; i < num_bins; i++)
            {
                std::cout << "Bin: " << i << "    Count:  " << histogram[i] << "\n";
            }
        }

        int max_cluster_size = thrust::reduce(histogram.begin(), histogram.end(), -1, thrust::maximum<int>());

        int last_offset = 0;

        medoids_changed = false;
        h_old_medoids_index = h_new_medoids_index;
        h_new_medoids_index.clear();

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

            check_mem();
           
             thrust::device_vector<int> cluster_indices(histogram[i]);
            thrust::copy(raw_indices.begin()+last_offset, raw_indices.begin()+cumulative_histogram[i], cluster_indices.begin());
            last_offset = cumulative_histogram[i];

            check_mem();
            
            if(DEBUG)
            {
                std::cout << "Raw indices for cluster: " << i << "\n";
                for(int z = 0; z < histogram[i]; z++)
                    std::cout << cluster_indices[z] << "\n";
            }

            all_element_distance<<<cluster_grid_dim,Dim>>>(d_data_in, cluster_indices.data().get(), d_distance_matrix_cluster, Dim);

            cudaThreadSynchronize();

            // sum reduce on the rows
            thrust::device_ptr<float> dptr(d_distance_matrix_cluster);
            thrust::device_vector<float> sum_result=sumOfRowSpace(dptr, histogram[i], histogram[i]);    // sum_result is a vector
            thrust::host_vector<float> H(dptr, dptr+histogram[i]*histogram[i]);
            if(DEBUG)
            {
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
            }
            
            check_mem();
            thrust::device_vector<float>::iterator iter = thrust::min_element(sum_result.begin(), sum_result.end());
            unsigned int pos = iter - sum_result.begin();
            float min_val = *iter;
            
            if(DEBUG)
                std::cout << "\nMin index cluster " << i << " is " << pos << " value " << min_val << "\n";
            
            cudaThreadSynchronize();

            h_new_medoids_index.insert(cluster_indices[pos]);

            std::set<int>::iterator it = h_old_medoids_index.find(cluster_indices[pos]);
            if(it == h_old_medoids_index.end())
                medoids_changed = true;
            
            if(DEBUG)
                std::cout << "New medoid for cluster " << i << " is " << cluster_indices[pos] << "\n";

            gpuErrchk(cudaFree(d_distance_matrix_cluster));
        }
            std::cout << "At end of iteration " << iterations << ": ";
            for(std::set<int>::iterator it=h_new_medoids_index.begin(); it != h_new_medoids_index.end(); ++it)
                std::cout <<  *it << ", ";
            std::cout << "\n";
        /*else
        {
            std::cout << ".";
            std::cout.flush();
        }*/
        
        cudaThreadSynchronize();
        try
        {
            thrust::sort_by_key(raw_indices.begin(), raw_indices.end(), indices.begin());
        }
        catch(...)
        {
            std::cout << "Error is here\n";
        }
        thrust::copy(indices.begin(), indices.end(), &h_assignment_out[0]);
    }

    thrust::copy(h_new_medoids_index.begin(), h_new_medoids_index.end(), &h_medoids_index[0]);
    cudaThreadSynchronize();

    gpuErrchk(cudaFree(d_medoids));
    gpuErrchk(cudaFree(d_distance_matrix));
    gpuErrchk(cudaFree(d_data_in));

    std::cout << "Converged in " << iterations << " iterations\n";
}

void Plotter2D (float *data_in, int *cluster_centers, int *cluster_assignments, int nRows, int nCols)
{
    thrust::device_vector<float> dvec(data_in, data_in+nRows*nCols);

    thrust::device_vector<int> tfmd(nRows*nCols);

    thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(nRows*nCols), tfmd.begin(), linear_index_to_col_index(nCols));

    thrust::device_vector<int> keys_out(nCols);
    thrust::device_vector<float> min_vals_out(nCols);
    thrust::device_vector<float> max_vals_out(nCols);

    for(int i=0; i < dvec.size(); i+=2)
    {
        std::cout << dvec[i] <<  ": " << tfmd[i] << "    " << dvec[i+1] << ": " << tfmd[i+1] << "\n";
    }
    thrust::sort_by_key(tfmd.begin(), tfmd.end(), dvec.begin());
    
    for(int i=0; i < dvec.size(); i+=2)
    {
        std::cout << dvec[i] <<  ": " << tfmd[i] << "    " << dvec[i+1] << ": " << tfmd[i+1] << "\n";
    }
    
    thrust::reduce_by_key(tfmd.begin(),
            tfmd.end(),
            dvec.begin(),
            keys_out.begin(),
            min_vals_out.begin(),
            thrust::equal_to<int>(),
            thrust::minimum<float>());

    thrust::reduce_by_key(tfmd.begin(),
            tfmd.end(),
            dvec.begin(),
            keys_out.begin(),
            max_vals_out.begin(),
            thrust::equal_to<int>(),
            thrust::maximum<float>());
   }
