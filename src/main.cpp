#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/flann/random.h>

extern void kmedoids_wrapper(float *h_data_in, int *h_medoids_index_in, int *h_assignment_out, int data_size, int medoids_size, int Dim);

int main(int argc, char **argv)
{
    int nMedoids = 4;
    int nData = 8;
    int nDim = 2;

    int medoids_in[nMedoids];

    cv::Mat training_data(nData,nDim,CV_32F);
    cv::randu(training_data,0,1);

    cvflann::UniqueRandom random_indices(nMedoids);

    for(int i=0; i < nMedoids; i++)
    {
        // Do some checks later
        medoids_in[i] = random_indices.next();
    }

    int assignment[nData];
    
    kmedoids_wrapper((float *)training_data.data, medoids_in, assignment, nData, nMedoids, nDim);
}

