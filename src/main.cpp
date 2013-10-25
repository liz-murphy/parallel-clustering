#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/flann/random.h>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Core>
#include "Color.h"

extern void kmedoids_wrapper(const float *const h_data_in, int *h_medoids_index_in, int *h_assignment_out, int data_size, int medoids_size, int Dim);

extern void Plotter2D (float *data_in, int *cluster_centers, int *cluster_assignments, int nRows, int nCols);

void PlotClusters(cv::Mat &data_set,const int *const medoids, const int *const assignments, int nData, int nMedoids);

int main(int argc, char **argv)
{
    int nMedoids = 2;
    int nData = 10000;
    int nDim = 8;

    int *medoids = new int(nMedoids);
    int *assignment = new int(nMedoids);

    cv::Mat training_data(nData,nDim,CV_32F);
    cv::randu(training_data,cv::Scalar(0),cv::Scalar(256));

    cvflann::UniqueRandom random_indices(nData);

    for(int i=0; i < nMedoids; i++)
    {
        // Do some checks later
        //medoids[i] = rand() % nData; //random_indices.next();
        medoids[i] = random_indices.next();
    }

    kmedoids_wrapper((float *)training_data.data, &medoids[0], &assignment[0], nData, nMedoids, nDim);

    //PlotClusters(training_data, &medoids[0], &assignment[0], nData, nMedoids);
   
    delete[] medoids;
    delete[] assignment;
}

void PlotClusters(cv::Mat &dataset, const int *const medoids, const int *const assignment, int nData, int nMedoids)
{
    float minx = std::numeric_limits<float>::max();
    float miny = std::numeric_limits<float>::max();
    float maxx = 0;
    float maxy = 0;

    for(int i=0; i < dataset.rows; i++)
    {
        cv::Mat tmp = dataset.row(i);
        if(tmp.at<float>(0,0) < minx)
            minx = tmp.at<float>(0,0);
        if(tmp.at<float>(0,0) > maxx)
            maxx = tmp.at<float>(0,0);
        if(tmp.at<float>(0,1) < miny)
            miny = tmp.at<float>(0,1);
        if(tmp.at<float>(0,1) > maxy)
            maxy = tmp.at<float>(0,1);
    }
    float xdim = maxx - minx;
    float ydim = maxy - miny;

    Eigen::MatrixXd colors(nMedoids,3);

    ColorPicker picker(nMedoids);

    cv::Mat img = cv::Mat::ones(1024,1024,CV_8UC3);
    
    for(int i=0; i < dataset.rows-1; i++)
    {
        cv::Mat tmp = dataset.row(i);
        float x = ((tmp.at<float>(0,0) - minx)/xdim)*1024;
        float y = ((tmp.at<float>(0,1) - miny)/ydim)*1024;
        cv::Point2f a(x,y);
        Color c = picker.getColor(assignment[i]);
        cv::circle(img, a, 2, cv::Scalar(c.r_,c.g_,c.b_), -1 );
    }
    for(int i=0; i < nMedoids; i++)
    {
        int center_ind = medoids[i];
        cv::Mat tmp = dataset.row(medoids[i]);
        float x = ((tmp.at<float>(0,0) - minx)/xdim)*1024;
        float y = ((tmp.at<float>(0,1) - miny)/ydim)*1024;
        cv::Point2f a(x,y);
        Color c = picker.getColor(assignment[medoids[i]]);
        cv::circle(img, a, 10, cv::Scalar(c.r_,c.g_,c.b_), -1 );
    }

    cv::imwrite("Clusters.jpg", img);
//    cv::imshow("Clusters",img);
 //   cv::waitKey(0);

}


