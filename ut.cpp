#include <iostream>
#include <Eigen/Dense>

#include "ukf.h"


using Eigen::Matrix;
using ukf::NumberT;
using ukf::kStDim;
using ukf::kObsDim;
using ukf::Gaussian;

template <typename Number,unsigned dim_st>
struct Predictor
{
    Matrix<Number,dim_st,1> operator()(const Matrix<Number,2*dim_st,1> &in)
    {
        Matrix<Number,dim_st,1> out = in.topLeftCorner(dim_st,1);
        return out;
    }

};
static Predictor<NumberT,kStDim> g_predictor;

template <typename  Number,unsigned dim_st,unsigned dim_obs>
struct Observer
{
    Observer():projector_(Matrix<Number,dim_obs,dim_st+dim_obs>::Random()){}
    Matrix<Number,dim_obs,1> operator()(const Matrix<Number,dim_st+dim_obs,1> &st)
    {
        return projector_*st;
    }
    Matrix<Number,dim_obs,dim_st+dim_obs> projector_;
};
static Observer<NumberT,kStDim,kObsDim> g_observer;

int main(int argc,char *argv[])
{
    Gaussian<NumberT,kStDim> x;
    Gaussian<NumberT,kStDim> w(0);
    Gaussian<NumberT,kObsDim> v(0);
    Matrix<NumberT,kObsDim,1> z;

    std::cout << "===== x:" << std::endl << x << std::endl;
    ukf::UKF<NumberT,kStDim,kObsDim> u(x,w,v,g_predictor,g_observer);
    for(unsigned i=0;i<2;++i)
        {
            Matrix<NumberT,kStDim+kObsDim,1> random_ext_st = Matrix<NumberT,kStDim+kObsDim,1>::Random();
            random_ext_st.topLeftCorner(kStDim,1) = random_ext_st.topLeftCorner(kStDim,1) + x.mean_;
            z  = g_observer(random_ext_st);
            std::cout << "===== x estimation: " << std::endl << u.step(z) << std::endl;
        }
}
