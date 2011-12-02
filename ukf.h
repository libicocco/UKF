#ifndef __UKF__UKF_H
#define __UKF__UKF_H

#include <functional>
#include <iostream>
#include <Eigen/Dense>

namespace ukf
{

    typedef double NumberT;
    static const unsigned kStDim = 2;
    static const unsigned kObsDim = 1;
    static const NumberT kAlpha = 0.001;
    static const NumberT kk = 0;
    static const NumberT kb = 2;

    using Eigen::Matrix;

    template<typename Number>
        inline void ComputeLambda(const Number &dim, Number &lambda)
        {
            lambda = kAlpha*kAlpha*(dim+kk)-dim;
        }

    template<typename Number>
        inline void ComputeWeights(const Number &dim, Number &w0c, Number &w0m, Number &wi)
        {
            Number lambda;
            ComputeLambda(dim,lambda);
            w0c =(lambda/(dim+lambda))+(1+kb-kAlpha*kAlpha);
            w0m =(lambda/(dim+lambda));
            wi = 1/(2*(dim+lambda));
        }

    template <typename Number,unsigned dim>
        void GeneratePSDMatrix(Matrix<Number,dim,dim> &psd)
    {
        psd = Matrix<Number,dim,dim>::Random();
        psd *= psd.transpose();
    }

    template<typename Number,unsigned dim>
        struct Gaussian
        {
        Gaussian(Number km,Number kv):mean_(Matrix<Number,dim,1>::Constant(km)),
                covar_(Matrix<Number,dim,dim>::Constant(kv))
            {}

        Gaussian(Number k):mean_(Matrix<Number,dim,1>::Constant(k))
            {
                GeneratePSDMatrix<Number,dim>(covar_);
            }

        Gaussian():mean_(100*Matrix<Number,dim,1>::Random())
            {
                GeneratePSDMatrix<Number,dim>(covar_);
            }

        Gaussian(const Matrix<Number,dim,1> &mean,
                 const Matrix<Number,dim,dim> &covar)
        : mean_(mean), covar_(covar)
            {}

            friend std::ostream &operator<<(std::ostream &out, const Gaussian<Number,dim> &g)
            {
                out << g.mean_ << std::endl << std::endl << g.covar_ << std::endl;
                return out;
            }

            Matrix<Number,dim,1> mean_;
            Matrix<Number,dim,dim> covar_;
        };

    template<typename Matrix>
        void ComputeMatSqrt(const Matrix &in, Matrix &sqrt)
        {
            //Eigen::SelfAdjointEigenSolver<Matrix> eigensolver(in);
            Eigen::EigenSolver<Matrix> eigensolver(in);
            if (eigensolver.info() != Eigen::Success)
                {
                    std::cerr << "problems computing the sqrt of covar" << std::endl;
                    exit(-1);
                }
            auto D = eigensolver.pseudoEigenvalueMatrix().cwiseSqrt();
            auto V = eigensolver.pseudoEigenvectors();
            sqrt = V*D*V.inverse();
        }

    template <typename Number,unsigned dim>
        void GeneratePoints(const Gaussian<Number,dim> &g,
                            Matrix<Number,dim,2*dim+1> &points)
    {
        Number lambda;
        ComputeLambda<Number>(dim,lambda);
        points.col(0) = g.mean_;
        Matrix<Number,dim,dim> sqrt;
        ComputeMatSqrt<Matrix<Number,dim,dim> >((dim+lambda)*g.covar_,sqrt);
        for(unsigned i=0;i<dim;++i)
            {
                points.col(i+1) = g.mean_ + sqrt.col(i);
                points.col(i+1+dim) = g.mean_ - sqrt.col(i);
            }
    }


    template<typename Number,unsigned dim_in,unsigned dim_out>
        void EstimateGaussian(const Matrix<Number,dim_out,2*dim_in+1> &points,
                              Gaussian<Number,dim_out> &g)
    {
        Number wi,w0c,w0m;
        ComputeWeights<Number>(dim_in,w0c,w0m,wi);
        g.mean_ = w0m*points.col(0);
        for(unsigned i=1;i<2*dim_in+1;++i)
            g.mean_ += wi*points.col(i);
        g.covar_ = w0c*((points.col(0)-g.mean_)*(points.col(0)-g.mean_).transpose());
        for(unsigned i=1;i<2*dim_in+1;++i)
            g.covar_ += wi*((points.col(i)-g.mean_)*(points.col(i)-g.mean_).transpose());
    }

    template<typename Number,unsigned dim_in,unsigned dim_out>
        void EstimateCrossCovar(const Matrix<Number,dim_in,1> &in_mean,
                                const Matrix<Number,dim_in,2*(dim_in+dim_out)+1> &in_points,
                                Matrix<Number,dim_out,1> &out_mean,
                                Matrix<Number,dim_out,2*(dim_in+dim_out)+1> &out_points,
                                Matrix<Number,dim_in,dim_out> &in_out_cross)
    {
        Number wi,w0c,w0m;
        ComputeWeights<Number>(dim_in+dim_out,w0c,w0m,wi);
        // XXX warning: transpose() is cast to take out the constness??
        in_out_cross = w0m*((in_points.col(0)-in_mean)*(out_points.col(0)-out_mean).transpose());
        for(unsigned i=1;i<2*dim_in+1;++i)
            in_out_cross += wi*((in_points.col(i)-in_mean)*(out_points.col(i)-out_mean).transpose());
    }

    template<typename Number,unsigned dim_in,unsigned dim_out>
        void ComputeUT(const Gaussian<Number,dim_in> &in,
                       const std::function<Matrix<Number,dim_out,1>(Matrix<Number,dim_in,1>) > &propagate,
                       Gaussian<Number,dim_out> &out,
                       Matrix<Number,dim_in,2*dim_in+1> &in_points,
                       Matrix<Number,dim_out,2*dim_in+1> &out_points)
    {
        GeneratePoints<Number,dim_in>(in,in_points);
        for(unsigned i=0;i<2*dim_in+1;++i)
            out_points.col(i) = propagate(in_points.col(i));
        EstimateGaussian<Number,dim_in,dim_out>(out_points,out);
    }

    template<typename Number,unsigned dim_in,unsigned dim_out>
        void ComputeExtUT(const Gaussian<Number,dim_in> &in,
                          const Gaussian<Number,dim_out> &out_noise,
                          const std::function<Matrix<Number,dim_out,1>(Matrix<Number,dim_in+dim_out,1>) > &propagate,
                          Gaussian<Number,dim_out> &out,
                          Matrix<Number,dim_in+dim_out,2*(dim_in+dim_out)+1> &in_points,
                          Matrix<Number,dim_out,2*(dim_in+dim_out)+1> &out_points)
    {
        Gaussian<Number,dim_in+dim_out> x_ext(0,0);
        x_ext.mean_ << in.mean_,out_noise.mean_;
        x_ext.covar_.topLeftCorner(dim_in,dim_in) = in.covar_;
        x_ext.covar_.bottomRightCorner(dim_out,dim_out) = out_noise.covar_;
        ComputeUT<Number,dim_in+dim_out,dim_out>(x_ext,propagate,out,in_points,out_points);
    }

    template<typename Number,unsigned dim_st,unsigned dim_obs>
        class UKF
    {
    public:
    UKF(const Gaussian<Number,dim_st> &x,
        const Gaussian<Number,dim_st> &w,
        const Gaussian<Number,dim_obs> &v,
        const std::function<Matrix<Number,dim_st,1>(Matrix<Number,2*dim_st,1>) > &predictor,
        const std::function<Matrix<Number,dim_obs,1>(Matrix<Number,dim_st+dim_obs,1>)> &observer):
        x_(x),w_(w),v_(v),predictor_(predictor),observer_(observer)
        {}

        const Gaussian<Number,dim_st> & step(const Matrix<Number,dim_obs,1> &z)
            {
                ComputeExtUT<Number,dim_st,dim_st>(x_,w_,predictor_,x_pred_,x_w_points_,x_pred_points_);
                ComputeExtUT<Number,dim_st,dim_obs>(x_pred_,v_,observer_,z_hat_,x_v_points_,z_hat_points_);
                EstimateCrossCovar<Number,dim_st,dim_obs>(x_pred_.mean_,
                                                          x_v_points_.topLeftCorner(dim_st,2*(dim_st+dim_obs)+1),
                                                          z_hat_.mean_,
                                                          z_hat_points_,xz_cross_);
                k_gain_ = xz_cross_*z_hat_.covar_.inverse();
                x_ = Gaussian<Number,dim_st>(x_pred_.mean_ + k_gain_*(z-z_hat_.mean_),
                                             x_pred_.covar_ - k_gain_*(z_hat_.covar_*k_gain_.transpose()));
                return x_;
            }

    private:
        Gaussian<Number,dim_st> x_;
        const Gaussian<Number,dim_st> w_;
        const Gaussian<Number,dim_obs> v_;
        const std::function<Matrix<Number,dim_st,1>(Matrix<Number,2*dim_st,1>)> predictor_;
        const std::function<Matrix<Number,dim_obs,1>(Matrix<Number,dim_st+dim_obs,1>)> observer_;

        // storage variables
        Gaussian<Number,dim_st> x_pred_;
        Gaussian<Number,dim_obs> z_hat_;
        Matrix<Number,dim_st,dim_obs> xz_cross_;
        Matrix<Number,dim_st+dim_st,2*(dim_st+dim_st)+1> x_w_points_;
        Matrix<Number,dim_st,2*(dim_st+dim_st)+1> x_pred_points_;
        Matrix<Number,dim_st+dim_obs,2*(dim_st+dim_obs)+1> x_v_points_;
        Matrix<Number,dim_obs,2*(dim_st+dim_obs)+1> z_hat_points_;
        Matrix<Number,dim_st,dim_obs> k_gain_;

    };

}//namespace ukf

#endif // __UKF__UKF_H
