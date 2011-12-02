#include <functional>
#include <vector>
#include <iostream>
#include <Eigen/Dense>

using Eigen::Matrix;

typedef float NumberT;
static const unsigned kStDim = 2;
static const unsigned kObsDim = 2;
static const float ka = 0.001;
static const float kk = 0;
static const float kb = 2;
static const float klam = ka*ka*(kStDim+kk)-kStDim;

template <typename Number,unsigned dim>
void GeneratePSDMatrix(Matrix<Number,dim,dim> &psd)
{
    psd = Matrix<Number,dim,dim>::Random();
    psd *= psd.transpose();
}

template<typename Number,unsigned dim>
struct Gaussian
{
    Gaussian(Number k):mean_(Matrix<Number,dim,1>::Constant(k)),
                       covar_(Matrix<Number,dim,dim>::Constant(k))
    {}

    Gaussian():mean_(Matrix<Number,dim,1>::Random())
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

template<typename Matrix>
void computeMatSqrt(const Matrix &in, Matrix &sqrt)
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
void GenerateX(const Gaussian<Number,dim> &x,
               Matrix<Number,dim,2*dim+1> &X)
{
    X.col(0) = x.mean_;
    auto tmp_mat = ((dim+klam)*x.covar_);
    for(unsigned i=0;i<dim;++i)
        {
            X.col(i+1) = x.mean_ + tmp_mat.col(i);
            X.col(i+1+dim) = x.mean_ - tmp_mat.col(i);
        }
}


template<typename Number,unsigned dim_in,unsigned dim_out>
void EstimateYmeanCovar(const Matrix<Number,dim_out,2*dim_in+1> &Y,
                        Gaussian<Number,dim_out> &y)
{
    y.mean_ = (klam/(dim_in+klam))*Y.col(0);
    Number W = 1/(2*(dim_in+klam));
    for(unsigned i=1;i<2*dim_in+1;++i)
        y.mean_ += W*Y.col(i);
    Number W0 =((klam/dim_in+klam)+(1+kb-ka*ka));
    y.covar_ = W0*((Y.col(0)-y.mean_)*(Y.col(0)-y.mean_).transpose());
    for(unsigned i=1;i<2*dim_in+1;++i)
        y.covar_ += W*((Y.col(i)-y.mean_)*(Y.col(i)-y.mean_).transpose());
}

template<typename Number,unsigned dim_in,unsigned dim_out>
void EstimateCrossCovar(const Matrix<Number,dim_in,1> &x_mean,
                        const Matrix<Number,dim_in,2*(dim_in+dim_out)+1> &X,
                        Matrix<Number,dim_out,1> &y_mean,
                        Matrix<Number,dim_out,2*(dim_in+dim_out)+1> &Y,
                        Matrix<Number,dim_in,dim_out> &xy_cross)
{
    Number W = 1/(2*(dim_in+klam));
    Number W0 =((klam/dim_in+klam)+(1+kb-ka*ka));
    xy_cross = W0*((X.col(0)-x_mean)*(Y.col(0)-y_mean).transpose());
    for(unsigned i=1;i<2*dim_in+1;++i)
      xy_cross += W*((X.col(i)-x_mean)*(Y.col(i)-y_mean).transpose());
}

// template<typename Number,unsigned dim_in,unsigned dim_out>
// void Propagate(const Matrix<Number,dim_in,2*dim_in+1> &X,
//                Matrix<Number,dim_out,2*dim_in+1> &Y)

template<typename Number,unsigned dim_in,unsigned dim_out>
void ComputeUT(const Gaussian<Number,dim_in> &x,
               const std::function<Matrix<Number,dim_out,1>(Matrix<Number,dim_in,1>) > &propagate,
               Gaussian<Number,dim_out> &y,
               Matrix<Number,dim_in,2*dim_in+1> &x_points,
               Matrix<Number,dim_out,2*dim_in+1> &y_points)
{
    GenerateX<Number,dim_in>(x,x_points);
    std::cout <<  "xpoints:" << std::endl << x_points << std::endl;
    for(unsigned i=0;i<2*dim_in+1;++i)
        y_points.col(i) = propagate(x_points.col(i));
    //    Propagate<Number,dim_in,dim_out>(x_points,y_points);
    EstimateYmeanCovar<Number,dim_in,dim_out>(y_points,y);
}

template<typename Number,unsigned dim_in,unsigned dim_out>
void process(const Gaussian<Number,dim_in> &x,
             const Gaussian<Number,dim_out> &w,
             const std::function<Matrix<Number,dim_out,1>(Matrix<Number,dim_in+dim_out,1>) > &propagate,
             Gaussian<Number,dim_out> &x_proc,
             Matrix<Number,dim_in+dim_out,2*(dim_in+dim_out)+1> &x_points,
             Matrix<Number,dim_out,2*(dim_in+dim_out)+1> &y_points)
{
    Gaussian<Number,dim_in+dim_out> x_ext(0);
    x_ext.mean_ << x.mean_,w.mean_;
    x_ext.covar_.topLeftCorner(dim_in,dim_in) = x.covar_;
    x_ext.covar_.bottomRightCorner(dim_out,dim_out) = w.covar_;
    std::cout <<  "x_ext:" << std::endl << x_ext << std::endl;
    ComputeUT<Number,dim_in+dim_out,dim_out>(x_ext,propagate,x_proc,x_points,y_points);
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
        x_(x),w_(w),v_(v),predictor_(predictor),observer_(observer_)
    {}

    Gaussian<Number,dim_st> & step(const Matrix<Number,dim_obs,1> &z)
    {
        process<Number,dim_st,dim_st>(x_,w_,predictor_,x_pred_,x_w_points_,x_pred_points_);
        std::cout << "x_pred:" << std::endl << x_pred_ << std::endl;
        process<Number,dim_st,dim_obs>(x_pred_,v_,observer_,z_hat_,x_v_points_,z_hat_points_);
        std::cout << "z_hat:" << std::endl << z_hat_ << std::endl;

        EstimateCrossCovar<Number,dim_st,dim_obs>(x_pred_.mean_,
                                                  x_v_points_.topLeftCorner(dim_st,2*(dim_st+dim_obs)+1),
                                                  z_hat_.mean_,
                                                  z_hat_points_,xz_cross_);

        k_gain_ = xz_cross_*z_hat_.covar_.inverse();
        std::cout << "k_gain:" << std::endl << k_gain_ << std::endl;
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
    Matrix<Number,dim_obs,2*(dim_st+dim_st)+1> x_pred_points_;
    Matrix<Number,dim_st+dim_obs,2*(dim_st+dim_obs)+1> x_v_points_;
    Matrix<Number,dim_obs,2*(dim_st+dim_obs)+1> z_hat_points_;
    Matrix<Number,dim_st,dim_obs> k_gain_;

};


int main(int argc,char *argv[])
{
    Gaussian<NumberT,kStDim> x,w,v;
    Matrix<NumberT,kObsDim,1> z = Matrix<NumberT,kObsDim,1>::Random();

    // Gaussian<NumberT,kStDim> x_est = x;
    // ukf<NumberT,kStDim,kObsDim>(x,w,v,z,
    //                             g_observer,
    //                             g_predictor,
    //                             x_est);
    // std::cout << x_est.mean_ << x_est.covar_ << std::endl;

    UKF<NumberT,kStDim,kObsDim> u(x,w,v,g_observer,g_predictor);
    std::cout << u.step(z) << std::endl;
}