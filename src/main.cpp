#include "Dense_Layer.hpp"
#include "ReLU.hpp"
#include "Softmax.hpp"
#include <Eigen/Core>

int main()
{
    unsigned int nout = 2;
    unsigned int nin = 4;
    unsigned int samples = 2;

    unn::Dense_Layer layer1{nin, nout, samples};
    unn::ReLU relu{};
    unn::Softmax sm{};

    Eigen::MatrixXd inputs = Eigen::MatrixXd::Random(nin, samples);

    auto foot = layer1(inputs);
    auto bar = relu(foot);
    auto bz = sm(bar);

    return 0;
}
