#ifndef DENSE_LAYER_HPP
#define DENSE_LAYER_HPP

#include <Eigen/Core>

namespace unn
{
struct Dense_Layer {
    const unsigned int nin;
    const unsigned int nout;
    const unsigned int samples;

    // Each row in weights represents the weight for a specific output neuron
    Eigen::MatrixXd weights;

    // Column vector containing all the biases Entry `i` of biases corresponds to the bias of output neuron `i`
    Eigen::VectorXd biases;

    Dense_Layer(unsigned int nin, unsigned int nout, unsigned int samples);

    // Forward Pass
    Eigen::MatrixXd operator()(const Eigen::MatrixXd &inputs) const;
};
} // namespace unn

#endif // !DENSE_LAYER_HPP
