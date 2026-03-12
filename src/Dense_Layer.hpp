#ifndef DENSE_LAYER_HPP
#define DENSE_LAYER_HPP

#include <Eigen/Core>

namespace unn
{
struct Dense_Layer {
    const unsigned int n_in;
    const unsigned int n_out;
    const unsigned int n_samples;

    // Each row of `weights` represents the weight for a specific output neuron
    Eigen::MatrixXd weights;

    // Column vector containing all the biases Entry `i` of biases corresponds to the bias of output neuron `i`
    Eigen::VectorXd biases;

    Dense_Layer(unsigned int n_in, unsigned int n_out, unsigned int n_samples);

    // Forward Pass
    Eigen::MatrixXd operator()(const Eigen::MatrixXd &inputs) const;
};
} // namespace unn

#endif // !DENSE_LAYER_HPP
