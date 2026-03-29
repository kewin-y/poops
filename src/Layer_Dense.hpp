#ifndef LAYER_DENSE_HPP
#define LAYER_DENSE_HPP

#include <Eigen/Core>

namespace unn
{
struct Layer_Dense {
  const unsigned int n_in;
  const unsigned int n_out;
  const unsigned int batch_size;

  // Each row of `weights` represents the weight for a specific output neuron
  Eigen::MatrixXd weights;

  // Column vector containing all the biases
  // Element `i` of `biases` corresponds to the bias of output neuron `i`
  Eigen::VectorXd biases;

  Layer_Dense(unsigned int n_in, unsigned int n_out, unsigned int batch_size);

  // Each column vector of `inputs` corresponds to a single sample
  Eigen::MatrixXd operator()(const Eigen::MatrixXd &inputs) const;
};
} // namespace unn

#endif // !LAYER_DENSE_HPP
