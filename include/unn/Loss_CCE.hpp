#ifndef UNN_LOSS_CCE_HPP
#define UNN_LOSS_CCE_HPP

#include "unn/Layer.hpp"
#include <Eigen/Core>

namespace unn
{
struct Loss_CCE : Layer {
  Loss_CCE(const Eigen::MatrixXd &y);

  Eigen::MatrixXd operator()(const Eigen::MatrixXd &inputs);
  void backward(const Eigen::MatrixXd &d_next) override;

private:
  bool is_sparse;

  // Forward Pass
  Eigen::MatrixXd y_true; // is_sparse == true -> shape(y_true) = (1, n_samples)
                          // is_sparse == false -> shape(y_true) = (n_classes, n_samples)
  Eigen::MatrixXd in;     // shape(in) = (n_classes, n_samples)
                          // `in` is `y_pred` -- the
};
} // namespace unn

#endif // UNN_LOSS_CCE_HPP
