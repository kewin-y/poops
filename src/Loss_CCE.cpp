#include "Loss_CCE.hpp"
#include "Eigen/Core"
#include <iostream>
#include <string>

namespace unn
{

Eigen::VectorXd Loss_CCE::operator()(const Eigen::MatrixXd &y_pred, const Eigen::VectorXi &y) const
{
  bool valid_y_pred_rows = y_pred.rows() == y.rows();

  assert(((valid_y_pred_rows) && "mismatch in size between y_pred and y"));

  bool valid_y_range = y.minCoeff() >= 0 && y.maxCoeff() < y.rows();

  assert(((valid_y_range) && "invalid range of values exist in y"));

  Eigen::VectorXd extracted(y.rows());

  for (int i = 0; i < extracted.rows(); i++) {
    extracted(i) = y_pred(y(i), i);
  }

  // Values smaller than 1e-7 are replaced with 1e-7.
  // Values larger than 1 - 1e7 are replaced with 1 - 1e-7.
  Eigen::VectorXd clipped = extracted.cwiseMax(1e-7).cwiseMin(1 - 1e-7);

  return -clipped.array().log();
}
} // namespace unn
