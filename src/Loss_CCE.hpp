#ifndef LOSS_CCE_HPP
#define LOSS_CCE_HPP

#include <Eigen/Core>

namespace unn
{
class Loss_CCE
{
public:
  Loss_CCE() = default;

  // Each column vector of `y_pred` represents the prediction values
  // `y` must be a column vector containing a sparse representation of ground truth class data
  Eigen::VectorXd operator()(const Eigen::MatrixXd &y_pred, const Eigen::VectorXi &y) const;
};

} // namespace unn

#endif // !LOSS_CCE_HPP
