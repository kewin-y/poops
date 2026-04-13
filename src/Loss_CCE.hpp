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
  // Each entry of `y` represents an "index of true probability"
  Eigen::VectorXd operator()(const Eigen::MatrixXd &y_pred, const Eigen::VectorXi &y) const;

  // Each column vector of `y_pred` represents the prediction values
  // `y` is a matrix whose rows are probability distributions; how data is formatted upon
  // calling pandas.get_dummies (I THINK ... I HAVE NO CLUE WHAT I'M TALKING ABOUT HAHA)
  Eigen::VectorXd operator()(const Eigen::MatrixXd &y_pred, const Eigen::MatrixXd &y) const;
};

} // namespace unn

#endif // !LOSS_CCE_HPP
