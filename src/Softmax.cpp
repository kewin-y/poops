#include "Softmax.hpp"
#include "Eigen/Core"

namespace unn
{
Eigen::MatrixXd Softmax::operator()(const Eigen::MatrixXd &inputs) const
{
    // Column vector containing the maximum input of each sample
    Eigen::VectorXd maxes = inputs.colwise().maxCoeff();

    // Adjust inputs
    Eigen::MatrixXd adjusted = inputs.rowwise() - maxes.transpose();

    // Apply exponential
    adjusted = adjusted.array().exp();

    // Get sum of each sample
    Eigen::VectorXd s = adjusted.colwise().sum();

    // Softmax (hopefully)
    return adjusted.rowwise() / s.transpose();
}
} // namespace unn
