#include "ReLU.hpp"

namespace unn
{
Eigen::MatrixXd ReLU::operator()(const Eigen::MatrixXd &inputs) const
{
    // cwise means "coefficient-wise"
    return inputs.cwiseMax(0);
}
} // namespace unn
