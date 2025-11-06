#ifndef RELU_HPP
#define RELU_HPP

#include <Eigen/Core>

namespace unn
{
struct ReLU {
        ReLU() = default;
        Eigen::MatrixXd operator()(const Eigen::MatrixXd &inputs) const;
};
} // namespace unn

#endif // !RELU_HPP
