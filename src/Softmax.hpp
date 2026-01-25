#include "Eigen/Core"
namespace unn
{
struct Softmax {
    Softmax() = default;
    Eigen::MatrixXd operator()(const Eigen::MatrixXd &inputs) const;
};
} // namespace unn
