#include "Layer_Dense.hpp"
#include "Eigen/Core"
#include <cassert>

namespace unn
{
Dense_Layer::Dense_Layer(unsigned int n_in, unsigned int n_out,
                         unsigned int batch_size)
    : n_in{n_in}, n_out{n_out}, batch_size{batch_size}
{
    weights = Eigen::MatrixXd::Random(n_out, n_in);
    biases = Eigen::VectorXd::Zero(n_out);
}

Eigen::MatrixXd Dense_Layer::operator()(const Eigen::MatrixXd &inputs) const
{
    bool valid_in_nrows = inputs.rows() == n_in;
    bool valid_in_ncols = inputs.cols() == batch_size;

    assert(
        ((valid_in_nrows && valid_in_ncols) && "inputs has invalid size"));

    auto outputs = (weights * inputs).colwise() + biases;

    bool valid_out_nrows;
    bool valid_out_ncols;

    valid_out_nrows = outputs.rows() == n_out;
    valid_out_ncols = outputs.cols() == batch_size;

    assert(((valid_out_nrows && valid_out_ncols) &&
            "outputs has invalid size"));

    return outputs;
}
} // namespace unn
