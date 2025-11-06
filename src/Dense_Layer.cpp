#include "Dense_Layer.hpp"
#include "Eigen/Core"
#include <cassert>

namespace unn
{
Dense_Layer::Dense_Layer(unsigned int nin, unsigned int nout,
                         unsigned int batches)
    : nin{nin}, nout{nout}, batches{batches}
{
        weights = Eigen::MatrixXd::Random(nout, nin);
        biases = Eigen::VectorXd::Zero(nout);
}

Eigen::MatrixXd Dense_Layer::operator()(const Eigen::MatrixXd &inputs) const
{
        bool valid_in_nrows = inputs.rows() == nin;
        bool valid_in_ncols = inputs.cols() == batches;

        assert(
            ((valid_in_nrows && valid_in_ncols) && "inputs has invalid size"));

        auto outputs = (weights * inputs).colwise() + biases;

        bool valid_out_nrows;
        bool valid_out_ncols;

        valid_out_nrows = outputs.rows() == nout;
        valid_out_ncols = outputs.cols() == batches;

        assert(((valid_out_nrows && valid_out_ncols) &&
                "outputs has invalid size"));

        return outputs;
}
} // namespace unn
