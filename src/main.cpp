#include "Dense_Layer.hpp"
#include "ReLU.hpp"
#include "Softmax.hpp"
#include <Eigen/Core>

int main()
{
        unsigned int nout = 2;
        unsigned int nin = 4;
        unsigned int batches = 2;

        unn::Dense_Layer layer1{nin, nout, batches};
        unn::ReLU relu{};
        unn::Softmax sm{};
        Eigen::MatrixXd inputs = Eigen::MatrixXd::Random(nin, batches);

        auto poop = layer1(inputs);
        auto pee = relu(poop);
        auto fart = sm(pee);

        return 0;
}
