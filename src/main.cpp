#include "Layer_Dense.hpp"
#include "Loss_CCE.hpp"
#include "ReLU.hpp"
#include "Softmax.hpp"
#include <Eigen/Core>
#include <iostream>

int main()
{
  unsigned int nout = 2;
  unsigned int nin = 4;
  unsigned int samples = 2;

  unn::Layer_Dense layer1{nin, nout, samples};
  unn::ReLU relu{};
  unn::Softmax sm{};
  unn::Loss_CCE cce{};

  Eigen::MatrixXd inputs = Eigen::MatrixXd::Random(nin, samples);

  auto foot = layer1(inputs);
  auto bar = relu(foot);
  auto baz = sm(bar);

  Eigen::Matrix3d softmax_outputs{
      {0.7, 0.1, 0.02},
      {0.1, 0.5, 0.9},
      {0.2, 0.4, 0.08},
  };

  Eigen::Vector3i class_targets{0, 1, 1};
  // std::cout << softmax_outputs << std::endl;
  // std::cout << class_targets;
  auto foo = cce(softmax_outputs, class_targets);
  std::cout << foo << "\n";
  return 0;
}
