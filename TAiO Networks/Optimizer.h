#pragma once
#include <Eigen/Dense>

#include "Loss.h"
#include "Activation.h"

using namespace Eigen;

class Optimizer
{
public:
	virtual MatrixXd operator()(const Loss& const loss, const Activation& const activation, const VectorXd& const X, const VectorXd& const A, const VectorXd& const YApprox, const VectorXd& const Y, const MatrixXd& const W, const VectorXd& const B = VectorXd(), VectorXd* outB = nullptr) const = 0;
};

