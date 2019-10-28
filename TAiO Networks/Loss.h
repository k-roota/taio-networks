#pragma once
#include <Eigen/Dense>

#include "Activation.h"

using namespace Eigen;

class Loss
{
public:
	virtual VectorXd operator()(const VectorXd& const YApprox, const VectorXd& const Y) const = 0;
	virtual VectorXd derivativeY(const VectorXd& const YApprox, const VectorXd& const Y) const = 0;
	virtual VectorXd derivativeW(const Activation& const activation, const VectorXd& const X, const VectorXd& const A, const VectorXd& const YApprox, const VectorXd& const Y) const = 0;
};

