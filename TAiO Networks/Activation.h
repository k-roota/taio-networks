#pragma once
#include<Eigen/Dense>

using namespace Eigen;

class Activation
{
public:
	virtual VectorXd operator()(const VectorXd& const A) const = 0;
	virtual VectorXd derivativeA(const VectorXd& const A, const VectorXd* const YApprox = nullptr) const = 0;
};

