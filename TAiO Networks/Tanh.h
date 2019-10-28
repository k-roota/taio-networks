#pragma once
#include <Eigen/Dense>

#include "Activation.h"

using namespace Eigen;

class Tanh : public Activation
{
private:
	double tau;

public:
	Tanh(double tau = 5);

	virtual VectorXd operator()(const VectorXd& const A) const;
	virtual VectorXd derivativeA(const VectorXd& const A, const VectorXd* const YApprox) const;
};

