#pragma once
#include <Eigen/Dense>

#include "Loss.h"

using namespace Eigen;

class Mse : public Loss
{
public:
	virtual double operator()(const VectorXd& const YApprox, const VectorXd& const Y) const;
	virtual VectorXd derivativeY(const VectorXd& const YApprox, const VectorXd& const Y) const;
	virtual VectorXd derivativeB(const Activation& const activation, const VectorXd& const A, const VectorXd& const YApprox, const VectorXd& const Y) const;
	virtual MatrixXd derivativeW(const Activation& const activation, const VectorXd& const X, const VectorXd& const A, const VectorXd& const YApprox, const VectorXd& const Y, const VectorXd* const lossDB = nullptr) const;
};

