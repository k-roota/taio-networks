#pragma once
#include <Eigen/Dense>

#include "Optimizer.h"

using namespace Eigen;

class GradientDescent : public Optimizer
{
private:
	double getLearningRate() const;
public:
	virtual MatrixXd operator()(const Loss& const loss, const Activation& const activation, const VectorXd& const X, const VectorXd& const A, const VectorXd& const YApprox, const VectorXd& const Y, const MatrixXd& const W) const;
	virtual MatrixXd operator()(const Loss& const loss, const Activation& const activation, const VectorXd& const X, const VectorXd& const A, const VectorXd& const YApprox, const VectorXd& const Y, const MatrixXd& const W, const VectorXd& const B, VectorXd& outB) const;
};

