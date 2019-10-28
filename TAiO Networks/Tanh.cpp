#include "Tanh.h"

Tanh::Tanh(double tau)
	: tau(tau)
{
}

VectorXd Tanh::operator()(const VectorXd& const A) const
{
	return (((tau * A).array().tanh() + 1) / 2).matrix();
}

VectorXd Tanh::derivativeA(const VectorXd& const A, const VectorXd* const YApprox) const
{
	return tau / 2.0 * (1 - YApprox->cwiseProduct(*YApprox).array()).matrix();
}
