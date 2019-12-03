#include "NetworkConfig.h"

Loss* NetworkConfig::createLoss(string lossType) const
{
	if (lossType == "MEAN_SQUARED")
	{
		return new Mse();
	}
	else
	{
		throw invalid_argument("Unknown loss function type");
	}
}

NetworkConfig::NetworkConfig(string lossType, double maxLoss, int maxIter, bool activateInput, bool bias, bool revertOnIncrease):
	loss(createLoss(lossType)),
	maxLoss(maxLoss),
	maxIter(maxIter),
	activateInput(activateInput),
	bias(bias),
	revertOnIncrease(revertOnIncrease)
{}
