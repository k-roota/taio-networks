#include "ConfigReader.h"

const char ConfigReader::COMMENT_SIGN = '#';

string ConfigReader::getLine(istream& stream)
{
	string line;

	do
	{
		if (!getline(stream, line))
		{
			throw invalid_argument("Not enough lines with data");
		}
		size_t first = line.find_first_not_of(" \t\r");
		if (first == std::string::npos)
		{
			line = "";
			continue;
		}
		size_t last = line.find_last_not_of(" \t\r");
		line = line.substr(first, (last - first + 1));
	} while (line.empty() || line[0] == COMMENT_SIGN);

	return line;
}

unique_ptr<TimeSeriesConfig> ConfigReader::readTimeSeriesConfig(istream& stream)
{
	string input = getLine(stream);
	istringstream inputStream(input);
	string matrixType;
	string dataPortionString;
	string timeOffsetString;
	inputStream >> matrixType >> dataPortionString >> timeOffsetString;

	int dataPortion = stoi(dataPortionString);
	int timeOffset = stoi(timeOffsetString);
	return make_unique<TimeSeriesConfig>(matrixType, dataPortion, timeOffset);
}

unique_ptr<NetworkConfig> ConfigReader::readNetworkConfig(istream& stream)
{
	string input = getLine(stream);
	istringstream inputStream(input);
	string errorType;
	string maxErrorString;
	string maxIterString;
	string activateInputString;
	string biasString;
	string revertString;
	inputStream >> errorType >> maxErrorString >> maxIterString >> activateInputString >> biasString >> revertString;

	double maxError = stod(maxErrorString);
	int maxIter = stoi(maxIterString);
	bool activateInput = stoi(activateInputString);
	bool bias = stoi(biasString);
	bool revertOnIncrease = stoi(revertString);

	return make_unique<NetworkConfig>(errorType, maxError, maxIter, activateInput, bias, revertOnIncrease);
}
