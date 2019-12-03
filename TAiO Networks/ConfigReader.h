#pragma once

#include <string>
#include <iostream>
#include <sstream>
#include <vector>

#include "TimeSeriesConfig.h"
#include "NetworkConfig.h"

using namespace std;

class ConfigReader
{
private:
	static const char COMMENT_SIGN;

	static string getLine(istream& stream);

public:
	static unique_ptr<TimeSeriesConfig> readTimeSeriesConfig(istream& stream);
	static unique_ptr<NetworkConfig> readNetworkConfig(istream& stream);
	
};

