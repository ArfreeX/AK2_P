#pragma once

#include <chrono>
#include <iostream>

class CMeasure
{
public:
	typedef std::chrono::high_resolution_clock Clock;
	typedef std::chrono::nanoseconds TIME;

	 void start();
	 long long int elapsed();

private:
	 Clock::time_point beginning;
};

