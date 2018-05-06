#include "stdafx.h"


void CMeasure::start()
{
	beginning = Clock::now();
}

long long int CMeasure::elapsed()
{
	return std::chrono::duration_cast<TIME>(Clock::now() - beginning).count();
}
