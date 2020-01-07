#pragma once
#include<fmod.hpp>
#include <iostream>
#include<fmod_errors.h>
class System {


public:
	static FMOD::System* _system;
	static FMOD_RESULT _result;

	System();
	~System();

	static void ERRCHECK(FMOD_RESULT result) {

		if (result != FMOD_OK) {
			std::cout << FMOD_ErrorString(result) << std::endl;
		}
	}
};