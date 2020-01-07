#pragma once

#include<fmod.hpp>
#include <iostream>
#include<fmod_errors.h>

class MyFMODSystem {

public:
	static FMOD::System* _system;
	static FMOD_RESULT _result;

	MyFMODSystem();
	~MyFMODSystem();

	static void ERRCHECK(FMOD_RESULT result);
	static void Update();
};