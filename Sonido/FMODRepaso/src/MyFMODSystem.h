#pragma once

#include<fmod.hpp>
#include <iostream>
#include<fmod_errors.h>

class MyFMODSystem {

public:
	MyFMODSystem();
	~MyFMODSystem();

	static FMOD::System* _system;
	static FMOD_RESULT _result;

	static void init();
	static void update();
	static void ERRCHECK(FMOD_RESULT result);
};