
#include<fmod.hpp>
#include<iostream>
#include <fmod_errors.h>

static class System
{
private:

public:
	System();
	~System();

	static FMOD::System* _system;
	static FMOD_RESULT _result;

	static void init();
	static void update();
	static void ERRCHECK(FMOD_RESULT result);
};