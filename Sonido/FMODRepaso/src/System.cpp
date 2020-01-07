#include "System.h"

FMOD::System* System::_system = nullptr;
FMOD_RESULT System::_result;

System::System()
{
	FMOD::System_Create(&_system);
	_result = _system->init(128, FMOD_INIT_NORMAL, 0);
	ERRCHECK(_result);
}

System::~System()
{
}
