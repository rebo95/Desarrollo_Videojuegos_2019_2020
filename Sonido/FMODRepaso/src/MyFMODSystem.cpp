#include "MyFMODSystem.h"

FMOD::System* MyFMODSystem::_system = nullptr;
FMOD_RESULT MyFMODSystem::_result;

MyFMODSystem::MyFMODSystem()
{
	FMOD::System_Create(&_system);
	_result = _system->init(128, FMOD_INIT_NORMAL, 0);
	ERRCHECK(_result);
}

MyFMODSystem::~MyFMODSystem()
{
	_result = _system->release();
	ERRCHECK(_result);
}

void MyFMODSystem::ERRCHECK(FMOD_RESULT result)
{
	if (result != FMOD_OK) {
		std::cout << FMOD_ErrorString(result) << std::endl;
	}
}

void MyFMODSystem::Update()
{
	_result = _system->update();
	ERRCHECK(_result);
}
