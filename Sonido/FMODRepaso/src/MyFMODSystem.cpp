#include "MyFMODSystem.h"

FMOD::System* MyFMODSystem::_system = nullptr;
FMOD_RESULT MyFMODSystem::_result;


MyFMODSystem::MyFMODSystem()
{


}

MyFMODSystem::~MyFMODSystem()
{

}

void MyFMODSystem::init()
{
	_result = System_Create(&_system);
	ERRCHECK(_result);
	_result = _system->init(128, FMOD_INIT_NORMAL, 0);
	ERRCHECK(_result);
}

void MyFMODSystem::update()
{
	_system->update();
}

void MyFMODSystem::ERRCHECK(FMOD_RESULT result)
{
	if (result != FMOD_OK) {
		std::cout << FMOD_ErrorString(result) << std::endl;
		exit(-1);
	}
	else std::cout << "FMOD_OK" << std::endl;
}
