#include"System.h"


FMOD::System* System::_system = nullptr;
FMOD_RESULT System::_result;

System::System()
{
	_result = System_Create(&_system);
}

System::~System()
{
	_result = _system->release();
}

void System::init()
{
	_result = System_Create(&_system);
	_result = _system->init(128, FMOD_INIT_NORMAL, 0);
}

void System::update()
{
	_system->update();
}
