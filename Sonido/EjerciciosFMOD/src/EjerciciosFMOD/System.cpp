#include"System.h"


FMOD::System* System::_system = nullptr;
FMOD_RESULT System::_result;

System::System()
{
	//_result = System_Create(&_system);
	//ERRCHECK(_result);
}

System::~System()
{
	_result = _system->release();
	ERRCHECK(_result);
	
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

void System::ERRCHECK(FMOD_RESULT result)
{
	if (result != FMOD_OK) {
		std::cout << FMOD_ErrorString(result) << std::endl;
		// printf("FMOD error %d - %s", result, FMOD_ErrorString(result));
		exit(-1);
	}else std::cout << "FMOD_OK" << std::endl;

}
