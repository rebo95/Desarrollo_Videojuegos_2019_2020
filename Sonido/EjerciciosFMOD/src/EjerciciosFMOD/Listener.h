#pragma once
#include<fmod.hpp>

class Listener {

private:
	FMOD_VECTOR _listenerPosition = {0, 0, 0};
	FMOD_VECTOR _listenerVelocity = {0, 0, 0};
	FMOD_VECTOR _listenerUp = {0, 0, 1};
	FMOD_VECTOR _listenerAt = {0, 1, 0};


	FMOD::System* _system;
	FMOD::Channel* _channel;

	FMOD_RESULT _resoult;

public:
	Listener(FMOD::System* system);
	~Listener();

	void SetListenerPosition(float x, float y, float z);
};