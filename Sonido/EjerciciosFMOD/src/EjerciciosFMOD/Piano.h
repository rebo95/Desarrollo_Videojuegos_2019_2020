#pragma once
#include <iostream>
#include <fmod.hpp>

class Piano {


private:
	float _pitch;
	float _octava;
public:
	Piano(FMOD::System* system);
	~Piano();

	FMOD::System* _system;
	FMOD::Sound* _sondio;
	FMOD::Channel* _channel;
	FMOD_RESULT _result;

	void Play();
	void SetPitch(float pitch);
	float calculatePitch(float i);

	void PalyKey(float pitch);

	void IncreaseOctave();
	void DecreaseOctave();

	void Teclado();
};