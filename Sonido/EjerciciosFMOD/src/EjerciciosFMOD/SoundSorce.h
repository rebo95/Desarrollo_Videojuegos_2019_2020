#pragma once
#include <fmod.hpp>
#include "fmod_errors.h"

class SoundSorce {
private:

	FMOD_VECTOR _sorcePosition = { 0,0,0 };
	FMOD_VECTOR _sorceVel = { 0.0,0,0 };
	FMOD_VECTOR _coneDir = {0.0f, 1.0f, 0.0f}; // por defecto hacia adelante

	float _minDistance = 1.0f;
	float _maxDistance = 1000.0f;

	float _internalConeAngle = 360.0f;
	float _exteriorConeAngle = 360.0f;


	float _vol;
	void setConeSettings();

public :

	FMOD::System* _system;
	FMOD::Channel* _channel;
	FMOD::Sound* _sound;
	FMOD_RESULT _result;

	SoundSorce(const char* filename, FMOD::System* system);
	~SoundSorce();

	void Play();
	void SetPosition(float x, float y, float z);


	void IncreaseMaxDistnace();
	void DecreaseMaxDistnace();

	void IncreaseMinDistnace();
	void DecreaseMinDistnace();


	void IncreaseInteriorConeAngle();
	void DecreaseInteriorConeAngle();

	void IncreaseExteriorConeAngle();
	void DecreaseExteriorConeAngle();


	void getSorceMovementAtributes();
	void SetVel(float x, float y, float z);










};