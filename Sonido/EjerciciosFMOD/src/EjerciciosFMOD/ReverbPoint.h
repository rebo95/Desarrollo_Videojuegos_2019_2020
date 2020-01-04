#pragma once
#include <fmod.hpp>
class ReverbPoint {
private:

	FMOD::System* _system;
	FMOD::Reverb3D* _reverb;
	FMOD_RESULT _result;
	FMOD_VECTOR _reverbPosition  = { 0.0f,0.0f,0.0f };

	float _minDistance = 5.0f;
	float _maxDistance = 50.0f;

	bool active = true;

	FMOD_REVERB_PROPERTIES _reverbProperty = FMOD_PRESET_BATHROOM;
public:
	ReverbPoint(FMOD::System* system);
	~ReverbPoint();

	void setProperty();
	void setPositionalAtributes();

	void resetReverbProperty(FMOD_REVERB_PROPERTIES property);
	void resetPositionalAtributes(float posX, float posY);

	void increaseMinDistance();
	void increaseMaxDistance();
	void decreaseMindistance();
	void decreaseMaxDistance();

	void setActive();
};