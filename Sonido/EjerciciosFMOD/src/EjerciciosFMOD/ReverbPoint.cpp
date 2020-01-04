#include "ReverbPoint.h"
#include <iostream>

ReverbPoint::ReverbPoint(FMOD::System* system)
{
	_system = system;
	_result = _system->createReverb3D(&_reverb);

	//FMOD_VECTOR _reverbPosition = { 0.0f,0.0f,0.0f };

	//_minDistance = 5.0f;
	//_maxDistance = 10.0f;

	//_reverbPosition.x = 21.0f;
	//_reverbPosition.x = 6.0f;

	//_reverbProperty = FMOD_PRESET_BATHROOM;
	setProperty();
	setPositionalAtributes();
	_reverb->setActive(false);
}

ReverbPoint::~ReverbPoint()
{
	_reverb->release();
}

void ReverbPoint::setProperty()
{
	_result = _reverb->setProperties(&_reverbProperty);
}

void ReverbPoint::setPositionalAtributes()
{
	_result = _reverb->set3DAttributes(&_reverbPosition, _minDistance, _maxDistance );
}

void ReverbPoint::resetReverbProperty(FMOD_REVERB_PROPERTIES property)
{
	_reverbProperty = property;
	_result = _reverb->setProperties(&_reverbProperty);
}

void ReverbPoint::resetPositionalAtributes(float posX, float posY)
{
	_reverbPosition.x = posX;
	_reverbPosition.y = posY;
	setPositionalAtributes();
}


void ReverbPoint::increaseMinDistance()
{
	if (_maxDistance > _minDistance + 3.0f) {
		setPositionalAtributes();
		_minDistance++;
	}

	std::cout << "Maxima distancia : " << _maxDistance << " Minima distancia : " << _minDistance;
}

void ReverbPoint::increaseMaxDistance()
{
	_maxDistance++;
	setPositionalAtributes();
	std::cout << "Maxima distancia : " << _maxDistance << " Minima distancia : " << _minDistance;
}

void ReverbPoint::decreaseMindistance()
{
	if (_minDistance > 1.0f)
	{
		_minDistance--;
		setPositionalAtributes();
	}
	std::cout << "Maxima distancia : " << _maxDistance << " Minima distancia : " << _minDistance;
}

void ReverbPoint::decreaseMaxDistance()
{
	if (_maxDistance > _minDistance + 3.0f)
	{
		_maxDistance--;
		setPositionalAtributes();
	}
	std::cout << "Maxima distancia : " << _maxDistance << " Minima distancia : " << _minDistance;
}

void ReverbPoint::setActive()
{
	_reverb->getActive(&active);
	active = !active;
	_reverb->setActive(active);
}
