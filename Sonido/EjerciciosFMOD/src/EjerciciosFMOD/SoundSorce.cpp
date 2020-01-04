#include "SoundSorce.h"
#include <iostream>



SoundSorce::SoundSorce(const char* filename, FMOD::System* system)
{
	_system = system;
	_result = _system->createSound(filename, FMOD_3D|FMOD_LOOP_NORMAL, 0, &_sound);
	_vol = 1.0f;
}

SoundSorce::~SoundSorce()
{
}



void SoundSorce::Play()
{
	//arrancamos el canal
	_result = _system->playSound(_sound, 0, true, &_channel);
	//colocamos el canal
	_result = _channel->set3DAttributes(&_sorcePosition, &_sorceVel);
	//inicializamos los valores de máxima y minima distancia
	_result = _channel->set3DMinMaxDistance(_minDistance, _maxDistance);
	//inicializamos el volumen del canal
	_result = _channel->setVolume(_vol);
	//ajustes de cono
	setConeSettings();
	_result = _channel->set3DConeOrientation(&_coneDir);
	//deshacemos reproducimos
	_result = _channel->setPaused(false);
}

void SoundSorce::SetPosition(float x, float y, float z)
{
	_sorcePosition.x = x;
	_sorcePosition.y = -y;
	_sorcePosition.z = z;

	_result = _channel->set3DAttributes(&_sorcePosition, &_sorceVel);
}

void SoundSorce::SetVel(float x, float y, float z)
{
	_sorceVel.x = x;
	_sorceVel.y = -y;
	_sorceVel.z = z;

	_result = _channel->set3DAttributes(&_sorcePosition, &_sorceVel);
}

void SoundSorce::IncreaseMaxDistnace()
{
	if (_maxDistance < 30.0f) {
		_maxDistance++;
		_result = _channel->set3DMinMaxDistance(_minDistance, _maxDistance);
		std::cout << "Máxima distancia para atenuación incrementada a : " << _maxDistance;
	}
	else std::cout << "Máxima distancia no puede incrementar más";
}

void SoundSorce::DecreaseMaxDistnace()
{
	if (_maxDistance > 5.0f) {
		_maxDistance--;
		_result = _channel->set3DMinMaxDistance(_minDistance, _maxDistance);
		std::cout << "Máxima distancia para atenuación reducida a : " << _maxDistance;
	}
	else std::cout << "Máxima distancia no puede reducir más";
}


void SoundSorce::IncreaseMinDistnace()
{
	if (_minDistance < 5.0f) {
		_minDistance += 0.5;
		_result = _channel->set3DMinMaxDistance(_minDistance, _maxDistance);
		std::cout << "Máxima distancia para atenuación reducida a : " << _minDistance;
	}
	else std::cout << "Minima distancia no puede incrementar más";
}

void SoundSorce::DecreaseMinDistnace()
{
	if (_minDistance > 0.7f) {
		_minDistance -= 0.5f;
		_result = _channel->set3DMinMaxDistance(_minDistance, _maxDistance);
		std::cout << "Minima distancia para atenuación reducida a : " << _minDistance;
	}
	else std::cout << "Minima distancia no puede reducirse más";
}

void SoundSorce::IncreaseInteriorConeAngle()
{
	if (_internalConeAngle < _exteriorConeAngle - 5.0f) {
		_internalConeAngle++;
		setConeSettings();
	}
	else std::cout << "Cono interior no puede superar en 5 el ángulo de el cono exterior : Los angulos son I/E = " <<_internalConeAngle <<"/"<< _exteriorConeAngle;
	std::cout << "Los angulos son I/E = " << _internalConeAngle << "/" << _exteriorConeAngle;

}

void SoundSorce::DecreaseInteriorConeAngle()
{
	if (_internalConeAngle > 5.0f) {
		_internalConeAngle--;
		setConeSettings();
	}
	else std::cout << "Cono interior no puede ser inferior a 5 es: " << _internalConeAngle;
	std::cout << "Los angulos son I/E = " << _internalConeAngle << "/" << _exteriorConeAngle;
}

void SoundSorce::IncreaseExteriorConeAngle()
{
	if (_exteriorConeAngle < 360.0f) {
		_exteriorConeAngle++;
		setConeSettings();
	}
	else std::cout << "Cono exterior no puede superar en 360 es = " << _exteriorConeAngle;
	std::cout << "Los angulos son I/E = " << _internalConeAngle << "/" << _exteriorConeAngle;
}

void SoundSorce::DecreaseExteriorConeAngle()
{
	if (_internalConeAngle < _exteriorConeAngle - 5.0f) {
		_exteriorConeAngle--;
		setConeSettings();
	}
	else std::cout << "Cono exterior no puede superar en menos de 5 el cono interior : Los angulos son I/E = " << _internalConeAngle << "/" << _internalConeAngle;
	std::cout << "Los angulos son I/E = " << _internalConeAngle << "/" << _exteriorConeAngle;
}

void SoundSorce::getSorceMovementAtributes()
{
	_result = _channel->get3DAttributes(&_sorcePosition, &_sorceVel);
	std::cout << "PosiciónDeLaFuente : " << " x = " <<_sorcePosition.x << " y = "<<_sorcePosition.y <<" z ="<< _sorcePosition.z;
	std::cout << "PosiciónDeLaFuente : " << " x = " << _sorceVel.x << " y = " << _sorceVel.y << " z =" << _sorceVel.z;
}

void SoundSorce::setConeSettings()
{
	_result = _channel->set3DConeSettings(_internalConeAngle, _exteriorConeAngle, 0.0f);
}

