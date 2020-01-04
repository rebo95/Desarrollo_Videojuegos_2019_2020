#include "Motor.h"
#include <conio.h>
#include <stdlib.h>
#include <math.h>

Motor::Motor(FMOD::System* system)
{
	_1104 = new Sonido(_1104_fileName, system);
	_1560 = new Sonido(_1560_fileName, system);;
	_1999 = new Sonido(_1999_fileName, system);;
	_2549 = new Sonido(_2549_fileName, system);;
	_2900 = new Sonido(_2900_fileName, system);;
	_3472 = new Sonido(_3472_fileName, system);;
}

Motor::~Motor()
{
}

void Motor::StartEngine()
{
	_1104->Resume();
	_1104->setPitch(1.0f + 0.1 * I_Function(0, 1000));
}

void Motor::ManageInput()
{
	if (_kbhit()) {
		int key = _getch();
		if (key == '+') {
			IncreaseRPM();
			UpdateEngine();
		}
		else if (key == '-') {
			DecreaseRPM();
			UpdateEngine();
		}
	}
}

void Motor::UpdateEngine()
{
	system("CLS");
	std::cout << "Revoluciones por minuto = : " << _currentRPM << "\n";

	if(_currentRPM < 1000)
		_1104->setPitch(1.0f + 0.1 * I_Function(0, 1000));

	if(_currentRPM >= 1000 && _currentRPM < 1400){//intervaloTransicion

		_1104->SetVolume(O_Function(1000, 1400));
		_1560->Play();
		_1560->SetVolume(I_Function(1000, 1400));
	}

	if (_currentRPM >= 1400 && _currentRPM < 1600) {
		_1104->Stop();
		_1560->setPitch(1.0f + 0.1 * I_Function(1400, 1600));
	}

	if (_currentRPM >= 1600 && _currentRPM < 2000) {	//intervaloTransicion

		_1560->SetVolume(O_Function(1600, 2000));
		_1999->Play();
		_1999->SetVolume(I_Function(1600, 2000));
	}

	if (_currentRPM >= 2000 && _currentRPM < 2200) {
		_1560->Stop();
		_1999->setPitch(1.0f + 0.1 * I_Function(2000, 2200));
	}

	if (_currentRPM >= 2200 && _currentRPM < 2600) {	//intervaloTransicion
		_1999->SetVolume(O_Function(2200, 2600));
		_2549->Play();
		_2549->SetVolume(I_Function(2200, 2600));
	}

	if (_currentRPM >= 2600 && _currentRPM < 2800) {
		_1999->Stop();
		_2549->setPitch(1.0f + 0.1 * I_Function(2600, 2800));
	}

	if (_currentRPM >= 2800 && _currentRPM < 3200) {	//intervaloTransicion
		_2549->SetVolume(O_Function(2800, 3200));
		_2900->Play();
		_2900->SetVolume(I_Function(2800, 3200));
	}

	if (_currentRPM >= 3200 && _currentRPM < 3400) {
		_2549->Stop();
		_2900->setPitch(1.0f + 0.1 * I_Function(3200, 3400));
	}

	if (_currentRPM >= 3400 && _currentRPM < 3800) {	//intervaloTransicion
		_2900->SetVolume(O_Function(3400, 3800));
		_3472->Play();
		_3472->SetVolume(I_Function(3400, 3800));
	}

	if (_currentRPM > 3800) {
		_2900->Stop();
		_3472->setPitch(1.0f + 0.1 * I_Function(3800, 4000));
	}

	//////if(_currentRPM < 1000)
	//////	_1560->Stop();

	//////if(_currentRPM >= 1000 && _currentRPM < 1400){
	//////	_1560->Play();
	//////	_1104->Play();
	//////}

	//////if (_currentRPM >= 1400 && _currentRPM < 1600) {
	//////	_1104->Stop();
	//////	_1560->Play();
	//////	_1999->Stop();
	//////}

	//////if (_currentRPM >= 1600 && _currentRPM < 2000) {
	//////	_1560->Play();
	//////	_1999->Play();
	//////}

	//////if (_currentRPM >= 2000 && _currentRPM < 2200) {
	//////	_1560->Stop();
	//////	_1999->Play();
	//////	_2549->Stop();
	//////}

	//////if (_currentRPM >= 2200 && _currentRPM < 2600) {
	//////	_1999->Play();
	//////	_2549->Play();
	//////}

	//////if (_currentRPM >= 2600 && _currentRPM < 2800) {
	//////	_1999->Stop();
	//////	_2549->Play();
	//////	_2900->Stop();
	//////}

	//////if (_currentRPM >= 2800 && _currentRPM < 3200) {
	//////	_2549->Play();
	//////	_2900->Play();
	//////}

	//////if (_currentRPM >= 3200 && _currentRPM < 3400) {
	//////	_2549->Stop();
	//////	_2900->Play();
	//////	_3472->Stop();
	//////}

	//////if (_currentRPM >= 3400 && _currentRPM < 3800) {
	//////	_2900->Play();
	//////	_3472->Play();
	//////}

	//////if (_currentRPM >= 3800) {
	//////	_2900->Stop();
	//////}

	_previousRPM = _currentRPM;
}

void Motor::IncreaseRPM()
{
	_currentRPM +=5;
}

void Motor::DecreaseRPM()
{
	if (_currentRPM > 0) _currentRPM -=5;
	else _currentRPM = 0;
}

float Motor::O_Function(float a, float b)
{
	float iV;

	float aux = 1.0f - ((_currentRPM - a) / (b - a));
	iV = sqrt(aux);
	return iV;
}

float Motor::I_Function(float a , float b)
{
	float Ov;

	float aux = ((_currentRPM - a) / (b - a));
	Ov = sqrt(aux);
	return Ov;
}



