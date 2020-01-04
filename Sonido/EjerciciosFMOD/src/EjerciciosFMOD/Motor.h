#pragma once
#include<fmod.hpp>
#include "Sonido.h";

class Motor {


private:
	int _currentRPM = 950;
	int _previousRPM = 0;

	Sonido* _1104;
	Sonido* _1560;
	Sonido* _1999;
	Sonido* _2549;
	Sonido* _2900;
	Sonido* _3472;

	const char* _1104_fileName = "res/1104.ogg";
	const char* _1560_fileName = "res/1560.ogg";
	const char* _1999_fileName = "res/1999.ogg";
	const char* _2549_fileName = "res/2549.ogg";
	const char* _2900_fileName = "res/2900.ogg";
	const char* _3472_fileName = "res/3472.ogg";

public:

	Motor(FMOD::System* system);
	~Motor();

	void StartEngine();
	void ManageInput();
	void UpdateEngine();//va a ser el que gestione la lógica del sonido, qué sonido suena y cual no.
	void IncreaseRPM();
	void DecreaseRPM();

	float I_Function(float a, float b);
	float O_Function(float a, float b);
};