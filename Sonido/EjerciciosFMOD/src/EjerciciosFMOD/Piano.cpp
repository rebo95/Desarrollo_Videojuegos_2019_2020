#include "Piano.h"
#include <conio.h>
#include <algorithm>

Piano::Piano(FMOD::System* system)
{
	_system = system;
	_result = _system->createSound("res/piano.ogg", FMOD_DEFAULT, 0, &_sondio);
	_octava = 0.0f;
}

Piano::~Piano()
{
}

void Piano::Play()
{
	_result = _system->playSound(_sondio, 0, false, &_channel);
}

void Piano::SetPitch(float pitch)
{
	_result = _channel->setPitch(pitch);
}

float Piano::calculatePitch(float i)
{
	float aux;

	aux = powf(2.0f, ((i + _octava * 12.0f )/12.0f));

	return aux;
}

void Piano::PalyKey(float pitch)
{
	FMOD::Channel* channel;
	_result = _system->playSound(_sondio, 0, true, &channel);
	_result = channel->setPitch(pitch);
	_result = channel->setPaused(false);
}

void Piano::IncreaseOctave()
{
	if (_octava < 3.0f) _octava++;
	std::cout << "Incrementado el valor de la octava a : " << _octava << "\n";
}

void Piano::DecreaseOctave()
{
	if (_octava > -3.0f) _octava--;
	std::cout << "Decrementado el valor de la octava a : " << _octava << "\n";
}


void Piano::Teclado()
{
	bool pianoKey = false;
	if (_kbhit()) {
		int key = _getch();
		
		switch (key)
		{

		case 'z': _pitch = calculatePitch(0.0f); pianoKey = true; break;
		case 'x': _pitch = calculatePitch(2.0f); pianoKey = true; break;
		case 'c': _pitch = calculatePitch(4.0f); pianoKey = true; break;
		case 'v': _pitch = calculatePitch(5.0f); pianoKey = true; break;
		case 'b': _pitch = calculatePitch(7.0f); pianoKey = true; break;
		case 'n': _pitch = calculatePitch(9.0f); pianoKey = true; break;
		case 'm': _pitch = calculatePitch(11.0f);pianoKey = true; break;
		case ',': _pitch = calculatePitch(12.0f);pianoKey = true; break;

		case 's': _pitch = calculatePitch(1.0f); pianoKey = true; break;
		case 'd': _pitch = calculatePitch(3.0f); pianoKey = true; break;
		case 'g': _pitch = calculatePitch(6.0f); pianoKey = true; break;
		case 'h': _pitch = calculatePitch(8.0f); pianoKey = true; break;
		case 'j': _pitch = calculatePitch(10.0f); pianoKey = true; break;



		case 'q': _pitch = calculatePitch(12.0f); pianoKey = true; break;
		case 'w': _pitch = calculatePitch(14.0f); pianoKey = true; break;
		case 'e': _pitch = calculatePitch(16.0f); pianoKey = true; break;
		case 'r': _pitch = calculatePitch(17.0f); pianoKey = true; break;
		case 't': _pitch = calculatePitch(19.0f); pianoKey = true; break;
		case 'y': _pitch = calculatePitch(21.0f); pianoKey = true; break;
		case 'u': _pitch = calculatePitch(23.0f); pianoKey = true; break;
		case 'i': _pitch = calculatePitch(24.0f); pianoKey = true; break;
			
		case '+': IncreaseOctave();  break;
		case '-': DecreaseOctave();  break;


		}

		if(pianoKey)
			PalyKey(_pitch);
	}
}
