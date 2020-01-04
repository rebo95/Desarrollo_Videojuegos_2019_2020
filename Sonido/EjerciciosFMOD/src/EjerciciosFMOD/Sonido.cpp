#include "Sonido.h"
#include <conio.h>


Sonido::Sonido(const char* filename, FMOD::System* system)
{
	_system = system;
	_resoult = _system->createSound(filename, FMOD_LOOP_NORMAL, 0, &_sound);
	_resoult = _system->playSound(_sound, 0, true, &_channel);
	_vol = 1.0f;
	_panorama = 0.0f;
}

Sonido::~Sonido()
{
}

void Sonido::Play()
{
	//_resoult = _system->playSound(_sound, 0, false, &_channel);
	_resoult = _channel->setPaused(false);
}

void Sonido::Resume()
{
	_resoult = _channel->setPaused(false);
}

void Sonido::Stop()
{
	_resoult = _channel->setPaused(true);
	//_resoult = _channel->stop();
}

void Sonido::Pause()
{
	bool pause;
	_resoult = _channel->getPaused(&pause);

	if (pause) std::cout << "Resumiendo\n";
	else std::cout << "Pausando \n";

	pause = !pause;
	_resoult = _channel->setPaused(pause);
}

void Sonido::IncreaseVolume()
{
	std::cout << " Sube Volumen\n";
	if (_vol < 1.0f) { _vol += 0.1f; }
	else { std::cout << " Valor de sonido maximo alcanzado \n"; }

	_resoult = _resoult = _channel->setVolume(_vol);
}

void Sonido::DecreaseVolume()
{
	std::cout << " Baja volumen\n";
	if (_vol > 0.0f) { _vol -= 0.1f; }
	else { std::cout << " Valor de sonido minimo alcanzado \n"; }

	_resoult = _channel->setVolume(_vol);
}

void Sonido::SetVolume(float vol)
{
	_vol = vol;
	if (vol <= 1.0f && vol >= 0.0f) {
		std::cout << "Volumen = : " << _vol << "\n";
		_resoult = _channel->setVolume(vol);
	}
}

void Sonido::FadesVolumeSetter(float vol)
{
	if (vol <= 1.0f && vol >= 0.0f)
	_resoult = _channel->setVolume(vol);
}

void Sonido::PanoramaDcha()
{
	std::cout << " Panorama Derecha\n";
	if (_panorama < 1.0) { _panorama += 0.1f; }
	else { std::cout << " Valor de panorama dcho maximo alcanzado \n"; }
	_resoult = _channel->setPan(_panorama);
}

void Sonido::PanoramaIza()
{
	std::cout << " Panorama Izquierda\n";
	if (_panorama > -1.0) { _panorama -= 0.1f; }
	else { std::cout << " Valor de panorama izdo minimo alcanzado \n"; }
	_resoult = _channel->setPan(_panorama);
}

void Sonido::SetPanorama(float pan)
{
	_resoult = _channel->setPan(pan);
}

void Sonido::FadeIn(float delayTime)
{
	std::cout << " My Fade In \n";
	fadeIn = true;
	delay = delayTime;
	sysTime = std::chrono::steady_clock::now();
}

void Sonido::FadeOut(float delayTime)
{
	std::cout << " My Fade Out \n";
	fadeOut = true;
	delay = delayTime;
	sysTime = std::chrono::steady_clock::now();
}

void Sonido::FmodFadeIn(long long time)
{
	std::cout << " FMOD Fade In \n";
	unsigned long long parentClock;
	_resoult = _channel->getDSPClock(NULL, &parentClock);
	_resoult = _channel->addFadePoint(parentClock, 0.0f);
	_resoult = _channel->addFadePoint(parentClock + time, 1.0f);
}

void Sonido::FmodFadeOut(long long time)
{
	std::cout << " FMOD Fade Out \n";
	unsigned long long parentClock;
	_resoult = _channel->getDSPClock(NULL , &parentClock);
	_resoult = _channel->addFadePoint(parentClock, 1.0f);
	_resoult = _channel->addFadePoint(parentClock + time, 0.0f);
}

void Sonido::setPitch(float pitch)
{
	_pitch = pitch;
	_resoult = _channel->setPitch(_pitch);
}

float Sonido::getPitch()
{
	float currentPitch;
	_resoult = _channel->getPitch(&currentPitch);
	std::cout << " Pitch " << currentPitch << "\n";
	return currentPitch;
}


void Sonido::Update()
{
	if (fadeIn) {
		//se asume que el sonido comienza con volumen 0.0
		float timeActive = std::chrono::duration_cast<std::chrono::milliseconds>(sysTime2 - sysTime).count();;
		float vol = timeActive / delay;
		FadesVolumeSetter(vol);

		if (timeActive >= delay) { fadeIn = false; _vol = 1.0; }
		sysTime2 = std::chrono ::steady_clock ::now();
	}
	else if (fadeOut) {
		float timeActive = std::chrono::duration_cast<std::chrono::milliseconds>(sysTime2 - sysTime).count();
		float vol = (1 - (timeActive / delay)) * _vol;
		FadesVolumeSetter(vol);

		if (timeActive >= delay) { fadeOut = false; _vol = 0.0f; }
		sysTime2 = std::chrono::steady_clock::now();
	}


}

void Sonido::ManageInput(int key)
{
	//if (_kbhit()) {
	//	int key = _getch();

		if ((key == 'P') || (key == 'p')) {
			/*if (paused) Resume();*/
			/*else */Pause();
			//paused = !paused;
		}
		else if ((key == 'S') || (key == 's')) { Stop(); }
		else if ((key == 'D') || (key == 'd')) { Play(); }
		else if ((key == '+')) { IncreaseVolume(); }
		else if ((key == '-')) { DecreaseVolume(); }
		else if ((key == 'O') || (key == 'o')) { PanoramaDcha(); }
		else if ((key == 'I') || (key == 'i')) { PanoramaIza(); }
	//}
}
