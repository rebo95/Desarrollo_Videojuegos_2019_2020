#pragma once

#include <iostream>
#include<fmod.hpp>
#include <chrono>

class Sonido
{
private:

	float _panorama;
	float _vol;
	bool paused;

	float _pitch;

	bool fadeIn;
	bool fadeOut;

	std::chrono::steady_clock::time_point sysTime;
	std::chrono::steady_clock::time_point sysTime2;

	float delay;

public:

	FMOD::System* _system;
	FMOD_RESULT _resoult;

	FMOD::Sound* _sound;
	FMOD::Channel* _channel;

	Sonido(const char* filename, FMOD::System* system); //La constructora tomará como argumento el archivo de sonido a reproducir y cargará ese sonido
	~Sonido();

	void Play();
	void Resume();
	void Stop();
	void Pause();
	void IncreaseVolume();
	void DecreaseVolume();
	void SetVolume(float vol);

	void FadesVolumeSetter(float vol);

	void PanoramaDcha();
	void PanoramaIza();
	void SetPanorama(float vol);
	void FadeIn(float time);
	void FadeOut(float time);
	void FmodFadeIn(long long time);
	void FmodFadeOut(long long time);

	void setPitch(float pitch);

	void Update();

	void ManageInput(int key);
};

