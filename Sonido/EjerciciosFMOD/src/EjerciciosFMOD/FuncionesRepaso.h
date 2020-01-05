#pragma once

#include <fmod.hpp>
class FuncionesRepaso {

private:
	//----------------------------------------------------------------
	//INICIALIZADO Y CREACION DE LA CLASE ESTATICA SYSTEM
	//----------------------------------------------------------------
	//static FMOD::System* _system;
	//static FMOD_RESULT _result;
	//----------------------------------------------------------------

	FMOD::System* _system;
	FMOD::Channel* _channel;
	FMOD_RESULT _result;

public:
	FuncionesRepaso();
	~FuncionesRepaso();


	//static void fmod_init();
	//static void ERRCHECK(FMOD_RESULT result);
	//static void update();

	//SONIDO Y CARGA DE SONIDO 2D
	void createSound();
	void playSound();
	void stopSound();
	void pauseResume();
	void setVolume(float vol);
	void startPlayingSoundAt(float time);
	void loopSound(int loop);

	void setPitch(float pitch);
	void muteUnmute();
	void setPanorama(float panorama);


	//creación de un canal
	void createChannel();

	void trabajandoConTiempo();
};
