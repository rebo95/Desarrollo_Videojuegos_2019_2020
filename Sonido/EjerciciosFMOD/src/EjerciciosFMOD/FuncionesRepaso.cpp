#include "FuncionesRepaso.h"
#include <iostream>
#include<fmod_errors.h>

#include <chrono>

FuncionesRepaso::FuncionesRepaso()
{
}

FuncionesRepaso::~FuncionesRepaso()
{
}

void FuncionesRepaso::createSound()
{
	FMOD::Sound* _sonido = nullptr;
	const char* _filename = "Sic Parvis Magna";

	_result = _system->createSound(_filename, FMOD_DEFAULT, 0, &_sonido);
	//para crear un stream utilizamos
	_result = _system->createStream(_filename, FMOD_DEFAULT, 0, &_sonido);
	// o también
	_result = _system->createSound(_filename, FMOD_CREATESTREAM, 0, &_sonido);
}
//otras posibilidades como FMOD_LOOP NORMAL, FMOD_3D ...

void FuncionesRepaso::playSound()
{
	FMOD::Sound* _sonido = nullptr;
	_result = _system->playSound(_sonido, 0, true, &_channel);//asignacion de un canal al sonido ->true implica que el sonido empieza pausado
	//tenemos cargado el sonido y en pause
}

void FuncionesRepaso::stopSound()
{
	_result = _channel->stop();//libera el canal
}

void FuncionesRepaso::pauseResume()
{
	bool pause;
	_result = _channel->getPaused(&pause);
	pause = !pause;
	//pausamos o despausamos en funcion del estado anterior de pausa
	_result = _channel->setPaused(pause);
}

void FuncionesRepaso::setVolume(float vol)
{
	_result = _channel->setVolume(vol);
}

void FuncionesRepaso::startPlayingSoundAt(float time)
{
	_result = _channel->setPosition(time, FMOD_TIMEUNIT_MS);
}

void FuncionesRepaso::loopSound(int loop)
{
	//loop =
	// 0 una vez loop
	// n se reproduce n+1 veces
	_result = _channel->setLoopCount(loop);

	//repeticion infinita
	_result = _channel->setLoopCount(0);
}

void FuncionesRepaso::setPitch(float pitch)
{
	_result = _channel->setPitch(pitch);
}

void FuncionesRepaso::muteUnmute()
{
	bool mute;
	_result = _channel->getMute(&mute);
	mute = !mute;
	_result = _channel->setMute(mute);

}

void FuncionesRepaso::setPanorama(float panorama)
{
	_result = _channel->setPan(panorama);
}

void FuncionesRepaso::createChannel()
{
	FMOD::ChannelGroup* _channelGroupMaster;
	const char* _masterChannelName = "canalMaster";

	//creación del canal
	_result = _system->createChannelGroup(_masterChannelName, &_channelGroupMaster);


	FMOD::ChannelGroup* _channelGroup;
	const char* _channelName = "canal1";

	//creación del canal
	_result = _system->createChannelGroup(_channelName, &_channelGroup);
	//añadimos chanel grup como hijo de master
	_channelGroup->addGroup(_channelGroupMaster);

	//para obtener una referencia a la raiz
	FMOD::ChannelGroup* _canalMaestro;
	_result = _system->getMasterChannelGroup(&_canalMaestro); //almacenará en _canal maestro la referencia al padre o raiz del grupo _channelGroupMaster


	//trabajando sobre canales como sobre sonidos

	_canalMaestro->setPaused(true);
	_canalMaestro->setPitch(1.0f);
	// ... y así con el mresto de atributos
}

void FuncionesRepaso::trabajandoConTiempo()
{
	//variables tiempo
	std::chrono::steady_clock::time_point timePoint1;
	std::chrono::steady_clock::time_point timePoint2;

	//diferencia temporal
	float diferenciaEnMilisegundos = std::chrono::duration_cast<std::chrono::milliseconds>(timePoint2 - timePoint1).count();

}

//----------------------------------------------------------------
//INICIALIZADO Y CREACION DE LA CLASE ESTATICA SYSTEM
//----------------------------------------------------------------
//void FuncionesRepaso::fmod_init()
//{
//	FMOD::System_Create(&_system);
//	_result = _system->init(128, FMOD_INIT_NORMAL, 0);
//}
//
//void FuncionesRepaso::update()
//{
//	_result = _system->update();
//}
//
//void FuncionesRepaso::ERRCHECK(FMOD_RESULT result)
//{
//	if (_result != FMOD_OK) {
//		std::cout << FMOD_ErrorString(_result) << std::endl;
//	}
//}
//----------------------------------------------------------------

