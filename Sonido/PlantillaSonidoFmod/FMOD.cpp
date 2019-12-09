#include "FMOD.h"
#include <conio.h>

// para salidas de error
void ERRCHECK(FMOD_RESULT result) {
	if (result != FMOD_OK) {
		std::cout << FMOD_ErrorString(result) << std::endl;
		// printf("FMOD error %d - %s", result, FMOD_ErrorString(result));
		exit(-1);
	}
}

int main() {

	FMOD::System* system;
	FMOD_RESULT result;
	result = System_Create(&system); // Creamos el objeto system
	ERRCHECK(result);
	// 128 canales (numero maximo que podremos utilizar simultaneamente)
	result = system->init(128, FMOD_LOOP_NORMAL, 0); // Inicializacion de FMOD
	ERRCHECK(result);


	FMOD::Sound* sound;
	FMOD::Channel* channel;
	sound = loadSound(system, result);
	channel = playSound(system, result, sound);

	//while (true)
	//{
	//	char a;
	//	result = system->update();
	//	std::cin >> a;
	//	if (a == 'e') break;
	//	if (a == 'p') TogglePaused(channel);
	//	if (a == 'm') mute(channel);
	//	if (a == 'u') unmute(channel);
	//}



	//result = sound->release();
	//ERRCHECK(result);

	//close();

	printf("[P] Pausar/Despausar\n[+/-] Subir/bajar volumen\n[M] Mutear\n[U] Desmutear\n[Q] Salir\n");
	bool paused = false;
	float volume = 1.0f;

	while (true) {
		if (!manageInput(channel, result, system, paused, volume)) break;
	}

	return 0;
}

FMOD::Sound* loadSound(FMOD::System* system, FMOD_RESULT &result) {
	FMOD::Sound* sound1;
	result = system->createSound(
		"res/Battle.wav", // path al archivo de sonido
		FMOD_DEFAULT, // valores (por defecto en este caso: sin loop, 2D)
		0, // informacion adicional (nada en este caso)
		&sound1);

	return sound1;
}

FMOD::Channel* playSound(FMOD::System* system, FMOD_RESULT& result, FMOD::Sound* sound) {
	FMOD::Channel* channel;
	result = system->playSound(
		sound, // buffer que se "engancha" a ese canal
		0, // grupo de canales, 0 sin agrupar (agrupado en el master)
		false, // arranca sin "pause" (se reproduce directamente)
		&channel); // devuelve el canal que asigna
		// el sonido ya se esta reproduciendo
	return channel;
}

void TogglePaused(FMOD::Channel* channel, FMOD_RESULT& result, bool& paused) {
	channel->getPaused(&paused); ERRCHECK(result);
	channel->setPaused(!paused); ERRCHECK(result);

	if(paused) printf("Resume\n");
	else printf("Paused\n");
}

void mute(FMOD::Channel* channel, FMOD_RESULT& result) {
	channel->setMute(true); ERRCHECK(result);
	printf("Muteado\n");
}

void unmute(FMOD::Channel* channel, FMOD_RESULT& result) {
	channel->setMute(false); ERRCHECK(result);
	printf("Desmuteado\n");
}

void setVolume(FMOD::Channel* channel, float val, FMOD_RESULT& result) {
	channel->setVolume(val); ERRCHECK(result);
	printf("Volume: %f\n", val);
}

bool manageInput(FMOD::Channel* channel, FMOD_RESULT& result, FMOD::System* system, bool& paused, float& volume) {

	if (_kbhit()) {
		int key = _getch();
		if ((key == 'P') || (key == 'p')) {
			TogglePaused(channel, result, paused);
		}
		else if (key == '+') {
			if (volume < 1.0) {
				volume = volume + 0.1;
				setVolume(channel, volume, result);
			} else printf("Maximo volumen alcanzado\n");
		}
		else if (key == '-') {
			if (volume > 0) {
				volume = volume - 0.1;
				setVolume(channel, volume, result);
			}else printf("Minimo volumen alcanzado\n");
		}
		else if ((key == 'M') || (key == 'm')) {
			mute(channel, result);
		}
		else if ((key == 'U') || (key == 'u')) {
			unmute(channel, result);
		}
		else if ((key == 'Q') || (key == 'q')) return false;
	}
	result = system->update();
	return true;
}

void close() {
	FMOD::System* system;
	FMOD_RESULT result;
	result = System_Create(&system);

	result = system->release();
	ERRCHECK(result);
}