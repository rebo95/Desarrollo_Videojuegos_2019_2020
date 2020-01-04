
#include "System.h"
#include "Sonido.h"
#include "Piano.h"
#include "Tablero.h"
#include "Motor.h"

#include <conio.h>
#include <ctime>
#include <chrono>

int main() {

	System::init();

	//Sonido sonido1 = Sonido("res/RobertMiles-Children.ogg", System::_system);
	//sonido1.Play();

	//Sonido sonido2 = Sonido("res/140-ximpla.ogg", System::_system);
	////sonido2.Play();

	//Piano piano = Piano(System::_system);

	//Listener listener = Listener(System::_system);
	//SoundSorce sonido3D = SoundSorce("res/steps1.ogg", System::_system);

	//ReverbPoint _reverbPoint1 = ReverbPoint(System::_system);
	//ReverbPoint _reverbPoint2 = ReverbPoint(System::_system);

	//Tablero tablero = Tablero(20, 40, &sonido3D, &listener, &_reverbPoint1, &_reverbPoint2, System::_system);
	//tablero.render();


	bool fadeIn = false;
	bool fadeOut = false;

	//introducir en milisegundos
	float MyfadeTime = 5000.0f;
	float FmodFadeTime = 100000.0f;

	//Ejercicio de los motores:
	Motor motocicleta = Motor(System::_system);
	motocicleta.StartEngine();

	while (true) {


		//if (_kbhit()) {
		//	int key = _getch();
		////	if (key == 'q' || key == 'Q') break;
		////	if (key == 'f') { /*sonido1.FadeIn(MyfadeTime);*/sonido1.setPitch(2.0f); }
		////	if (key == 'F'){ sonido1.FmodFadeIn(FmodFadeTime); }
		////	if (key == 'g') {/* sonido1.FadeOut(MyfadeTime); */sonido1.setPitch(1.0f); /*sonido2.Play(); sonido2.SetVolume(0.0f); sonido2.FadeIn(fadeTime);*/}
		////	if (key == 'G'){ sonido1.FmodFadeOut(FmodFadeTime); }

		////	sonido1.ManageInput(key);

		//}


		//tablero.manageInput();


		////métodos update necesarios para los FadeInOut manuales implementados
		//sonido1.Update();
		//sonido2.Update();


		//piano.Teclado();



		//ejercicio del motor
		motocicleta.ManageInput();

		System::_result = System::_system->update();
	}

	return 0;
}

////float ppio = 0;
////std::chrono::steady_clock::time_point begin;
////begin = std::chrono::steady_clock::now();
////std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();


////end = std::chrono::steady_clock::now();
////std::cout << "Tiempo Tanscurrido = ";

////float p = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / 1000;
////std::cout << p;
////std::cout << "\n";