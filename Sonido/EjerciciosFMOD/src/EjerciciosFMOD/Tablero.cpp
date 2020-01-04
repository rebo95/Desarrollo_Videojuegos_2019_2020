#include "Tablero.h"
#include <conio.h>

Tablero::Tablero(int fils, int cols, SoundSorce* soundSorce, Listener* listener, ReverbPoint* reverbPont1, ReverbPoint* reverbPont2, FMOD::System*system)
{

	_system = system;
	_fils = fils;
	_cols = cols;
	wallWidth = 10;
	wallHeight = 1;
	wallRow = 10;
	wallCol = 15;


	posXListener = 5.0f;
	posYListener = 0.0f;

	posXSorce = 10.0f;
	posYSorce = 5.0f;


	//creación de la geometría (se puede mejorar la limpieza)

	_result = _system->createGeometry(3,5,&_geometry);
	poligonIndex = 1;
	directOclusion = 1.0f;
	reverbOclusion = 1.0f;
	doubleSided = true;
	numVertices = 4;

	vA = {-5.0, 1.0f,0};
	vB = { -5.0f, -1.0,0};
	vC = { 5.0f, -1.0f, 0  };
	vD = { 5.0f, 1.0f,0 };

	vertices[0] = vA;
	vertices[1] = vB;
	vertices[2] = vC;
	vertices[3] = vD;

	wallPos = { 20.0f, 10.0f, 0.0 };
	_result = _geometry->setPosition(&wallPos);
	_result = _geometry->addPolygon(directOclusion, reverbOclusion, doubleSided, numVertices, vertices, &poligonIndex);

	//_geometry->setRotation();

	_source = soundSorce;
	setSourcePosition();
	_source->Play();

	_listener = listener;
	setListenerPosition();

	_reverb_1_posX = 21.0f;
	_reverb_1_posY = 6.0f;
	_reverb_2_posX = 12.0f;
	_reverb_2_posY = 14.0f;

	_reverbPont1 = reverbPont1;
	_reverbPont1->resetPositionalAtributes(_reverb_1_posX, _reverb_1_posY);

	_reverebPoint2 = reverbPont2;
	_reverbPont1->resetPositionalAtributes(_reverb_2_posX, _reverb_2_posY);


}

Tablero::~Tablero()
{
}

void Tablero::clear()
{
	system("CLS");
}

void Tablero::manageInput()
{
	if (_kbhit()) {
		int key = _getch();
		if (key == 'a') { posXListener --;}
		else if (key == 's') { posYListener++; }
		else if (key == 'd') { posXListener++; }
		else if (key == 'w') { posYListener--; }
		else if (key == 'j') { posXSorce--; }
		else if (key == 'k') { posYSorce++; }
		else if (key == 'l') { posXSorce++; }
		else if (key == 'i') { posYSorce--; }

		//setSourcePosition();
		setListenerPosition();
		render();
		std::cout << " ";

		if (key == 'e') { _source->DecreaseMinDistnace(); }
		else if (key == 'r') { _source->IncreaseMinDistnace(); }
		else if (key == 'y') { _source->DecreaseMaxDistnace(); }
		else if (key == 'u') { _source->IncreaseMaxDistnace(); }


		if (key == 'z') { _source->DecreaseInteriorConeAngle(); }
		else if (key == 'x') { _source->IncreaseInteriorConeAngle(); }
		else if (key == 'n') { _source->DecreaseExteriorConeAngle(); }
		else if (key == 'm') { _source->IncreaseExteriorConeAngle(); }

		if (key == '1') { _reverbPont1->decreaseMindistance(); }
		else if (key == '2') { _reverbPont1->increaseMinDistance(); }
		else if (key == '3') { _reverebPoint2->decreaseMaxDistance(); }
		else if (key == '4') { _reverebPoint2->increaseMaxDistance(); }
		else if (key == '9') { _reverbPont1->setActive(); }
		else if (key == '0') { _reverebPoint2->setActive(); }
		_source->moveSorce();
	}


}

void Tablero::setSourcePosition()
{
	_source->SetPosition(posXSorce ,posYSorce ,0);
}

void Tablero::setListenerPosition()
{
	_listener->SetListenerPosition(posXListener, posYListener, 0);
}

void Tablero::render()
{
	clear();

	for (int i = 0; i < _fils; i++) {
		for (int j = 0; j < _cols; j++) {

			if (i == posYListener && j == posXListener) std::cout << "L ";
			else if (i == posYSorce && j == posXSorce) std::cout << "S ";
			else if (i == _reverb_1_posY && j == _reverb_1_posX) std::cout << "1 ";
			else if (i == _reverb_2_posY && j == _reverb_2_posX) std::cout << "2 ";
			else if (i >= wallRow && i < wallRow+wallHeight && j >= wallCol && j < wallCol + wallWidth)  std::cout << "= ";
			else std::cout << ". ";
		}
		std::cout << "\n";
	}

}
