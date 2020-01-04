#pragma once

#include <stdlib.h>
#include <iostream>
#include <fmod.hpp>

#include "SoundSorce.h"
#include "Listener.h"
#include "ReverbPoint.h"
#include <vector>

class Tablero {


private:

	int _fils;
	int _cols;

	int posXListener;
	int posYListener;

	int posXSorce;
	int posYSorce;

	int _reverb_1_posX;
	int _reverb_1_posY;

	int _reverb_2_posX;
	int _reverb_2_posY;

	int wallRow;
	int wallCol;

	int wallWidth;
	int wallHeight;

	SoundSorce* _source;
	Listener* _listener;
	ReverbPoint* _reverbPont1;
	ReverbPoint* _reverebPoint2;
	FMOD_RESULT _result;

	int poligonIndex = 1;
	float directOclusion = 1.0f;
	float reverbOclusion = 1.0f;
	bool doubleSided = true;
	int numVertices = 4;
	FMOD_VECTOR vA = {0.0f,0.0f,0.0f};
	FMOD_VECTOR vB = { 0.0f,0.0f,0.0f };
	FMOD_VECTOR vC = { 0.0f,0.0f,0.0f };
	FMOD_VECTOR vD = { 0.0f,0.0f,0.0f };

	FMOD_VECTOR _forwardGeometry;
	FMOD_VECTOR _upGeometry;

	FMOD_VECTOR vertices[4];

	FMOD_VECTOR wallPos = { wallCol + wallWidth / 2.0f, wallRow + wallHeight / 2.0f, 0.0 };

	FMOD::Geometry* _geometry;
	FMOD::System* _system;

public:
	Tablero(int fils, int cols, SoundSorce* soundSorce, Listener* listener, ReverbPoint* reverbPont1, ReverbPoint* reverbPont2, FMOD::System* system);
	~Tablero();

	void clear();
	void manageInput();
	void setSourcePosition();
	void setListenerPosition();
	void render();
};