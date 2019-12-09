#pragma once
#include <iostream>
#include <stdio.h>

#include <fmod_errors.h>
#include <fmod.hpp>


FMOD::Sound* loadSound(FMOD::System* system, FMOD_RESULT& result);

FMOD::Channel* playSound(FMOD::System* system, FMOD_RESULT& result, FMOD::Sound* sound);

void TogglePaused(FMOD::Channel* channel, FMOD_RESULT& result, bool& paused);

void mute(FMOD::Channel* channel, FMOD_RESULT& result);

void unmute(FMOD::Channel* channel, FMOD_RESULT& result);

void setVolume(FMOD::Channel* channel, float val, FMOD_RESULT& result);

bool manageInput(FMOD::Channel* channel, FMOD_RESULT& result, FMOD::System* system, bool& paused, float& volume);

void close();
