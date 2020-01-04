#include "Listener.h"

Listener::Listener(FMOD::System* system)
{
	_system = system;
}

Listener::~Listener()
{
}

void Listener::SetListenerPosition(float x, float y, float z)
{
	_listenerPosition.x = x;
	_listenerPosition.y = -y;
	_listenerPosition.z = z;
	_resoult = _system->set3DListenerAttributes(0, &_listenerPosition, &_listenerVelocity, &_listenerUp, &_listenerAt);
}
