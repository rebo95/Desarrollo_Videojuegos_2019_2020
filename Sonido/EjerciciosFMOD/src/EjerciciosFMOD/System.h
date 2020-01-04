
#include<fmod.hpp>

static class System
{
private:

public:
	System();
	~System();

	static FMOD::System* _system;
	static FMOD_RESULT _result;

	static void init();
	static void update();
};