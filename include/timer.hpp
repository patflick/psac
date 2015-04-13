/******************************************************************************\
 *                                                                            *
 * Copyright (c) 2012 Marat Dukhan                                            *
 *                                                                            *
 * This software is provided 'as-is', without any express or implied          *
 * warranty. In no event will the authors be held liable for any damages      *
 * arising from the use of this software.                                     *
 *                                                                            *
 * Permission is granted to anyone to use this software for any purpose,      *
 * including commercial applications, and to alter it and redistribute it     *
 * freely, subject to the following restrictions:                             *
 *                                                                            *
 * 1. The origin of this software must not be misrepresented; you must not    *
 * claim that you wrote the original software. If you use this software       *
 * in a product, an acknowledgment in the product documentation would be      *
 * appreciated but is not required.                                           *
 *                                                                            *
 * 2. Altered source versions must be plainly marked as such, and must not be *
 * misrepresented as being the original software.                             *
 *                                                                            *
 * 3. This notice may not be removed or altered from any source               *
 * distribution.                                                              *
 *                                                                            *
\******************************************************************************/

#pragma once

#ifdef _WIN32
	#include <windows.h>
#else
	#include <time.h>
#endif

#if 0
#define TIMER_START() timer _t; double _last_time = _t.get_ms();\
                      _last_time += 0.0; /* otherwise: warning: unused variable */ \
                      if (rank == 0) {\
                          fprintf(stderr, "-------- p = %d ---------\n", p);\
                          fflush(stderr);}

#define TIMER_END_SECTION(str) if (rank == 0) {\
                          fprintf(stderr, "TIMER\tSECTION\t%s\t%f\n", str,\
                                  _t.get_ms() - _last_time);\
                                  _last_time = _t.get_ms();\
                                  if(false){printf("%f",_last_time);} /* no warnings */\
                                  fflush(stderr);}
// for loops within the upper timer
#define TIMER_LOOP_START() timer _lt; double _llast_time = _lt.get_ms();
#define TIMER_END_LOOP_SECTION(iter, str) if (rank == 0) {\
                          fprintf(stderr, "TIMER\tLOOP\t%lu\t%s\t%f\n", iter, str,\
                                  _lt.get_ms() - _llast_time);\
                                  _llast_time = _lt.get_ms(); fflush(stderr);}
class timer {
public:
	timer();
	
	// Returns miliseconds since the timer was created
	double get_ms();
	
	~timer();
private:
#ifdef _WIN32
	LARGE_INTEGER creation_ticks;
	LARGE_INTEGER frequency;
#else
	struct timespec creation_time;
#endif
};
#endif
