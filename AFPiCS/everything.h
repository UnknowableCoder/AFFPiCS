#ifndef AFFPICS_EVERYTHING
#define AFFPICS_EVERYTHING

#include "header.h"
#include "depositers/Esirkepov.h"
#include "depositers/NoDepositer.h"
#include "evolvers/FDTDEvolver.h"
#include "evolvers/NoEvolver.h"
#include "particle_shapes/polynomial.h"
#include "particle_shapes/splines.h"
#include "particles/particle_base.h"
#include "particles/particle_simple.h"
#include "particles/common_particles.h"
#include "pushers/simple_pusher.h"
#include "pushers/Boris.h"
#include "pushers/HigueraCary.h"
#include "pushers/Vay.h"
#include "pushers/Boris.h"
#include "pushers/NoPusher.h"
#include "system_info/system_info_base.h"
#include "system_info/system_info_constant.h"
#include "system_info/system_info_maker.h"
#include "system_info/periodic_boundary_conditions.h"
#include "system_info/reflecting_boundary_conditions.h"
#include "system_info/symbolic_shapes.h"
#include "system_info/symbolic_shapes_simple.h"
#include "system_info/yee_cell.h"

#include "simul.h"


#endif