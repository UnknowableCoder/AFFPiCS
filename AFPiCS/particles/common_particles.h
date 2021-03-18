#ifndef AFFPICS_PARTICLES_COMMON_PARTICLES
#define AFFPICS_PARTICLES_COMMON_PARTICLES

/*!
  \file common_particles.h
  
  \brief The most common particles: electrons, protons.
  
  \author Nuno Fernandes
*/

#include "../header.h"
#include "particle_base.h"

namespace AFFPiCS
{
  namespace Particles
  {
    namespace Common
    {
      
      template <indexer, class> class Electron;
      template <indexer, class> class Proton;
      
      template <indexer num_dims, class derived = void> class Electron :
      public particle_base<num_dims, std::conditional_t<std::is_void_v<derived>, Electron<num_dims, derived>, derived>>
      {
        public:
                
        template <class system_info>
        CUDA_HOS_DEV FLType mass(const system_info &info) const
        {
          return info.units.m_e();
        }
        
        template <class system_info>
        CUDA_HOS_DEV FLType charge(const system_info &info) const
        {
          return -info.units.q_e();
        }
      };
      
      template <indexer num_dims, class derived = void> class Proton :
      public particle_base<num_dims, std::conditional_t<std::is_void_v<derived>, Proton<num_dims, derived>, derived>>
      {
        public:
        
        using particle_simple<num_dims>::particle_simple;
        //Inherit constructors.
        
        template <class system_info>
        CUDA_HOS_DEV FLType mass(const system_info &info) const
        {
          return info.units.m_p();
        }
        
        template <class system_info>
        CUDA_HOS_DEV FLType charge(const system_info &info) const
        {
          return info.units.q_e();
        }
      };
      
    }
  }
}

#endif