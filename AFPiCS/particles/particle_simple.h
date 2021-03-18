#ifndef AFFPICS_PARTICLES_PARTICLE_SIMPLE
#define AFFPICS_PARTICLES_PARTICLE_SIMPLE

/*!
  \file particle_simple.h
  
  \brief A class that specifies an instance of a particle with fixed charge and rest mass.
  
  \author Nuno Fernandes
*/

#include "../header.h"
#include "particle_base.h"

namespace AFFPiCS
{
  namespace Particles
  {    
    template <indexer num_dims, class derived = void>
    class particle_simple : 
    public particle_base<num_dims, std::conditional_t<std::is_void_v<derived>, particle_simple<num_dims, derived>, derived>>
    {
      private:
      
      using deriv_t = std::conditional_t<std::is_void_v<derived>, particle_simple, derived>;
    
      protected:
      
      vector_type<indexer, num_dims> grid_position;
      
      vector_type<FLType, num_dims> position;
      //In units of cell separation
      
      vector_type<FLType, num_dims> mom_over_mass;
      //In whatever units specified by system_info.

      public:
                                   
      CUDA_HOS_DEV particle_simple(const vector_type<indexer, num_dims> &cll = vector_type<FLType, num_dims>(0),
                                   const vector_type<FLType, num_dims> ps = vector_type<FLType, num_dims>(FLType(0.)),
                                   const vector_type<FLType, num_dims> mm = vector_type<FLType, num_dims>(FLType(0.)) ):
      grid_position(cll), position(ps), mom_over_mass(mm)
      {
      }
    
      template <class system_info>
      CUDA_HOS_DEV vector_type<FLType, num_dims> u(const system_info &info) const
      {
        return mom_over_mass;
      }
      
      template <class system_info>
      CUDA_HOS_DEV vector_type<FLType, num_dims> pos(const system_info &info) const
      {
        return position;
      }
      
      template <class system_info>
      CUDA_HOS_DEV vector_type<indexer, num_dims> cell(const system_info &info) const
      {
        return grid_position;
      }     
      
      template <class system_info>
      CUDA_HOS_DEV FLType gamma(system_info &info) const
      {
        const deriv_t* dhis = static_cast<const deriv_t*>(this);
        using namespace std;
        return sqrt(dhis->u(info).square_norm2()/info.units().c()/info.units().c()+FLType(1));
      }
      
      template <class system_info>
      CUDA_HOS_DEV void set_u(const vector_type<FLType, num_dims>& new_u, const system_info &info)
      {
        mom_over_mass = new_u;
      }
      
      template <class system_info>
      CUDA_HOS_DEV void set_pos(const vector_type<FLType, num_dims>& new_pos, const system_info &info)
      {
        position = new_pos;
      }
      
      template <class system_info>
      CUDA_HOS_DEV void set_cell(const vector_type<indexer, num_dims>& new_cell, const system_info &info)
      {
        grid_position = new_cell;
      }
      
      template<class stream, class str = std::basic_string<typename stream::char_type>>
      CUDA_ONLY_HOS void textual_output(stream &s, const str& separator = " ") const
      {
        g24_lib::textual_output(s, grid_position, separator);
        s << separator;
        g24_lib::textual_output(s, position, separator);
        s << separator;
        g24_lib::textual_output(s, mom_over_mass, separator);
      }
      
      template<class stream>
      CUDA_ONLY_HOS void binary_output(stream &s) const
      {
        g24_lib::binary_output(s, grid_position);
        g24_lib::binary_output(s, position);
        g24_lib::binary_output(s, mom_over_mass);
      }
      
      template<class stream>
      CUDA_ONLY_HOS void textual_input(stream &s)
      {
        g24_lib::textual_input(s, grid_position);
        g24_lib::textual_input(s, position);
        g24_lib::textual_input(s, mom_over_mass);
      }
      
      template<class stream>
      CUDA_ONLY_HOS void binary_input(stream &s)
      {
        g24_lib::binary_input(s, grid_position);
        g24_lib::binary_input(s, position);
        g24_lib::binary_input(s, mom_over_mass);
      }
    };
  }
}

#endif