#ifndef AFFPICS_PARTICLES_PARTICLE_BASE
#define AFFPICS_PARTICLES_PARTICLE_BASE

/*!
  \file particle_base.h
  
  \brief Holds some useful definitions for specifying particles within the simulation.
  
  \author Nuno Fernandes
*/

#include "../header.h"

namespace AFFPiCS
{
  namespace Particles
  {
    
    template <indexer num_dims, class derived>
    class particle_base
    {
      public:
      /*!
          \brief The (relativistic) momentum over the rest mass
      */
      template <class system_info>
      CUDA_HOS_DEV vector_type<FLType, num_dims> u(const system_info &info) const
      {
        static_assert(AFFPICS_INHERITANCE_HACK_CHECK(), "Should define this somewhere else! Blame C++ for the lack of virtual templates...");
        return vector_type<FLType, num_dims>(0);
      }
      
      
      /*! \brief The relative position of the particle inside the cell
                 (in units of cell separation).
                 
           \remark The center of the cell is given by (0.5, ... , 0.5),
                   so all the entries are in the [0, 1[ interval.
      */
      template <class system_info>
      CUDA_HOS_DEV vector_type<FLType, num_dims> pos(const system_info &info) const
      {
        static_assert(AFFPICS_INHERITANCE_HACK_CHECK(), "Should define this somewhere else! Blame C++ for the lack of virtual templates...");
        return vector_type<FLType, num_dims>(0);
      }
      
      /*!
        \brief The cell in which the particle is.
      */
      template <class system_info>
      CUDA_HOS_DEV vector_type<indexer, num_dims> cell(const system_info &info) const
      {
        static_assert(AFFPICS_INHERITANCE_HACK_CHECK(), "Should define this somewhere else! Blame C++ for the lack of virtual templates...");
        return vector_type<indexer, num_dims>(0);
      }
      
      /*! \brief The absolute position of the particle (in relevant units)
      */
      template <class system_info>
      CUDA_HOS_DEV vector_type<FLType, num_dims> absolute_pos(const system_info &info) const
      {
        const derived* dhis = static_cast<const derived*>(this);
        return (dhis->pos(info)+vector_type<FLType, num_dims>(dhis->cell(info))).element_multiply(info.cell_sizes());
      }
      
      
      template <class system_info>
      CUDA_HOS_DEV FLType mass(const system_info &info) const
      {
        static_assert(AFFPICS_INHERITANCE_HACK_CHECK(), "Should define this somewhere else! Blame C++ for the lack of virtual templates...");
        return 0;
      }
      
      template <class system_info>
      CUDA_HOS_DEV FLType charge(const system_info &info) const
      {
        static_assert(AFFPICS_INHERITANCE_HACK_CHECK(), "Should define this somewhere else! Blame C++ for the lack of virtual templates...");
        return 0;
      }
      
      template <class system_info>
      CUDA_HOS_DEV FLType gamma(const system_info &info) const
      {
        const derived* dhis = static_cast<const derived*>(this);
        using namespace std;
        return sqrt(dhis->u(info).square_norm2()/info.units().c()/info.units().c()+FLType(1));
      }
      
      /*!
        \brief The momentum of the particle, in appropriate units.
      */
      template <class system_info>
      CUDA_HOS_DEV vector_type<FLType, num_dims> p(const system_info &info) const
      {
        const derived* dhis = static_cast<const derived*>(this);
        return dhis->u(info) * dhis->mass(info);
      }
      
      /*!
        \brief Returns the velocity of the particle, with the spatial part in units of cell separation.
      */
      template <class system_info>
      CUDA_HOS_DEV vector_type<FLType, num_dims> vel(const system_info &info) const
      {
        const derived* dhis = static_cast<const derived*>(this);
        return dhis->u(info).element_divide(info.cell_sizes())/dhis->gamma(info);
      }
      
      /*!
        \brief Sets the value of the momentum of the particle over its rest mass.
      */    
      template <class system_info>
      CUDA_HOS_DEV void set_u(const vector_type<FLType, num_dims>& new_u, const system_info &info)
      {
        static_assert(AFFPICS_INHERITANCE_HACK_CHECK(), "Should define this somewhere else! Blame C++ for the lack of virtual templates...");
      }
      
      
      /*!
        \brief Sets the value of the momentum of the particle (in relevant units).
      */    
      template <class system_info>
      CUDA_HOS_DEV void set_p(const vector_type<FLType, num_dims>& new_p, const system_info &info)
      {
        derived* dhis = static_cast<derived*>(this);
        dhis->set_u(new_p/dhis->mass(info), info);
      }
      
      
      /*!
        \brief Sets the value of the velocity of the particle (in units of cell size).
      */
      template <class system_info>
      CUDA_HOS_DEV void set_vel(const vector_type<FLType, num_dims>& new_vel, const system_info &info)
      {
        derived* dhis = static_cast<derived*>(this);        
        const vector_type<FLType, num_dims> real_vel = new_vel.element_multiply(info.cell_sizes());
        const vector_type<FLType, num_dims> target_u = real_vel /
                                          (sqrt(1 - real_vel.square_norm2()/info.units().c()/info.units().c()));
        dhis->set_u(target_u, info);
      }
      
      /*!
        \brief Sets the position within the cell, in cell separations.
        
         \remark The center of the cell is given by (0.5, ... , 0.5),
                 so all the entries are in the [0, 1[ interval.
      */    
      template <class system_info>
      CUDA_HOS_DEV void set_pos(const vector_type<FLType, num_dims>& new_pos, const system_info &info) 
      {
        static_assert(AFFPICS_INHERITANCE_HACK_CHECK(), "Should define this somewhere else! Blame C++ for the lack of virtual templates...");
      }
      
      /*!
        \brief Sets the cell where the particle is.
      */    
      template <class system_info>
      CUDA_HOS_DEV void set_cell(const vector_type<indexer, num_dims>& new_cell, const system_info &info)
      {
        static_assert(AFFPICS_INHERITANCE_HACK_CHECK(), "Should define this somewhere else! Blame C++ for the lack of virtual templates...");
      }
      
      /*!
        \brief Changes the position of the particle by \p how_much, measured in cell separations,
               and updates the cell at which the particle is situated accordingly.
      */
      template <class system_info>
      CUDA_HOS_DEV void move(const vector_type<FLType, num_dims>& how_much, const system_info &info)
      {
        derived* dhis = static_cast<derived*>(this);
        using namespace std;
        const vector_type<indexer, num_dims> new_cell = dhis->cell(info) + vector_type<indexer, num_dims>(floor(how_much));
        const vector_type<FLType, num_dims> remainder = dhis->pos(info) + how_much - floor(how_much);
                
        dhis->set_cell(new_cell + vector_type<indexer, num_dims>(floor(remainder)), info);
        
        dhis->set_pos(remainder - floor(remainder), info );
        
        info.boundary_particles(*dhis);
        //This will check for the particle being outside the array
        //and apply the appropriate changes according to the boundary conditions.
      }
      
    };
  }
}

#endif