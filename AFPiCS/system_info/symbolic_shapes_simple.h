#ifndef AFFPICS_SYSTEM_INFO_SYMBOLIC_SHAPES_SIMPLE
#define AFFPICS_SYSTEM_INFO_SYMBOLIC_SHAPES_SIMPLE

/*!
  \file symbolic_shapes_simple.h
  
  \brief Uses Simbpolic's simple symbolic computations to describe
         the shapes of the particles and the field computations
         in a computationally efficient way, but weighting the fields
         with the particle shape instead of integrating the interpolation.
  
  \author Nuno Fernandes
*/

#include "../header.h"
#include "system_info_base.h"
#include "simbpolic.h"
#include "../utilities/interpolators.h"

namespace AFFPiCS
{
  namespace SystemDefinitions
  {
    /*!
      \tparam ParticleShape Something whose `static constexpr get_shape_1D()` returns
                            the 1D shape of the particle as a Simbpolic function 
                            and whose `static constexpr get_width(dim)` returns 
                            a possibly symbolic value that holds the maximum width
                            in the dimension
                            
                            
    */
    template <indexer num_dims, class ParticleShape, class derived = void>
    //Could be easily extended to have different shapes
    //for different particles, but we won't care.
    class SymbolicShapeSimpler
    {
      private:
            
      using deriv_t = std::conditional_t<std::is_void_v<derived>, SymbolicShapeSimpler, derived>;
          
     
      CUDA_HOS_DEV constexpr static indexer compile_time_ceil(const FLType & val)
      {
        const indexer casted(val);
        if (FLType(casted) == val)
          {
            return casted;
          }
        else
          {
            return casted + 1;
          }
      }
      
     public:
      
      template <class part_type>
      CUDA_HOS_DEV inline FLType particle_fraction(const part_type& part, const vector_type<FLType, num_dims> &pos) const
      {
        const auto prim = ParticleShape::get_shape_1D().template primitive<1>();
        FLType intgral = 1;
        for (indexer i = 0; i < num_dims; ++i)
        {
          FLType factor = FLType(prim(pos[i]+0.5)) - FLType(prim(pos[i]-0.5));
          intgral *= factor;
        }
        return intgral;
      }
      
      template <class part_type>
      CUDA_HOS_DEV constexpr indexer particle_cell_radius(const part_type& part) const
      {
        return compile_time_ceil(FLType(ParticleShape::get_width(0))/2);
      }
      
    private:
      
      template <indexer dim, class part_type, class Func, class MeasureFunc>
      CUDA_HOS_DEV auto gather_helper(const part_type &particle,
                                      const Func& field_getter,
                                      const MeasureFunc& measurement,
                                      const vector_type<indexer, num_dims> &cell  ) const
      {
        using ret_t = decltype(field_getter(cell));
        
        const deriv_t* dhis = static_cast<const deriv_t*>(this);
                
        if constexpr (dim >= num_dims)
          {
            ret_t ret = field_getter(cell);
            
            for (indexer i = 0; i < ret.size(); ++i)
              {
                FLType factor = dhis->particle_fraction( particle, particle.pos(*dhis) -
                                                             measurement(i) +
                                                             vector_type<FLType, num_dims>(particle.cell(*dhis) - cell) );
                ret[i] *= factor;
              }
            return ret;
          }
        else
          {
            
            ret_t ret(FLType(0));
            
            for (indexer i = -dhis->particle_cell_radius(particle)-1; i <= dhis->particle_cell_radius(particle)+1; ++i)
              {
                ret += gather_helper<dim+1>(particle, field_getter, measurement, cell.add(dim, i));
              }
            
            return ret;
          }
          
      }
      
    public:
      template <class ArrT, class part>
      CUDA_HOS_DEV E_field_type<num_dims> E_gather(const ArrT &E_arr, const part& particle) const
      {
        const deriv_t* dhis = static_cast<const deriv_t*>(this);
        
        E_field_type<num_dims> ret = gather_helper<0>( particle,
                                                       [&](const vector_type<indexer, num_dims>& cell)->E_field_type<num_dims>
                                                       { return dhis->boundary_E(cell, E_arr); },
                                                       [&](const indexer dim){ return dhis->E_measurement(dim); },
                                                       particle.cell(*dhis));
                                                       
        global::log << "E " << particle.cell(*dhis) << " | " << particle.pos(*dhis) << " = " << ret << std::endl;
        
        return ret;
        
      }
      
      template <class ArrT, class part>
      CUDA_HOS_DEV B_field_type<num_dims> B_gather(const ArrT &B_arr, const part& particle) const
      {
        const deriv_t* dhis = static_cast<const deriv_t*>(this);
        
        B_field_type<num_dims> ret = gather_helper<0>( particle,
                                                        [&](const vector_type<indexer, num_dims>& cell)->B_field_type<num_dims>
                                                        { return dhis->boundary_B(cell, B_arr); },
                                                       [&](const indexer dim){ return dhis->B_measurement(dim); },
                                                       particle.cell(*dhis));
                                     
        global::log << "B " << particle.cell(*dhis) << " | " << particle.pos(*dhis) << " = " << ret << std::endl;
        
        return ret;
      }
      
    };
  }
}

#endif