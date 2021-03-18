#ifndef AFFPICS_SYSTEM_PARTICLE_SHAPES_SPLINES
#define AFFPICS_SYSTEM_PARTICLE_SHAPES_SPLINES

/*!
  \file splines.h
  
  \brief Implements the usual particle shapes, that correspond to (scaled) basis functions of b-splines.
  
  \author Nuno Fernandes
  
  \remark The third order polynomial shape is a nice approximation for the second-order spline!
*/

#include "../header.h"
#include "simbpolic.h"


namespace AFFPiCS
{
  namespace ParticleShapes
  {
    template <indexer num_dims, indexer order = 0>
    class Spline
    {
     private:
      
      template <indexer i>
      using I = Simbpolic::Intg<i>;
            
      template <indexer i>
      using V = Simbpolic::Var<i>;
      
      template <indexer a, indexer b>
      using R = Simbpolic::Rational<a, b>;
            
      template <indexer ord>
      CUDA_HOS_DEV static constexpr auto create_spline()
      {
        if constexpr (ord == 0)
          {
            return Simbpolic::branched(V<1>{}, Simbpolic::Zero{}, R<-1,2>{}, Simbpolic::One{}, R<1,2>{}, Simbpolic::Zero{});
          }
        else
          {
            const auto integrand = create_spline<ord-1>();
            const auto prim = integrand.template primitive<1>();
            const auto off_forward = Simbpolic::offset(V<1>{}, R<1,2>{}, prim);
            const auto off_backward = Simbpolic::offset(V<1>{}, R<-1,2>{}, prim);
            const auto diff = off_forward - off_backward;
            return diff;//Simbpolic::distribute<5>(diff);
          }
      }

      static constexpr auto func_1D = create_spline<order>();
      
      template <indexer dim, class F>
      CUDA_HOS_DEV static constexpr auto shape_helper(const F &f)
      {
        if constexpr (dim > num_dims)
          {
            return f;
          }
        else
          {
            return shape_helper<dim+1>(f * Simbpolic::change_dim(V<1>{}, V<dim>{}, func_1D));
          }
      }
      
      
      static constexpr auto func_3D = shape_helper<1>(Simbpolic::One{});
      
     public:
      
      template <indexer dim = 0>
      CUDA_HOS_DEV static constexpr auto get_shape_1D()
      {
        return Simbpolic::change_dim(V<1>{}, V<dim + 1>{}, func_1D);
      }
      
      CUDA_HOS_DEV static constexpr auto get_shape()
      {        
        return func_3D;
      }
      
      CUDA_HOS_DEV static constexpr auto get_width(const indexer dim)
      {
        return I<order + 1>{};
      }
      
    };
    
  }
}

#endif
