#ifndef AFFPICS_SYSTEM_PARTICLE_SHAPES_POLYNOMIAL
#define AFFPICS_SYSTEM_PARTICLE_SHAPES_POLYNOMIAL

/*!
  \file polynomial.h
  
  \brief Implements polynomial particle shapes as provided in Na, Teixeira and Chew (10.1109/JMMCT.2019.2958069),
  
  \author Nuno Fernandes
  
  \remark The third order polynomial is a nice approximation for the second-order spline!
*/

#include "../header.h"
#include "simbpolic.h"

namespace AFFPiCS
{
  namespace ParticleShapes
  {
    template <indexer num_dims, indexer order = 0>
    class Polynomial
    {
     private:
      
      template <indexer i>
      using I = Simbpolic::Intg<i>;
      
      template <indexer a, indexer b>
      using X = Simbpolic::Monomial<a, b>;
      
      template <indexer i>
      using V = Simbpolic::Var<i>;
      
      static constexpr auto width = Simbpolic::Rational<order + 1, 2>{};
      //To be consistent with splines.
      
      static constexpr auto base_func = (I<1>{} - X<2, 1>{}) ^ I<order>{};
      
      static constexpr auto integral = Simbpolic::integrate(base_func, V<1>{}, I<-1>{}, I<1>{});
      
      static constexpr auto coeff = I<1>{}/(integral * width);
      
      static constexpr auto func_dim = ((I<1>{} - X<2, 1>{} / (width * width)) ^ I<order>{}) * coeff;
      
      static constexpr auto func_1D = Simbpolic::branched(V<1>{}, Simbpolic::Zero{}, width*I<-1>{}, func_dim, width, Simbpolic::Zero{});
      
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
        return width * I<2>{};
      }
    };
  }
}

#endif
