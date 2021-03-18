#ifndef AFFPICS_INTERPOLATORS
#define AFFPICS_INTERPOLATORS

/*!
  \file interpolators.h
  
  \brief Uses Simbpolic's simple symbolic computations to describe
         arbitratily sized interpolations.
  
  \author Nuno Fernandes
*/


#include "../header.h"
#include "simbpolic.h"

namespace AFFPiCS
{
  class Interpolator
  {
    private:
  
    /*!
      \brief Gives the index of the variable that corresponds to the coordinate of the i-th interpolation point.
    */
    CUDA_HOS_DEV static constexpr indexer x_i(const indexer i)
    {
      return 2*i + 2;
    }

    /*!
      \brief Gives the index of the variable that corresponds to the value at the i-th interpolation point.
    */
    CUDA_HOS_DEV static constexpr indexer y_i(const indexer i)
    {
      return 2*i + 3;
    }
    
    template <indexer i>
    using I = Simbpolic::Intg<i>;
    
    template <indexer a, indexer b>
    using X = Simbpolic::Monomial<a, b>;
    
    template <indexer i>
    using V = Simbpolic::Var<i>;
    
    /*!
      \brief Returns the polynomial interpolation for several orders.
      
      Variables shall be (x, x_0, y_0, x_1, y_1, ...)
      
      Uses Neville's Algorithm.
    */
    template <indexer order, indexer idx>
    CUDA_HOS_DEV static constexpr auto neville_interpolation()
    {
      static_assert(order >= 0, "Must have valid order!");
      static_assert(idx >= 0, "Must have valid index!");
      if constexpr (order == 0)
        {
          return X<1, y_i(idx)>{};
        }
      else
        {
          constexpr auto left = (X<1, x_i(idx+order)>{}-X<1, 1>{}) * interpol_storage<order-1,idx>;
          
          constexpr auto right = (X<1, 1>{}-X<1, x_i(idx)>{}) * interpol_storage<order-1,idx+1>;
          
          return (left + right)/(X<1, x_i(idx+order)>{} - X<1, x_i(idx)>{});
        }
    }
    
    template <indexer order, indexer idx>
    static constexpr auto interpol_storage = neville_interpolation<order, idx>();
    //This is mostly to aid the compiler in memoization
    //of the intermediate results, to cut on compilation times.
    //(GCC is known to do it, but not all do so.)
    
    
    public:
    
    /*!
      \brief Returns the polynomial interpolation for several orders.
             Variables shall be (x, x_0, y_0, x_1, y_1, ...)
    */
    template <indexer order>
    inline static constexpr auto symbolic = interpol_storage<order, 0>;
    
    
    
      
    /*!
      \brief Performs an interpolation through all the x, y points provided as arguments.
      
      Variables should be x_0, y_0, ...
    */
    template <indexer dim, class ... Points>
    CUDA_HOS_DEV static constexpr auto interpolate(const V<dim> &var, const Points& ... p)
    {
      const auto ret = symbolic<sizeof...(Points)/2 - 1>(X<1,dim>{}, p...);
      return ret;
    }
    
    private:
    
    template <indexer desired_num_args, indexer dim, class Current, class Increment, class Point, class ... Points>
    CUDA_HOS_DEV static constexpr auto interpolate_y_helper(const V<dim> &var, 
                                                            const Current &x_i, const Increment & dx,
                                                            const Point & one, const Points& ... others )
    {
      if constexpr (sizeof...(Points) == 2 * desired_num_args - 2)
        {
          return interpolate(var, others..., x_i, one);
        }
      else
        {
          return interpolate_y_helper<desired_num_args>(var, x_i + dx, dx, others..., x_i, one);
        }
    }
    
    public:
    
    template <indexer dim, class Initial, class Increment, class Point>
    CUDA_HOS_DEV static constexpr auto interpolate_y(const V<dim> &var, const Initial& x_0, const Increment&  dx, const Point& p)
    {
      return interpolate(var, x_0, p);
    }

    template <indexer dim, class Initial, class Increment, class ... Points>
    CUDA_HOS_DEV static constexpr auto interpolate_y(const V<dim> &var, const Initial& x_0, const Increment&  dx, const Points& ... p)
    {
      return interpolate_y_helper<sizeof...(Points)>(var, x_0, dx, p...);
    }

    private:

    template <indexer desired_num_args, indexer dim, class Func, class Point, class ... Points>
    CUDA_HOS_DEV static constexpr auto interpolate_x_helper(const V<dim> &var, const Func &f, const Point & one, const Points& ... others )
    {
      if constexpr (sizeof...(Points) == 2 * desired_num_args - 2)
        {
          return interpolate(var, others..., one, f(one));
        }
      else
        {
          return interpolate_x_helper<desired_num_args>(var, f, others..., one, f(one));
        }
    }

    public:

    template <indexer dim, class Func, class Point>
    CUDA_HOS_DEV static constexpr auto interpolate_x(const V<dim> &var, const Func& f, const Point& p)
    {
      return interpolate(var, p, f(p));
    }

    template <indexer dim, class Func, class ... Points>
    CUDA_HOS_DEV static constexpr auto interpolate_x(const V<dim> &var, const Func& f, const Points& ... p)
    {
      return interpolate_x_helper<sizeof...(Points)>(var, f, p...);
    }
    
  };
}

#endif
