#ifndef AFFPICS_SYSTEM_INFO_SYMBOLIC_SHAPES
#define AFFPICS_SYSTEM_INFO_SYMBOLIC_SHAPES

/*!
  \file symbolic_shapes.h
  
  \brief Uses Simbpolic's simple symbolic computations to describe
         the shapes of the particles and the field computations
         in a computationally efficient way.
  
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
      \tparam ParticleShape Something whose `static constexpr get_shape()` returns
                            the 3-d shape of the particle as a Simbpolic function 
                            and whose `static constexpr get_width(dim)` returns 
                            a possibly symbolic value that holds the maximum width
                            in the dimension
                            
      
      \tparam extra_interpolation Number of extra interpolation points to the sides of the particle.
                                  (The fields will be interpolated from the cell where the particle is
                                   to +/- ParticleShape::get_widths()/2 + extra_interpolation)
                            
    */
    template <indexer num_dims, class ParticleShape, indexer extra_interpolation = 0, class derived = void>
    //Could be easily extended to have different shapes
    //for different particles, but we won't care.
    class SymbolicShape
    {
      private:
            
      using deriv_t = std::conditional_t<std::is_void_v<derived>, SymbolicShape, derived>;
          
     
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
      
      //-----------------------------------------------------------------------------------------------------//
      //                                       SHAPESHIFT                                                    //
      //-----------------------------------------------------------------------------------------------------//

      CUDA_HOS_DEV inline static constexpr indexer dim_to_id_shsh(const indexer dim)
      {
        return - dim - 1;
      }
      
      CUDA_HOS_DEV inline static constexpr indexer id_to_dim_shsh(const indexer id)
      {
        return -id - 1;
      }
      
      CUDA_HOS_DEV inline static constexpr bool id_is_dim_shsh(const indexer id)
      {
        return (id < 0);
      }
      
      template <indexer dim, class Func>
      CUDA_HOS_DEV inline auto shape_shift(const Func& f) const
      {
        if constexpr (dim >= num_dims)
          {
            return f;
          }
        else
          {
            const auto off_1 = Simbpolic::offset(Simbpolic::Var<dim+1>{}, Simbpolic::Stored<dim_to_id_shsh(dim)>{}, f);
            return shape_shift<dim+1>(off_1);
          }
      }
      
      //-----------------------------------------------------------------------------------------------------//
      //                                       INTERPOLATE HELPERS                                           //
      //-----------------------------------------------------------------------------------------------------//
      
      CUDA_HOS_DEV inline static constexpr indexer half_points(const indexer dim)
      {
        return compile_time_ceil(FLType(ParticleShape::get_width(dim))/2) + extra_interpolation;
      }
      
      CUDA_HOS_DEV inline static constexpr indexer num_points(const indexer dim)
      {
        return 2 * half_points(dim) + 1;
      }
      
      CUDA_HOS_DEV inline static constexpr indexer total_num_points()
      {
        indexer ret = 1;
        for (indexer i = 0; i < num_dims; ++i)
          {
            ret *= num_points(i);
          }
        return ret;
      }
      
      template <indexer dim, indexer ... Is>
      CUDA_HOS_DEV inline static constexpr auto point_sequence_helper(const std::integer_sequence<indexer, Is...> &seq)
      {
        return std::integer_sequence<indexer, (Is - half_points(dim))...>{};
      }
      
      template <indexer dim>
      CUDA_HOS_DEV inline static constexpr auto point_sequence()
      {
        return point_sequence_helper<dim>(std::make_integer_sequence<indexer, num_points(dim)>{});
      }
      
      CUDA_HOS_DEV inline static constexpr auto inter_indexes()
      {
        return std::make_integer_sequence<indexer, total_num_points()>();
      }
      
      CUDA_HOS_DEV inline static constexpr indexer points_stride(const indexer dim)
      {
        if (dim >= num_dims)
          {
            return 0;
          }
        else if (dim == num_dims - 1)
          {
            return 1;
          }
        else
          {
            return points_stride(dim + 1) * num_points(dim + 1);
          }
      }
       
             
      CUDA_HOS_DEV inline static constexpr vector_type<indexer, num_dims> inter_index_to_delta_cell(const indexer idx)
      {
        vector_type<indexer, num_dims> ret;
        indexer temp = idx;
        for (indexer i = 0; i < num_dims; ++i)
          {
            ret[i] = temp / points_stride(i) - half_points(i);
            temp = temp % points_stride(i);
          }
        return ret;
      }
             
      CUDA_HOS_DEV inline static constexpr indexer inter_index_to_delta_cell(const indexer idx, const indexer dim)
      {
        indexer temp = idx;
        for (indexer i = 0; i < dim; ++i)
          {
            temp = temp % points_stride(i);
          }
        return temp/points_stride(dim) - half_points(dim);
      }
      
      CUDA_HOS_DEV constexpr inline static indexer simple_factorial(const indexer num)
      {
        indexer ret = 1;
        for (indexer i = 2; i < num; ++i)
          {
            ret *= i;
          }
        return ret;
      }
      
      template <indexer dim, indexer skip, indexer num>
      CUDA_HOS_DEV constexpr inline static auto lagrange_term()
      {
        if constexpr (num == skip)
          {
            return Simbpolic::One{};
          }
        else
          {
            return (Simbpolic::Monomial<1, dim+1>{} - Simbpolic::Intg<num>{} + Simbpolic::Stored<dim_to_id_shsh(dim)>{});
          }
      }
      
      template <indexer dim, indexer skip, indexer ... Is>
      CUDA_HOS_DEV constexpr inline static auto lagrange_helper(const std::integer_sequence<indexer, Is...> &seq)
      {
        return (lagrange_term<dim, skip, Is>() * ...);
      }
      
      template <indexer dim, indexer skip>
      CUDA_HOS_DEV constexpr inline static auto lagrange_factor()
      {
        constexpr indexer pre_factor = simple_factorial(skip+half_points(dim))*simple_factorial(half_points(dim) - skip);
        constexpr indexer sign = ((half_points(dim) - skip) % indexer(2) == indexer(0) ? indexer(1) : indexer(-1));
        return  Simbpolic::Rational<sign, pre_factor>{};
      }
      
      template <indexer dim, indexer skip>
      CUDA_HOS_DEV constexpr inline static auto lagrange_1D()
      {
        
        return lagrange_helper<dim, skip>(point_sequence<dim>());
      }
      
      template <indexer ... dims, indexer ... skips>
      CUDA_HOS_DEV constexpr inline static auto lagrange_full(const std::integer_sequence<indexer, dims...> & dim,
                                                              const std::integer_sequence<indexer, skips...> & skip)
      {
        return  (lagrange_factor<dims, skips>() * ...) * (lagrange_1D<dims, skips>() * ...);
      }
      
      template <indexer idx, indexer ... is>
      CUDA_HOS_DEV constexpr static inline auto interpolate_one(const std::integer_sequence<indexer, is...> &seq)
      {
        return Simbpolic::Stored<idx>{} * lagrange_full( std::make_integer_sequence<indexer, num_dims>{},
                                                       std::integer_sequence<indexer, inter_index_to_delta_cell(idx, is)...>{} );
      }
      
      template <indexer ... is>
      CUDA_HOS_DEV constexpr static inline auto interpolate_helper(const std::integer_sequence<indexer, is...> &seq)
      {
        return (interpolate_one<is>(std::make_integer_sequence<indexer, num_dims>{}) + ... );
      }
      
      CUDA_HOS_DEV constexpr static inline auto interpolate_helper()
      {
        return interpolate_helper(inter_indexes());
      }
      
      //-----------------------------------------------------------------------------------------------------//
      //                                      INTEGRAL HELPER                                                //
      //-----------------------------------------------------------------------------------------------------//

      template <indexer dim, class Func, class LimitHelper>
      CUDA_HOS_DEV auto integral_helper(const Func& f, const LimitHelper& helper) const
      {
        if constexpr (dim >= num_dims)
          {
            return f;
          }
        else
          {
            const auto integral = Simbpolic::integrate(f, Simbpolic::Var<dim+1>{},
                                                          helper.template begin<dim>(),
                                                          helper.template end<dim>()    );
                        
            return integral_helper<dim + 1>(integral, helper);
          }
      }

      //-----------------------------------------------------------------------------------------------------//
      //                                        GATHER HELPER                                                //
      //-----------------------------------------------------------------------------------------------------//
      
      struct E_helper
      {
        template <indexer dim, class ArrT>
        CUDA_HOS_DEV static inline auto eval(const deriv_t* dhis, const vector_type<indexer, num_dims> &cell, const ArrT& array)
        {
          return dhis->boundary_E(cell, array)[dim];
        }
        
        CUDA_HOS_DEV static inline auto pos(const deriv_t* dhis, const indexer dim)
        {
          return dhis->E_measurement(dim);
        }
        
      };
      
      struct B_helper
      {
        template <indexer dim, class ArrT>
        CUDA_HOS_DEV static inline auto eval(const deriv_t* dhis, const vector_type<indexer, num_dims> &cell, const ArrT& array)
        {
          return dhis->boundary_B(cell, array)[dim];
        }
        
        CUDA_HOS_DEV static inline auto pos(const deriv_t* dhis, const indexer dim)
        {
          return dhis->B_measurement(dim);
        }
      };
      
      template <class HelpFunctor, class ArrT, indexer dim>
      struct gather_store : public Simbpolic::Store
      {
        const vector_type<FLType, num_dims> &new_center;
        
        const vector_type<indexer, num_dims> &cell;
        
        const deriv_t* dhis;
        
        const ArrT& arr;
        
        CUDA_HOS_DEV gather_store(const vector_type<FLType, num_dims> & n_c,
                                  const vector_type<indexer, num_dims> & c,
                                  const deriv_t *p,
                                  const ArrT& array):
        new_center(n_c), cell(c), dhis(p), arr(array)
        {
        }
        
        template <indexer id> SIMBPOLIC_CUDA_HOS_DEV inline constexpr FLType get() const
        {
          if constexpr (id_is_dim_shsh(id))
            {
              return new_center[id_to_dim_shsh(id)];
            }
          else
            {
              const vector_type<indexer, num_dims> delta_cell = inter_index_to_delta_cell(id);
              return HelpFunctor::template eval<dim, ArrT>(dhis, cell + delta_cell, arr);
            }
        }
      };
      
      struct gather_integrate_helper
      {        
        template <indexer dim>
        CUDA_HOS_DEV inline constexpr auto begin() const
        {
          return -ParticleShape::get_width(dim) / Simbpolic::Intg<2>{};
        }
        
        template <indexer dim>
        CUDA_HOS_DEV inline  constexpr auto end() const
        {
          return ParticleShape::get_width(dim) / Simbpolic::Intg<2>{};
        }
      };
      
      template <class HelpFunctor, indexer dim = 0, class retT, class ArrT>
      CUDA_HOS_DEV inline void gather_helper(retT& ret,
                                             const ArrT & arr,
                                             const vector_type<indexer, num_dims> &cell,
                                             const vector_type<FLType, num_dims> &position) const
      {
        if constexpr (dim < ret.size())
          {
            const auto interpol = interpolate_helper();
                        
            const auto particle_shape = ParticleShape::get_shape();
            
            const auto func_to_eval = interpol * particle_shape;
            
            const auto integrated = integral_helper<0>(func_to_eval, gather_integrate_helper{});
            
            const deriv_t* dhis = static_cast<const deriv_t*>(this);
            
            const vector_type<FLType, num_dims> new_center = HelpFunctor::pos(dhis, dim) - position;
            
            gather_store<HelpFunctor, ArrT, dim> store(new_center, cell, dhis, arr);
            
            ret[dim] = FLType(integrated(store));
            
            gather_helper<HelpFunctor, dim+1>(ret, arr, cell, position);
          }
      }
      
      //-----------------------------------------------------------------------------------------------------//
      //                                             REST                                                    //
      //-----------------------------------------------------------------------------------------------------//

      struct particle_fraction_helper
      {
        const vector_type<FLType, num_dims> &pos;
        
        template <indexer dim>
        CUDA_HOS_DEV inline FLType begin() const
        {
          return pos[dim] - FLType(0.5);
        }
        
        template <indexer dim>
        CUDA_HOS_DEV inline FLType end() const
        {
          return po[dim] + FLType(0.5);
        }
      };
      
     public:
      
      template <class part_type>
      CUDA_HOS_DEV FLType particle_fraction(const part_type& part, const vector_type<FLType, num_dims> &pos) const
      {
        particle_fraction_helper helper{pos};
        
        const auto intgral = integral_helper<0>(ParticleShape::get_shape(), helper);
        
        return FLType(intgral);
      }
      
    private:
    
      template <indexer dim, indexer current>
      CUDA_HOS_DEV static constexpr indexer maximum_width_getter()
      {
        if constexpr (dim >= num_dims)
          {
            return current;
          }
        else
          {
            constexpr indexer other = compile_time_ceil(FLType(ParticleShape::get_width(dim))/2);
            if constexpr (other > current)
              {
                return maximum_width_getter<dim+1, other>();
              }
            else
              {
                return maximum_width_getter<dim+1, current>();
              }
          }
      }
      
    public:
      template <class part_type>
      CUDA_HOS_DEV constexpr indexer particle_cell_radius(const part_type& part) const
      {
        return maximum_width_getter<1, compile_time_ceil(FLType(ParticleShape::get_widths(0))/2)>();
      }
      
      template <class ArrT, class part>
      CUDA_HOS_DEV E_field_type<num_dims> E_gather(const ArrT &E_arr, const part& particle) const
      {
        const deriv_t* dhis = static_cast<const deriv_t*>(this);
        E_field_type<num_dims> ret;
        gather_helper<E_helper, 0>(ret, E_arr, particle.cell(*dhis), particle.pos(*dhis));
        return ret;
        
      }
      
      template <class ArrT, class part>
      CUDA_HOS_DEV B_field_type<num_dims> B_gather(const ArrT &B_arr,  const part& particle) const
      {
        const deriv_t* dhis = static_cast<const deriv_t*>(this);
        B_field_type<num_dims> ret;
        gather_helper<B_helper, 0>(ret, B_arr, particle.cell(*dhis), particle.pos(*dhis));
        return ret;
        
      }
      
      
    };
  }
}

#endif