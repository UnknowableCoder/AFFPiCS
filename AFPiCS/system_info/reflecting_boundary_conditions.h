#ifndef AFFPICS_SYSTEM_INFO_REFLECTING_BOUNDARY
#define AFFPICS_SYSTEM_INFO_REFLECTING_BOUNDARY

/*!
  \file reflecting_boundary_conditions.h
  
  \brief A system with reflecting boundary conditions.
  
  \author Nuno Fernandes
*/

#include "../header.h"
#include "system_info_base.h"

namespace AFFPiCS
{
  namespace SystemDefinitions
  {
    template <indexer num_dims, class derived = void>
    class ReflectingBoundaryConditions
    {
      private:
      
      using deriv_t = std::conditional_t<std::is_void_v<derived>, ReflectingBoundaryConditions, derived>;
            
      template <indexer dim, bool check_for_border, class Func, class ... Args>
      CUDA_HOS_DEV void for_all_neighbours_impl( const indexer radius,
                                                 const vector_type<indexer, num_dims> &cell,
                                                 const vector_type<bool, num_dims> &to_mirror,
                                                 Func && func, Args&& ... args                             ) const
      {
        const deriv_t* dhis = static_cast<const deriv_t*>(this);
        if constexpr (dim == num_dims)
          {
            func(dhis->to_index(cell), cell, to_mirror, std::forward<Args>(args)...);
          }
        else if constexpr (dim < num_dims)
          {
            if constexpr (check_for_border)
              {                
                indexer j;
                for (j = -radius; j <= radius && cell[dim] + j < 0; ++j)
                  {
                    for_all_neighbours_impl<dim + 1, check_for_border, Func, Args...>
                      (radius, cell.set(dim, -cell[dim] - j - 1), to_mirror.set(dim, true),
                        std::forward<Func>(func), std::forward<Args>(args)...);
                  }
                for (; j <= radius && cell[dim] + j < dhis->num_cells(dim); ++j)
                  {
                    for_all_neighbours_impl<dim + 1, check_for_border, Func, Args...>
                      (radius, cell.add(dim, j), to_mirror.set(dim, false), std::forward<Func>(func), std::forward<Args>(args)...);
                  }
                for (; j <= radius; ++j)
                  {
                    for_all_neighbours_impl<dim + 1, check_for_border, Func, Args...>
                      (radius, cell.set(dim, 2*dhis->num_cells(dim) - (j + cell[dim]) - 1), to_mirror.set(dim, true),
                        std::forward<Func>(func), std::forward<Args>(args)...);
                  }
              }
            else
              {
                for (indexer i = cell[dim] - radius; i <= cell[dim] + radius; ++i)
                  {
                    for_all_neighbours_impl<dim + 1, check_for_border, Func, Args...>
                      (radius, cell.set(dim, i), to_mirror.set(dim, false), std::forward<Func>(func), std::forward<Args>(args)...);
                  }
              }
          }
      }
      
      public:
      
      template <bool check_for_border = true, class Func, class ... Args>
      CUDA_HOS_DEV void for_all_neighbours( const indexer radius,
                                            const vector_type<indexer, num_dims> &cell,
                                            Func && func, Args&& ... args                             ) const
      {
        const derived* dhis = static_cast<const derived*>(this);
        for_all_neighbours_impl<0, check_for_border, Func, Args...>
                          (radius, cell, vector_type<bool, num_dims>(false),
                            std::forward<Func>(func), std::forward<Args>(args)...           );
      }
      
      template <class particle>
      CUDA_HOS_DEV void boundary_particles (particle& part, const bool force_apply = false) const
      {
        const deriv_t* dhis = static_cast<const deriv_t*>(this);
        if (force_apply || dhis->is_outside(part.cell(*dhis)))
          {
            for (indexer dim = 0; dim < num_dims; ++dim)
              {
                if (part.cell(*dhis)[dim] < 0)
                  {
                    part.set_u(part.u(*dhis).multiply(dim, -1), *dhis);
                    //v -> -v => u -> -u
                    part.set_pos(part.pos(*dhis).multiply(dim, -1).add(dim, 1), *dhis);
                    //x -> 1 - x
                    part.set_cell(part.cell(*dhis).multiply(dim, -1).subtract(dim, 1), *dhis);
                    //cell -> -cell - 1
                    //(-1 -> 0, -2 -> 1 and so on)
                  }
                else if (part.cell(*dhis)[dim] >= dhis->num_cells(dim))
                  {
                    part.set_u(part.u(*dhis).multiply(dim, -1), *dhis);
                    //v -> -v => u -> -u
                    part.set_pos(part.pos(*dhis).multiply(dim, -1).add(dim, 1), *dhis);
                    //x -> 1 - x
                    part.set_cell(part.cell(*dhis).multiply(dim, -1).add(dim, 2 * dhis->num_cells(dim) - 1), *dhis);
                    //cell -> num_cells - cell + 1
                    //(num_cells -> num_cells - 1, num_cells + 1 -> num_cells - 2 and so on)
                  }
              }
          }
      }
      
    private:
      template <class ArrT>
      CUDA_HOS_DEV auto boundary_general( const vector_type<indexer, num_dims> &cell,
                                          const ArrT& arr                               ) const
      {
        const deriv_t* dhis = static_cast<const deriv_t*>(this);
        vector_type<indexer, num_dims> new_cell;
        for (indexer dim = 0; dim < num_dims; ++dim)
          {
            if(cell[dim] < 0)
              {
                new_cell[dim] = -cell[dim] - 1;
              }
            else if(cell[dim] >= dhis->num_cells(dim))
              {
                new_cell[dim] = 2 * dhis->num_cells(dim) - cell[dim] - 1;
              }
            else
              {
                new_cell[dim] = cell[dim];
              }
          }
        return arr[dhis->to_index(new_cell)];
      }
      
    public:
      template <class ArrT>
      CUDA_HOS_DEV E_field_type<num_dims> boundary_E(const vector_type<indexer, num_dims> &cell,
                                                     const ArrT & E_fields                                      ) const
      {
        const deriv_t* dhis = static_cast<const deriv_t*>(this);
        E_field_type<num_dims> ret = boundary_general(cell, E_fields);
        for (indexer dim = 0; dim < num_dims; ++dim)
          {
            if (cell[dim] < 0 || cell[dim] >= dhis->num_cells(dim))
              {
                ret[dim] = -ret[dim];
              }
          }
        return ret;
      }
      
      template <class ArrT>
      CUDA_HOS_DEV B_field_type<num_dims> boundary_B(const vector_type<indexer, num_dims> &cell,
                                                     const ArrT & B_fields                                      ) const
      {
        const deriv_t* dhis = static_cast<const deriv_t*>(this);
        B_field_type<num_dims> ret = boundary_general(cell, B_fields);
        if constexpr (num_dims == 3)
        //In the remaining cases, the reflecting boundary conditions do not change the magnetic field.
          {
            for (indexer dim = 0; dim < num_dims; ++dim)
              {
                if (cell[dim] < 0 || cell[dim] >= dhis->num_cells(dim))
                  {
                    ret[dim] = -ret[dim];
                  }
              }
          }
        return ret;
      }
                                                             
      
      template <class ArrT>
      CUDA_HOS_DEV current_type<num_dims> boundary_J(const vector_type<indexer, num_dims> &cell,
                                                     const ArrT & currents                                      ) const
      {
        return boundary_general(cell, currents);
      }
    };
  }
}

#endif