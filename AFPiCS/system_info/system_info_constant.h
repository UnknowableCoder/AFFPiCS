#ifndef AFFPICS_SYSTEM_INFO_CONSTANT
#define AFFPICS_SYSTEM_INFO_CONSTANT

/*!
  \file system_info_constant.h
  
  \brief Defines basic functionality for a constant-sized system
         with constant epsilon and mu.
         (Which should really correspond to everything we do.)
  
  \author Nuno Fernandes
*/

#include "../header.h"
#include "system_info_base.h"

namespace AFFPiCS
{
  namespace SystemDefinitions
  {
    template <indexer num_dims, class derived = void>
    class SystemInfoConstant
    {
      private:
      
      using deriv_t = std::conditional_t<std::is_void_v<derived>, SystemInfoConstant, derived>;
      
      protected:
      
      g24_lib::ndview<indexer, num_dims> view;
      
      vector_type<FLType, num_dims> cellsize;
      
      FLType system_epsilon;
      
      FLType system_mu;
            
      public:
            
      /*!
        \param n_cells The number of cells in each dimension.
        \param cell_size The size of each cell in each dimension.
      */
      CUDA_HOS_DEV SystemInfoConstant(const vector_type<indexer, num_dims> &n_cells,
                                      const vector_type<FLType, num_dims> &cell_size):
        view(n_cells), cellsize(cell_size), 
        system_epsilon(AFFPiCS::DefaultUnits::SI.epsilon_zero()),
        system_mu(AFFPiCS::DefaultUnits::SI.mu_zero())
      {
        static_cast<deriv_t*>(this)->set_units(AFFPiCS::DefaultUnits::SI);
      }
      
      CUDA_HOS_DEV SystemInfoConstant(const vector_type<indexer, num_dims> &n_cells,
                                      const vector_type<FLType, num_dims> &cell_size,
                                      const FLType sys_epsilon, const FLType sys_mu):
        view(n_cells), cellsize(cell_size), 
        system_epsilon(sys_epsilon), system_mu(sys_mu)
      {
        static_cast<deriv_t*>(this)->set_units(AFFPiCS::DefaultUnits::SI);
      }
      
      CUDA_HOS_DEV SystemInfoConstant(const vector_type<indexer, num_dims> &n_cells,
                                      const vector_type<FLType, num_dims> &cell_size,
                                      const UnitSystem &new_units):
        view(n_cells), cellsize(cell_size), 
        system_epsilon(new_units.epsilon_zero()), system_mu(new_units.mu_zero())
      {
        static_cast<deriv_t*>(this)->set_units(new_units);
      }
      
      CUDA_HOS_DEV SystemInfoConstant(const vector_type<indexer, num_dims> &n_cells,
                                      const vector_type<FLType, num_dims> &cell_size,
                                      const FLType sys_epsilon, const FLType sys_mu,
                                      const UnitSystem &new_units):
        view(n_cells), cellsize(cell_size), 
        system_epsilon(sys_epsilon), system_mu(sys_mu)
      {
        static_cast<deriv_t*>(this)->set_units(new_units);
      }
      
      void set_cell_sizes(const vector_type<FLType, num_dims> &new_cell_size)
      {
        cellsize = new_cell_size;
      }
      
      void resize_system(const vector_type<indexer, num_dims> &new_num_cells)
      {
        view = g24_lib::ndview<indexer, num_dims>(new_num_cells);
      }
      
      void set_epsilon(const FLType new_epsilon)
      {
        system_epsilon = new_epsilon;
      }
      
      void set_mu(const FLType new_mu)
      {
        system_mu = new_mu;
      }
      
      CUDA_HOS_DEV vector_type<indexer, num_dims> num_cells() const
      {
        return view.numbers();
      }
      
      CUDA_HOS_DEV FLType mu(const indexer i) const
      {
        return system_mu;
      }
      
      CUDA_HOS_DEV FLType mu(const vector_type<indexer, num_dims> &p) const
      {
        return system_mu;
      }
      
      CUDA_HOS_DEV FLType epsilon(const indexer i) const
      {
        return system_epsilon;
      }
      
      CUDA_HOS_DEV FLType epsilon(const vector_type<indexer, num_dims> &p) const
      {
        return system_epsilon;
      }
            
      CUDA_HOS_DEV indexer num_cells(const indexer i) const
      {
        return view.numbers(i);
      }
      
      CUDA_HOS_DEV vector_type<indexer, num_dims> to_cell(const indexer index) const
      {
        return view.to_point(index);
      }
      
      CUDA_HOS_DEV indexer to_index(const vector_type<indexer, num_dims> &cell) const
      {
        return view.to_elem(cell);
      }
      
      CUDA_HOS_DEV vector_type<FLType, num_dims> cell_sizes() const
      {
        return cellsize;
      }
      
      CUDA_HOS_DEV bool is_border(const indexer i) const
      {
        return view.is_border(i);
      }
      
      CUDA_HOS_DEV bool is_border(const vector_type<indexer, num_dims> &p) const
      {
        return view.is_border(p);
      }
      
      CUDA_HOS_DEV bool is_outside(const indexer i) const
      {
        return view.is_outside(i);
      }
      
      CUDA_HOS_DEV bool is_outside(const vector_type<indexer, num_dims> &p) const
      {
        return view.is_outside(p);
      }
      
    };
  }
}

#endif