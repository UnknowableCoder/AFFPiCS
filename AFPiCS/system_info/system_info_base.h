#ifndef AFFPICS_SYSTEM_INFO_BASE
#define AFFPICS_SYSTEM_INFO_BASE

/*!
  \file system_info_base.h
  
  \brief Holds some useful definitions for specifying the properties of the system that will be simulated.
  
  \author Nuno Fernandes
*/

#include "../header.h"
#include "../utilities/unit_system.h"

namespace AFFPiCS
{
  namespace SystemDefinitions
  {
    template <indexer num_dims, class derived>
    class BaseSystemInfo
    {
      protected:
      
      UnitSystem unit_system;
    
      public:
      
      
      /*!
        \brief Gives the magnetic permeability at the cell with index \p i.
      */
      CUDA_HOS_DEV FLType mu(const indexer i) const
      {
        static_assert(AFFPICS_INHERITANCE_HACK_CHECK(), "Should define this somewhere else! Blame C++ for the lack of virtual templates...");
        return 0;
      }
      
      /*!
        \brief Gives the magnetic permeability at the cell specified by the tuple \p p.
      */
      CUDA_HOS_DEV FLType mu(const vector_type<indexer, num_dims> &p) const
      {
        static_assert(AFFPICS_INHERITANCE_HACK_CHECK(), "Should define this somewhere else! Blame C++ for the lack of virtual templates...");
        return 0;
      }
      
      /*!
        \brief Gives the electric permitivity at the cell with index \p i.
      */
      CUDA_HOS_DEV FLType epsilon(const indexer i) const
      {
        static_assert(AFFPICS_INHERITANCE_HACK_CHECK(), "Should define this somewhere else! Blame C++ for the lack of virtual templates...");
        return 0;
      }
      
      /*!
        \brief Gives the electric permitivity at the cell specified by the tuple \p p.
      */
      CUDA_HOS_DEV FLType epsilon(const vector_type<indexer, num_dims> &p) const
      {
        static_assert(AFFPICS_INHERITANCE_HACK_CHECK(), "Should define this somewhere else! Blame C++ for the lack of virtual templates...");
        return 0;
      }
      
      /*!
        \brief Returns the unit system in use.
      */
      CUDA_HOS_DEV const UnitSystem& units() const
      {
        return unit_system;
      }
      
      /*!
        \brief Sets a new unit system.
      */
      CUDA_HOS_DEV void set_units(const UnitSystem& new_units)
      {
        unit_system = new_units;
      }
      
      
      /*!
        \brief Returns the number of dimensions of the system.
      */
      CUDA_HOS_DEV constexpr indexer dimensions() const
      {
        return num_dims;
      }
      
      /*!
        \brief Returns the number of cells in each dimension of the system.
      */
      CUDA_HOS_DEV vector_type<indexer, num_dims> num_cells() const
      {
        static_assert(AFFPICS_INHERITANCE_HACK_CHECK(), "Should define this somewhere else! Blame C++ for the lack of virtual templates...");
        return vector_type<indexer, num_dims>(0);
      }
      
      /*!
        \brief Returns the number of cells in the \p i th dimension of the system.
      */
      CUDA_HOS_DEV indexer num_cells(const indexer i) const
      {
        const derived* dhis = static_cast<const derived*>(this);
        return dhis->num_cells()[i];
      }
      
      /*!
        \brief Returns the total number of cells in the system.
      */
      CUDA_HOS_DEV indexer total_cells() const
      {
        const derived* dhis = static_cast<const derived*>(this);
        return dhis->num_cells().multiply_all();
      }
      
      
      /*!
        \brief Tranforms \p index into a `num_dims`-dimensional tuple.
      */
      CUDA_HOS_DEV vector_type<indexer, num_dims> to_cell(const indexer index) const
      {
        static_assert(AFFPICS_INHERITANCE_HACK_CHECK(), "Should define this somewhere else! Blame C++ for the lack of virtual templates...");
        return vector_type<indexer, num_dims>(0);
      }
      
      /*!
        \brief Transforms a `num_dims`-dimensional tuple into an unique (linear) index.
      */
      CUDA_HOS_DEV indexer to_index(const vector_type<indexer, num_dims> &cell) const
      {
        static_assert(AFFPICS_INHERITANCE_HACK_CHECK(), "Should define this somewhere else! Blame C++ for the lack of virtual templates...");
        return 0;
      }
      
    private:
    
      struct E_boundary_functor
      {
        const derived * dhis;
        template <class ArrT>
        CUDA_HOS_DEV inline auto operator() (const vector_type<indexer, num_dims>& cell, const ArrT& arr) const
        {
          return dhis->boundary_E(cell, arr);
        }
      };
    
      struct B_boundary_functor
      {
        const derived * dhis;
        template <class ArrT>
        CUDA_HOS_DEV inline auto operator() (const vector_type<indexer, num_dims>& cell, const ArrT& arr) const
        {
          return dhis->boundary_B(cell, arr);
        }
      };
      
    public:
      /*!
        \brief Computes the curl of the electric fields at the cell with index \p i.
      */
      template <class E_holder> 
      CUDA_HOS_DEV B_field_type<num_dims> E_curl(const E_holder &E_fields, const indexer i) const
      {
        const derived* dhis = static_cast<const derived*>(this);
        return dhis->E_curl(E_fields, dhis->to_cell(i));
      }
      
      
      /*!
        \brief Computes the curl of the electric fields at the cell specified by the tuple \p p.
      */
      template <class E_holder> 
      CUDA_HOS_DEV B_field_type<num_dims> E_curl(const E_holder &E_fields,
                                            const vector_type<indexer, num_dims> &p) const
      {
        const derived* dhis = static_cast<const derived*>(this);
        return curl<magnetic_field_dimensions<num_dims>()>(p, E_fields, E_boundary_functor{dhis}, dhis->cell_sizes());
      }
                                            
      /*!
        \brief Computes the curl of the magnetic fields at the cell with index \p i.
      */
      template <class B_holder> 
      CUDA_HOS_DEV E_field_type<num_dims> B_curl(const B_holder &B_fields, const indexer i) const
      {
        const derived* dhis = static_cast<const derived*>(this);
        return dhis->B_curl(B_fields, dhis->to_cell(i));
      }
      
      
      /*!
        \brief Computes the curl of the magnetic fields at the cell specified by the tuple \p p.
      */
      template <class B_holder> 
      CUDA_HOS_DEV E_field_type<num_dims> B_curl(const B_holder &B_fields,
                                                 const vector_type<indexer, num_dims> &p) const
      {
        const derived* dhis = static_cast<const derived*>(this);
        return curl<electric_field_dimensions<num_dims>()>(p, B_fields, B_boundary_functor{dhis}, dhis->cell_sizes());
      }
      
      /*!
        \brief Gathers the electric fields given by \p E_arr as felt by the particle \p particle, given its position.
      */
      template <class ArrT, class part>
      CUDA_HOS_DEV E_field_type<num_dims> E_gather( const ArrT &E_arr, 
                                                    const part& particle) const
      {
        static_assert(AFFPICS_INHERITANCE_HACK_CHECK(), "Should define this somewhere else! Blame C++ for the lack of virtual templates...");
        return E_field_type<num_dims>(0);
      }
      
      /*!
        \brief Gathers the magnetic fields given by \p B_arr as felt by the particle \p particle, given its position.
      */
      template <class ArrT, class part>
      CUDA_HOS_DEV B_field_type<num_dims> B_gather( const ArrT &B_arr, 
                                                    const part& particle) const
      {
        static_assert(AFFPICS_INHERITANCE_HACK_CHECK(), "Should define this somewhere else! Blame C++ for the lack of virtual templates...");
        return B_field_type<num_dims>(0);
      }
      
      /*!
        \brief Returns the dimension of the cell in relevant units.
      */
      CUDA_HOS_DEV vector_type<FLType, num_dims> cell_sizes() const
      {
        static_assert(AFFPICS_INHERITANCE_HACK_CHECK(), "Should define this somewhere else! Blame C++ for the lack of virtual templates...");
        return vector_type<FLType, num_dims>(0);
      }
      
      /*!
        \brief Checks if the cell given by the index \p i is on the border of the system.
      */
      CUDA_HOS_DEV bool is_border(const indexer i) const
      {
        const derived* dhis = static_cast<const derived*>(this);
        return dhis->is_border(dhis->to_cell(i));
      }
      
      /*!
        \brief Checks if the cell given by the tuple \p p is on the border of the system.
      */
      CUDA_HOS_DEV bool is_border(const vector_type<indexer, num_dims> &p) const
      {
        static_assert(AFFPICS_INHERITANCE_HACK_CHECK(), "Should define this somewhere else! Blame C++ for the lack of virtual templates...");
        return false;
      }
      
      /*!
        \brief Checks if the cell given by the index \p i is outside of the system.
      */
      CUDA_HOS_DEV bool is_outside(const indexer i) const
      {
        const derived* dhis = static_cast<const derived*>(this);
        return dhis->is_outside(dhis->to_cell(i));
      }
      
      /*!
        \brief Checks if the cell given by the tuple \p p is outside of the system.
      */
      CUDA_HOS_DEV bool is_outside(const vector_type<indexer, num_dims> &p) const
      {
        static_assert(AFFPICS_INHERITANCE_HACK_CHECK(), "Should define this somewhere else! Blame C++ for the lack of virtual templates...");
        return false;
      }
      
      /*!
        \brief Returns the maximum number of neighbouring cells for whose charge density a particle
               of type \p part_type may still contribute.
      */
      template <class part_type>
      CUDA_HOS_DEV constexpr indexer particle_cell_radius(const part_type& part) const
      {
        //static_assert(AFFPICS_INHERITANCE_HACK_CHECK(), "Should define this somewhere else! Blame C++ for the lack of virtual templates...");
        return 0;
      }
      
      /*!
        \brief Gives the fraction of a particle that is inside a given cell
               when the particle has a position given by \p pos (in units of cell size).
      */
      template <class part_type>
      CUDA_HOS_DEV FLType particle_fraction(const part_type& part, const vector_type<FLType, num_dims> &pos) const
      {
        static_assert(AFFPICS_INHERITANCE_HACK_CHECK(), "Should define this somewhere else! Blame C++ for the lack of virtual templates...");
        return 0;
      }
      
      /*!
        \brief Returns the position within the cell (in units of cell size) where the
               electric field is measured in the \p dim dimension.
      */
      CUDA_HOS_DEV vector_type<FLType, num_dims> E_measurement(const indexer dim) const
      {
        static_assert(AFFPICS_INHERITANCE_HACK_CHECK(), "Should define this somewhere else! Blame C++ for the lack of virtual templates...");
        return vector_type<FLType, num_dims>(0);
      }
      
      /*!
        \brief Returns the position within the cell (in units of cell size) where the
               magnetic field is measured in the \p dim dimension.
      */
      CUDA_HOS_DEV vector_type<FLType, num_dims> B_measurement(const indexer dim) const
      {
        static_assert(AFFPICS_INHERITANCE_HACK_CHECK(), "Should define this somewhere else! Blame C++ for the lack of virtual templates...");
        return vector_type<FLType, num_dims>(0);
      }
      
    private:
      
      template <indexer dim, bool check_for_border, class Func, class ... Args>
      CUDA_HOS_DEV void for_all_neighbours_impl( const indexer radius,
                                                 const vector_type<indexer, num_dims> &cell,
                                                 Func && func, Args&& ... args                             ) const
      {
        const derived* dhis = static_cast<const derived*>(this);
        if constexpr (dim == num_dims)
          {
            func(dhis->to_index(cell), cell, vector_type<bool, num_dims>(false), std::forward<Args>(args)...);
          }
        else if constexpr (dim < num_dims)
          {
            if constexpr (check_for_border)
              {
                using namespace std;
                for (indexer i = max(cell[dim], radius) - radius; i <= cell[dim] + radius && i < dhis->num_cells(dim); ++i)
                  {
                    for_all_neighbours_impl<dim + 1, check_for_border, Func, Args...>
                      (radius, cell.set(dim, i), std::forward<Func>(func), std::forward<Args>(args)...);
                  }
              }
            else
              {
                for (indexer i = cell[dim] - radius; i <= cell[dim] + radius; ++i)
                  {
                    for_all_neighbours_impl<dim + 1, check_for_border, Func, Args...>
                      (radius, cell.set(dim, i), std::forward<Func>(func), std::forward<Args>(args)...);
                  }
              }
          }
      }
      
    public:
      
      /*!
        \brief Applies \p func to all cells within radius \p radius of the cell given by \p index.
        
        \param radius The radius within which we will operate.
        
        \param index The index of the cell around which we will operate.
        
        \param func The function (or functor, or lambda) that will be applied.
        
        \param args Any extra arguments for the function.
        
        \tparam check_for_border If `true`, carefully check for not going outside the boundaries of the array.
                                 However, if it is already known that the cell is not at the border,
                                 the branching can be reduced by skipping that check.
        
        \tparam Func The type of the function that will be applied.
        
        \tparam Args The types of the extra arguments that will be passed to the function.
        
        \pre \p func must take as arguments:
                  * An `indexer` that corresponds to the index of the cell being evaluated
                  * A `const vector_type<indexer, num_dims> &` that also corresponds to that cell
                    (it's passed since it is needed for the neighbour-finding algorithm)
                  * A `const vector_type<bool, num_dims> &` that indicates
                    if the values should be mirrored for each direction (for reflecting boundary conditions)
                  * The arguments given by \p args.
        
        \remark This must check (in case of periodic boundary conditions)
                for neighbours on the other side.
      */
      template <bool check_for_border = true, class Func, class ... Args>
      CUDA_HOS_DEV void for_all_neighbours(const indexer radius, const indexer index, Func && func, Args&& ... args) const
      {
        const derived* dhis = static_cast<const derived*>(this);
        dhis-> template for_all_neighbours<check_for_border, Func, Args...>
                          (radius, dhis->to_cell(index), std::forward<Func>(func), std::forward<Args>(args)...);
      }
      
      /*!
        \brief Applies \p func to all cells within radius \p radius of the cell given by \p cell.
        
        \param radius The radius within which we will operate.
        
        \param cell The tuple that gives the cell around which we will operate.
        
        \param func The function (or functor, or lambda) that will be applied.
        
        \param args Any extra arguments for the function.
        
        \tparam check_for_border If `true`, carefully check for not going outside the boundaries of the array.
                                 However, if it is already known that the cell is not at the border,
                                 the branching can be reduced by skipping that check.
        
        \tparam Func The type of the function that will be applied.
        
        \tparam Args The types of the extra arguments that will be passed to the function.
        
        \pre \p func must take as arguments:
                  * An `indexer` that corresponds to the index of the cell being evaluated
                  * A `const vector_type<indexer, num_dims> &` that also corresponds to that cell
                    (it's passed since it is needed for the neighbour-finding algorithm)
                  * A `const vector_type<bool, num_dims> &` that indicates
                    if the values should be mirrored for each direction (for reflecting boundary conditions)
                  * The arguments given by \p args.
                  
        \remark This must check (in case of periodic boundary conditions)
                for neighbours on the other side.
      */
      template <bool check_for_border = true, class Func, class ... Args>
      CUDA_HOS_DEV void for_all_neighbours( const indexer radius,
                                            const vector_type<indexer, num_dims> &cell,
                                            Func && func, Args&& ... args                             ) const
      {
        const derived* dhis = static_cast<const derived*>(this);
        for_all_neighbours_impl<0, check_for_border, Func, Args...>
                          (radius, cell, std::forward<Func>(func), std::forward<Args>(args)...);
      }
      
      /*!
        \brief Applies boundary conditions to particle \p part, according to its position.
               Should mostly change position and velocity...
        
        \param part The particle to which the boundary conditions may be applied.
        
        \param force_apply If `true`, skip checks and apply boundary conditions even if inside the system.
        
        \remark The current architecture does not easily support particle creation or destruction,
                so absorbing boundary conditions are not easily implemented.
      */
      template <class particle>
      CUDA_HOS_DEV void boundary_particles (particle& part, const bool force_apply = false) const
      {
        static_assert(AFFPICS_INHERITANCE_HACK_CHECK(), "Should define this somewhere else! Blame C++ for the lack of virtual templates...");
      }
      
      /*!
        \brief Gives the electric field at the cell (outside the system)
               specified by the tuple (with possible negative values) given by \p cell,
               taking into account the electric fields \p E_fields.
               
        \remark If \p cell is inside the system, this must return the correct value for the electric field!
      */
      template <class ArrT>
      CUDA_HOS_DEV E_field_type<num_dims> boundary_E(const vector_type<indexer, num_dims> &cell,
                                                     const ArrT & E_fields                                      ) const
      {
        static_assert(AFFPICS_INHERITANCE_HACK_CHECK(), "Should define this somewhere else! Blame C++ for the lack of virtual templates...");
        return E_field_type<num_dims>(0);
      }
      
      /*!
        \brief Gives the magnetic field at the cell (outside the system)
               specified by the tuple (with possible negative values) given by \p cell,
               taking into account the magnetic fields \p B_fields.
               
        \remark If \p cell is inside the system, this must return the correct value for the electric field!
      */
      template <class ArrT>
      CUDA_HOS_DEV B_field_type<num_dims> boundary_B(const vector_type<indexer, num_dims> &cell,
                                                     const ArrT & B_fields                                      ) const
      {
        static_assert(AFFPICS_INHERITANCE_HACK_CHECK(), "Should define this somewhere else! Blame C++ for the lack of virtual templates...");
        return B_field_type<num_dims>(0);
      }
                                                             
      
      /*!
        \brief Gives the current at the cell (outside the system)
               specified by the tuple (with possible negative values) given by \p cell,
               taking into account the currents \p currents.
      */
      template <class ArrT>
      CUDA_HOS_DEV current_type<num_dims> boundary_J(const vector_type<indexer, num_dims> &cell,
                                                     const ArrT & currents                                      ) const
      {
        static_assert(AFFPICS_INHERITANCE_HACK_CHECK(), "Should define this somewhere else! Blame C++ for the lack of virtual templates...");
        return current_type<num_dims>(0);
      }
                                                             
                                                             
      /*!
        \brief Sets the initial conditions for the system.
        
        \warning \p PartStorage will be something like utilities/ParticleStorage...
      */
      template <class EArr, class BArr, class JArr, class PartStorage>
      CUDA_HOS_DEV void initial_condition(EArr& E_fields, BArr & B_fields, JArr & currents, PartStorage& particles) const
      {
        static_assert(AFFPICS_INHERITANCE_HACK_CHECK(), "Should define this somewhere else! Blame C++ for the lack of virtual templates...");
      }
      
    };
  }
}

#endif