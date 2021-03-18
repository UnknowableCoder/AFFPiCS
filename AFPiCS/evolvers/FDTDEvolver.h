#ifndef AFFPICS_EVOLVERS_FDTD_EVOLVER
#define AFFPICS_EVOLVERS_FDTD_EVOLVER


/*!
  \file FDTDEvolver.h
  
  \brief Evolves the fields according to a finite-difference time-domain method
         like the Yee method.
  
  \author Nuno Fernandes
*/

#include "../header.h"

namespace AFFPiCS
{
  namespace Evolvers
  {
    /*!
      \brief Evolves the fields according to a finite-difference time-domain method
             like the Yee method.
    */
    template <class system_info, indexer num_dims>
    class FDTD
    {
      struct B_Evolve_Functor
      {
        template <class B_arr, class E_arr, class S_Info>
        CUDA_HOS_DEV void operator() ( B_arr & B_fields,
                                       const indexer i,
                                       const E_arr & E_fields,
                                       const FLType dt,
                                       const S_Info& info        ) const
        {
          B_fields[i] = B_fields[i] - info.E_curl(E_fields, i) * dt;
        }
      };
      
      struct E_Evolve_Functor
      {
        template <class E_arr, class B_arr, class J_arr, class S_Info>
        CUDA_HOS_DEV void operator() ( E_arr & E_fields,
                                       const indexer i,
                                       const B_arr & B_fields,
                                       const J_arr & currents,
                                       const FLType dt,
                                       const S_Info& info          ) const
        {
          E_fields[i] = E_fields[i] + (info.B_curl(B_fields, i)/info.epsilon(i)/info.mu(i) - currents[i]/info.epsilon(i)) * dt;
        }
      };
      public:
        template <class parallelism> struct storage
        {
          typename parallelism::kernel_size_type E_kernel, B_kernel;
          
          void initialize(const E_field_holder<parallelism, num_dims> &E_fields,
                          const B_field_holder<parallelism, num_dims> &B_fields,
                          const current_holder<parallelism, num_dims> &currents,
                          const system_info& info                                 )
          {
            E_kernel = parallelism::template estimate_loop_kernel_size
                                      < E_field_holder<parallelism, num_dims>,
                                        E_Evolve_Functor,
                                        B_field_holder<parallelism, num_dims>,
                                        current_holder<parallelism, num_dims>,
                                        FLType, system_info                     > (E_fields.size());
            B_kernel = parallelism::template estimate_loop_kernel_size
                                      < B_field_holder<parallelism, num_dims>,
                                        B_Evolve_Functor,
                                        E_field_holder<parallelism, num_dims>,
                                        FLType, system_info                     > (E_fields.size());
          }
          
          
          template <class stream> void save(stream &s, bool binary = Defaults::data_i_o_as_binary) const
          {
            
          }
          
          
          template <class stream> void load(stream &s, bool binary = Defaults::data_i_o_as_binary)
          {
          }
        };
        
        struct results {};
        //There's no need to return anything from here.
        //Might be useful for debugging, but later...
      
        template <class parallelism>
        static results evolve ( storage<parallelism> &store,
                                E_field_holder<parallelism, num_dims> &E_fields,
                                B_field_holder<parallelism, num_dims> &B_fields,
                                const current_holder<parallelism, num_dims> &currents,
                                const FLType dt,
                                const system_info &info                                 )
        {
          parallelism::loop(store.B_kernel, B_fields, B_Evolve_Functor{}, E_fields, dt/2, info);
          parallelism::loop(store.E_kernel, E_fields, E_Evolve_Functor{}, B_fields, currents, dt, info);
          parallelism::loop(store.B_kernel, B_fields, B_Evolve_Functor{}, E_fields, dt/2, info);
          return results{};
        }
    };
  }
}

#endif