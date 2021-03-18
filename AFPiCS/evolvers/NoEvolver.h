#ifndef AFFPICS_EVOLVERS_NO_EVOLVER
#define AFFPICS_EVOLVERS_NO_EVOLVER


/*!
  \file NoEvolver.h
  
  \brief Evolver that does exactly nothing.
  
  \author Nuno Fernandes
*/

#include "../header.h"

namespace AFFPiCS
{
  namespace Evolvers
  {
    template <class system_info, indexer num_dims>
    class None
    {
      public:
        template <class parallelism> struct storage
        {
          void initialize(const E_field_holder<parallelism, num_dims> &E_fields,
                          const B_field_holder<parallelism, num_dims> &B_fields,
                          const current_holder<parallelism, num_dims> &currents,
                          const system_info& info                                 )
          {
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
          return results{};
        }
    };
  }
}

#endif