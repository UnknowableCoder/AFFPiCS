#ifndef AFFPICS_DEPOSITERS_NO_DEPOSITER
#define AFFPICS_DEPOSITERS_NO_DEPOSITER

/*!
  \file NoDepositer.h
  
  \brief Depositer that does exactly nothing.
  
  \author Nuno Fernandes
*/

#include "../header.h"
#include "../utilities/particle_storage.h"

namespace AFFPiCS
{
  namespace Depositers
  {
    template <class system_info, indexer num_dims, template <indexer> class ... particles>
    class None
    {
      
      private:
      template <class parallelism>
      using particle_storage_type = particle_storage<parallelism, particles<num_dims>...>;
      
      public:
      
      template <class parallelism> struct storage
      {
        
        void initialize( const particle_storage_type<parallelism> &part_store,
                         const current_holder<parallelism, num_dims> &currents,
                         const system_info& info                                )
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
      static results deposit(storage<parallelism> &store,
                             current_holder<parallelism, num_dims> &currents,
                             const particle_storage_type<parallelism>& part_storage,
                             const FLType dt,
                             const system_info& info)
      {
        return results{};
      }
    };
  }
}

#endif