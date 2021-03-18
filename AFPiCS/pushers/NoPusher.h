#ifndef AFFPICS_PUSHERS_NO_PUSHER
#define AFFPICS_PUSHERS_NO_PUSHER


/*!
  \file NoPusher.h
  
  \brief A pusher that does exactly nothing.
  
  \author Nuno Fernandes
*/

#include "../header.h"
#include "../utilities/helpers.h"
#include "../utilities/particle_storage.h"

namespace AFFPiCS
{
  namespace Pushers
  {
    template <class pusher_functor, class system_info, indexer num_dims, template <indexer> class ... particles>
    class None
    {    
      private:
      template <class parallelism>
      using particle_storage_type = particle_storage<parallelism, particles<num_dims>...>;
      
      public:
      
      template <class parallelism> struct storage
      {
        void initialize( const particle_storage_type<parallelism> &part_store,
                         const E_field_holder<parallelism, num_dims> &E_fields,
                         const B_field_holder<parallelism, num_dims> &B_fields,
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
      static results push(storage<parallelism> &store,
                               particle_storage_type<parallelism>& part_storage,
                               const E_field_holder<parallelism, num_dims> &E_fields,
                               const B_field_holder<parallelism, num_dims> &B_fields,
                               const FLType dt,
                               const system_info& info)
      {
        return results{};
      }
    };
  }
}

#endif