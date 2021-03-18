#ifndef AFFPICS_PUSHERS_SIMPLE_PUSHER
#define AFFPICS_PUSHERS_SIMPLE_PUSHER


/*!
  \file simple_pusher.h
  
  \brief A general class to implement a particle pusher from a functor.
  
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
    class SimplePusher
    {    
      private:
      template <class parallelism>
      using particle_storage_type = particle_storage<parallelism, particles<num_dims>...>;
      
      public:
      
      template <class parallelism> struct storage
      {
        typename parallelism::kernel_size_type kernel[sizeof...(particles)];
        
        private:
        
        template <indexer idx, class part, class ... parts> void initialize_in(const particle_storage_type<parallelism> &part_store)
        {
          initialize_single<idx, part>(part_store);
          if constexpr (sizeof...(parts) > 0)
            {
              initialize_in<idx + 1, parts...>(part_store);
            }
        }
        
        template <indexer idx, class part> void initialize_single(const particle_storage_type<parallelism> &part_store)
        {
          kernel[idx] = parallelism::template estimate_loop_kernel_size
                                      < particle_holder<parallelism, part>, pusher_functor,
                                        E_field_holder<parallelism, num_dims>,
                                        B_field_holder<parallelism, num_dims>,
                                        FLType, system_info                     >
                                    (part_store.template get_particles<part>().size());
        }
        
        public:
        
        void initialize( const particle_storage_type<parallelism> &part_store,
                         const E_field_holder<parallelism, num_dims> &E_fields,
                         const B_field_holder<parallelism, num_dims> &B_fields,
                         const system_info& info                                )
        {
          initialize_in<0, particles<num_dims>...>(part_store);
        }
        
        template <class stream> void save(stream &s, bool binary = Defaults::data_i_o_as_binary) const
        {
        }
        
        template <class stream> void load(stream &s, bool binary = Defaults::data_i_o_as_binary)
        {
        }
      };
     
      protected:
      
      template <indexer idx, class parallelism, class part, class ... parts>
      static void push_impl(storage<parallelism> &store,
                            particle_storage_type<parallelism>& part_storage,
                            const E_field_holder<parallelism, num_dims> &E_fields,
                            const B_field_holder<parallelism, num_dims> &B_fields,
                            const FLType dt,
                            const system_info& info)
      {
        push_impl_single<idx, parallelism, part>(store, part_storage, E_fields, B_fields, dt, info);
        if constexpr (sizeof...(parts) > 0)
          {
            push_impl<idx + 1, parallelism, parts...>(store, part_storage, E_fields, B_fields, dt, info);
          }
      }
      
      template <indexer idx, class parallelism, class part>
      static void push_impl_single(storage<parallelism> &store,
                                   particle_storage_type<parallelism>& part_storage,
                                   const E_field_holder<parallelism, num_dims> &E_fields,
                                   const B_field_holder<parallelism, num_dims> &B_fields,
                                   const FLType dt,
                                   const system_info& info)
      {
        parallelism::loop(store.kernel[idx], part_storage.template get_particles<part>(),
                          pusher_functor{}, E_fields, B_fields, dt, info);
      }
      
      public:
      
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
        push_impl<0, parallelism, particles<num_dims>...>(store, part_storage, E_fields, B_fields, dt, info);
        return results{};
      }
    };
  }
}

#endif