#ifndef AFFPICS_SIMUL
#define AFFPICS_SIMUL

/*!
  \file simul.h
  
  \brief The class that handles the simulation itself.
  
  \author Nuno Fernandes
*/

#include "header.h"
#include "utilities/particle_storage.h"
#include "utilities/diagnostic_handler.h"
#include <fstream>

namespace AFFPiCS
{

  template <class parallelism, indexer num_dims, class particle_pusher,
            class field_evolver, class charge_depositer, template <indexer> class ... Parts>
  struct simul_storage
  {
    typename parallelism::kernel_size_type move_kernel[sizeof...(Parts)];
  
    typename particle_pusher::template storage<parallelism> pusher;
    typename field_evolver::template storage<parallelism> evolver;
    typename charge_depositer::template storage<parallelism> depositer;
  
    particle_storage<parallelism, Parts<num_dims>...> particles;
  
    E_field_holder<parallelism, num_dims> E_fields;
    
    B_field_holder<parallelism, num_dims> B_fields;
    
    current_holder<parallelism, num_dims> currents;
    
    template <class system_info>
    void initialize(const system_info &info)
    {
      pusher.initialize(particles, E_fields, B_fields, info);
      evolver.initialize(E_fields, B_fields, currents, info);
      depositer.initialize(particles, currents, info);
    }
    
    template <class stream> void save(stream &s, bool binary = Defaults::data_i_o_as_binary) const
    {
      pusher.save(s, binary);
      evolver.save(s, binary);
      depositer.save(s, binary);
      particles.save(s, binary);
      if (binary)
        {
          E_fields.binary_output(s);
          B_fields.binary_output(s);
          currents.binary_output(s);
        }
      else
        {
          E_fields.textual_output(s);
          B_fields.textual_output(s);
          currents.textual_output(s);
        }
    }
    
    
    template <class stream> void load(stream &s, bool binary = Defaults::data_i_o_as_binary)
    {
      pusher.load(s, binary);
      evolver.load(s, binary);
      depositer.load(s, binary);
      particles.load(s, binary);
      if (binary)
        {
          E_fields.binary_input(s);
          B_fields.binary_input(s);
          currents.binary_input(s);
        }
      else
        {
          E_fields.textual_input(s);
          B_fields.textual_input(s);
          currents.textual_input(s);
        }
    }
        
  };
  
  template <class parallelism, indexer num_dims, class system_info,
            template <class, indexer, template <indexer> class ...> class pusher,
            template <class, indexer> class evolver,
            template <class, indexer, template <indexer> class ...> class depositer,
            template <indexer> class ... particles >
  class Simulation : public Saver
  {
    static_assert(g24_lib::is_parallelism<parallelism>, "parallelism must be a valid form of parallelism for g24_lib!");
    
    using particle_pusher = pusher<system_info, num_dims, particles...>;
    using field_evolver = evolver<system_info, num_dims>;
    using charge_depositer = depositer<system_info, num_dims, particles...>;
    
    using storage = simul_storage<parallelism, num_dims, particle_pusher, field_evolver, charge_depositer, particles...>;
        
    private:
      storage store;
      system_info info;
      bool initialized;
      
    public:
      
      const storage& get_storage() const
      {
        return store;
      }
      
      storage& get_storage()
      {
        return store;
      }
      
      static StrType default_name()
      {
        return StrType("PIC_simul");
      }
      
      Simulation(const system_info &s_info, const StrType& new_name = default_name()):
      info(s_info), initialized(false)
      {
        this->set_name(new_name);
        this->set_save_on_all(true);
      }
      
      void save(const StrType& save_name, const bool binary = Defaults::data_i_o_as_binary) const
      {
        std::ofstream file(save_name + Defaults::file_extension);
        store.save(file, binary);
        if (binary)
          {
            g24_lib::binary_output(file, initialized);
          }
        else
          {
            file << " ";
            g24_lib::textual_output(file, initialized);
          }
        file.close();
      }
      
      void save(const bool binary = Defaults::data_i_o_as_binary) const
      {
        save(this->get_name(), binary);
      }
      
      void load(const StrType& load_name, const bool binary = Defaults::data_i_o_as_binary)
      {
        std::ifstream file(save_name + Defaults::file_extension);
        store.load(file, binary);
        if (binary)
          {
            g24_lib::binary_input(file, initialized);
          }
        else
          {
            g24_lib::textual_input(file, initialized);
          }
        file.close();
        initialize(false);
      }
      
      void load(const bool binary = Defaults::data_i_o_as_binary)
      {
        load(this->get_name(), binary);
      }
      
      ~Simulation()
      {
        if (this->get_save_on_exit())
          {
            save(this->get_name() + StrType("_exit"));
          }
      }
      
      
      void set_particles(const particle_storage<parallelism, particles<num_dims>...> & new_particles)
      {
        store.particles = new_particles;
      }
      
      void set_E(const E_field_holder<parallelism, num_dims> & new_fields)
      {
        store.E_fields = new_fields;
      }
      
      void set_B(const B_field_holder<parallelism, num_dims> & new_fields)
      {
        store.B_fields = new_fields;
      }
      
      void set_fields( const E_field_holder<parallelism, num_dims> & new_E,
                       const B_field_holder<parallelism, num_dims> & new_B  )
      {
        store.E_fields = new_E;
        store.B_fields = new_B;
      }
      
      void set_currents(const current_holder<parallelism, num_dims> & new_currents)
      {
        store.currents = new_currents;
      }
      
      void set_info(const system_info &new_info)
      {
        info = new_info;
      }
      
      
      private:
      
      struct mover_functor
      //Moves the particles by half a timestep.
      {
        template <class PartArr, class S_Info>
        CUDA_HOS_DEV void operator() (PartArr &parts, const indexer i, const FLType dt, const S_Info &sys_info) const
        {
          parts[i].move(parts[i].vel(sys_info) * dt/2, sys_info);
        }
      };
      
      template <indexer idx, class part, class ... parts>
      void kernel_size_estimation()
      {
        kernel_size_estimation_single<idx, part>();
        if constexpr (sizeof...(parts) > 0)
           {
              kernel_size_estimation<idx + 1, parts...>();
           }
      }
      
      template <indexer idx, class part>
      void kernel_size_estimation_single()
      {
        store.move_kernel[idx] = parallelism::template estimate_loop_kernel_size
                                  <particle_holder<parallelism, part>, mover_functor, FLType, system_info>
                                  (store.particles.template get_particles<part>().size());
      }      
      
      template <indexer idx, class part, class ... parts>
      void half_move_impl(const FLType dt)
      {
        half_move_impl_single<idx, part>(dt);
        if constexpr (sizeof...(parts) > 0)
        {
          half_move_impl<idx, parts...>(dt);
        }
      }
      
      template <indexer idx, class part>
      void half_move_impl_single(const FLType dt)
      {
        parallelism::loop(store.particles.template get_particles<part>(), mover_functor{}, dt, info);
      }
      
      void half_move_particles(const FLType dt)
      {
        half_move_impl<0, particles<num_dims>...>(dt);
      }
      
      public:
      
      /*!
        \brief Initializes the fields and particles and the temporary storage.
        
        Fields and particles are set according to `system_info::initial_condition()`
        unless \p initial_condition is `false` (useful for when the fields and particles
        have been specified otherwise).
      */
      void initialize(const bool initial_condition = true)
      {
        if (initial_condition)
          {
            info.initial_condition(store.E_fields, store.B_fields, store.currents, store.particles);
            initialized = true;
          }
        store.initialize(info);
        kernel_size_estimation<0, particles<num_dims>...>();
      }
      
      /*!
        \brief This holds some possible useful results and/or diagnostic information
               from the three steps of the simulation.
      */
      struct simulation_results
      {
        typename particle_pusher::results pusher_results;
        typename field_evolver::results evolver_results;
        typename charge_depositer::results depositer_results;
      };
      
      /*!
        \brief Simulates one step of duration given by \p dt.
        
        \tparam diagnostics Allows some diagnostics to run at specific points in the simulation steps,
                            depending on the available static member functions of that class.
        
        All diagnostics must have the signature:
~~~~~{cpp}
void DIAGNOSTIC(const particle_storage<parallelism, particles<num_dims>...> &particles_in_the_system,
                const E_field_holder<parallelism, num_dims> &E_fields,
                const B_field_holder<parallelism, num_dims> &B_fields,
                const current_holder<parallelism, num_dims> &currents,
                const FLType dt, const system_info &info)
~~~~~
       The valid choices for DIAGNOSTIC are: `pre_step`, `before_mover`, `after_mover`,
       `before_pusher`, `after_pusher`, `before_evolver , `after_evolver`,
       `before_depositer`, `after_depositer` and `post_step`.
       If and only if these static functions exist, they are called at appropriate steps.
       
       All in all, the execution follows:
~~~~~
diagnostics::pre_step(...);
diagnostics::before_mover(...);

                    half_move_particles(...)

diagnostics::after_mover(...);
diagnostics::before_pusher(...);

                    pusher::push(...);

diagnostics::after_pusher(...);
diagnostics::before_evolver(...);

                    evolver::evolve(...);

diagnostics::after_evolver(...);
diagnostics::before_depositer(...);

                    depositer::deposit(...);
  
diagnostics::after_depositer(...);
diagnostics::before_mover(...);

                    half_move_particles(...)

diagnostics::after_mover(...);
diagnostics::post_step(...);
~~~~~


        With, once again, the diagnostic functions only being called if and only if they exist.
      */
      template <class diagnostics>
      simulation_results simulate_once(const FLType dt, diagnostics & diag)
      {
        simulation_results ret;
        
        if constexpr (diagnostic_handler<diagnostics>::pre_step)
          {
            diag.pre_step(store.particles, store.E_fields, store.B_fields, store.currents, dt, info);
          }
        
        if (initialized)
        //If initialized is true, the system has just been put to the initial conditions
        //we must update the currents by half a timestep before moving the particles.
        {
          ret.depositer_results = charge_depositer::template deposit<parallelism>
                              (store.depositer, store.currents, store.particles, dt/2, info);
          initialized = false;
        }
        
        if constexpr (diagnostic_handler<diagnostics>::before_mover)
          {
            diag.before_mover(store.particles, store.E_fields, store.B_fields, store.currents, dt, info);
          }
          
        half_move_particles(dt);
        
        if constexpr (diagnostic_handler<diagnostics>::after_mover)
          {
            diag.after_mover(store.particles, store.E_fields, store.B_fields, store.currents, dt, info);
          }
          
        if constexpr (diagnostic_handler<diagnostics>::before_pusher)
          {
            diag.before_pusher(store.particles, store.E_fields, store.B_fields, store.currents, dt, info);
          }
          
        ret.pusher_results = particle_pusher::template push<parallelism>
                            (store.pusher, store.particles, store.E_fields, store.B_fields, dt, info);
                                                        
        if constexpr (diagnostic_handler<diagnostics>::after_pusher)
          {
            diag.after_pusher(store.particles, store.E_fields, store.B_fields, store.currents, dt, info);
          }
                                       
        if constexpr (diagnostic_handler<diagnostics>::before_evolver)
          {
            diag.before_evolver(store.particles, store.E_fields, store.B_fields, store.currents, dt, info);
          }
          
        ret.evolver_results = field_evolver::template evolve<parallelism>
                            (store.evolver, store.E_fields, store.B_fields, store.currents, dt, info);
        
        if constexpr (diagnostic_handler<diagnostics>::after_evolver)
          {
            diag.after_evolver(store.particles, store.E_fields, store.B_fields, store.currents, dt, info);
          }
          
        if constexpr (diagnostic_handler<diagnostics>::before_depositer)
          {
            diag.before_depositer(store.particles, store.E_fields, store.B_fields, store.currents, dt, info);
          }
          
        ret.depositer_results = charge_depositer::template deposit<parallelism>
                            (store.depositer, store.currents, store.particles, dt, info);
        
        if constexpr (diagnostic_handler<diagnostics>::after_depositer)
          {
            diag.after_depositer(store.particles, store.E_fields, store.B_fields, store.currents, dt, info);
          }
        
        if constexpr (diagnostic_handler<diagnostics>::before_mover)
          {
            diag.before_mover(store.particles, store.E_fields, store.B_fields, store.currents, dt, info);
          }
          
        half_move_particles(dt);
        
        if constexpr (diagnostic_handler<diagnostics>::after_mover)
          {
            diag.after_mover(store.particles, store.E_fields, store.B_fields, store.currents, dt, info);
          }
          
        if constexpr (diagnostic_handler<diagnostics>::post_step)
          {
            diag.post_step(store.particles, store.E_fields, store.B_fields, store.currents, dt, info);
          }
          
        return ret;
      }
      
      template <class diagnostics = int>
      simulation_results simulate_once(const FLType dt)
      {
        diagnostics diag;
        return simulate_once(dt, diag);
      }
  };
  
}


#endif
