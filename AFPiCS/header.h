#ifndef AFFPICS_HEADER
#define AFFPICS_HEADER

/*!
  \file header.h
  
  \brief Some general purpose definitions and inclusions for the project.
  
  \author Nuno Fernandes
*/

#include <cstdint>
#include <string>
#include <cmath>
#include <utility>

namespace AFFPiCS
{
  using indexer = int64_t;
  using FLType = double;
  //To allow easier adjustment to other types, e. g., for use in GPUs (32 bit works best there)
  
  using StrType = std::string;
  //There may be situations in which we could want to change the string type...
  
  namespace Defaults
  {
    inline static constexpr indexer max_iterations = 10000;
    inline static constexpr FLType precision = 1e-6;
    inline static constexpr indexer derivative_accuracy = 1;
    
    /*! \brief Decide if data input/output is binary or text.
      
        \remark Though binary reduces portability between different systems
                (in the current implementation),
                it's obviously much faster and smaller.
    !*/
    inline constexpr bool data_i_o_as_binary = true;
    
    /*! \brief The extension for the files used to store simulations.
    */
    inline static StrType file_extension = StrType(".dat");
    
  }
}

namespace g24_lib
{
  class Configuration
  {
    public:
    
    using default_unsigned_indexer = AFFPiCS::indexer;
    //(The 'unsigned' just means negatives needn't be supported,
    // so this is an entirely valid choice.)
    using default_signed_indexer = AFFPiCS::indexer;
    using default_floating_point = AFFPiCS::FLType;
    
    using elementwise_point_operators = std::true_type;
    //This is just to allow solving of vectorial laplace equations in several dimensions.
    
    using throw_exceptions = std::true_type;
    //This is to allow exceptions to be toggled
    
    using print_exceptions = std::true_type;
  };
  //To configure the G24 Lib library.
}

#include "g24_lib.h"

namespace Simbpolic
{
  class Configuration
  {
    public:
    
    using ResultType = AFFPiCS::FLType;
    using IntegerType = AFFPiCS::indexer;
  };
}
//For now, the only option for particle shapes
//inclues the use of Simbpolic's Simple Symbolic computations,
//but, in general, we might not need it;
//in any case, we provide the configuration
//to be consistent with the choice of types for the rest of the code.
//#include "simbpolic.h"

namespace AFFPiCS
{

  template <class parallelism, class particle>
  using particle_holder = g24_lib::array_parallel<parallelism, particle>;
  
  template <class T, indexer num_dims>
  using vector_type = g24_lib::fspoint<T, indexer, num_dims>;
  
  template <indexer dimensions> inline constexpr indexer electric_field_dimensions()
  {
    return dimensions;
  }
  
  template <indexer dimensions> inline constexpr indexer magnetic_field_dimensions()
  {
    if constexpr (dimensions == 2)
    //A 2D simulation has B perpendicular to the E-plane.
    {
      return 1;
    }
    else if constexpr (dimensions == 1)
    //A 1D simulation would have B in a plane surrounding E.
    {
      return 2;
    }
    else
    {
      return dimensions;
    }
  }  
  
  template <indexer num_dims>
  using E_field_type = g24_lib::fspoint< FLType, indexer, electric_field_dimensions<num_dims>() >;
  
  template <indexer num_dims>
  using B_field_type = g24_lib::fspoint< FLType, indexer, magnetic_field_dimensions<num_dims>() >;
  
  template <indexer num_dims>
  using current_type = E_field_type<num_dims>;
  //The currents will need to have the same dimensionality as the electric field.
  
  template <class parallelism, indexer num_dims>
  using E_field_holder = g24_lib::array_parallel<parallelism, E_field_type<num_dims>>;
  
  template <class parallelism, indexer num_dims>
  using B_field_holder = g24_lib::array_parallel<parallelism, B_field_type<num_dims>>;
  
  template <class parallelism, indexer num_dims>
  using current_holder = g24_lib::array_parallel<parallelism, current_type<num_dims>>;
  
  
  struct Saver
  //This kind-of allows us to use signal()
  //to catch interruptions and save our work
  //before exiting the program.
  {
    private:
      StrType name;
      bool save_on_exit;
      bool save_on_interrupt;
    public:
  
    virtual void save(const bool = Defaults::data_i_o_as_binary) const 
    {
      return;
    }
    
    virtual void save(const StrType&, const bool = Defaults::data_i_o_as_binary) const 
    {
      return;
    }
    
    virtual const StrType& get_name() const 
    {
      return name;
    }
    
    virtual void set_name(const StrType& new_name)
    {
      name = new_name;
    }
    
    virtual bool get_save_on_exit() const
    {
      return save_on_exit;
    }
    
    virtual void set_save_on_exit(const bool new_save_on_exit)
    {
      save_on_exit = new_save_on_exit;
    }
    
    virtual bool get_save_on_interrupt() const
    {
      return save_on_interrupt;
    }
    
    virtual void set_save_on_interrupt(const bool new_save_on_interrupt)
    {
      save_on_interrupt = new_save_on_interrupt;
    }
    
    virtual void set_save_on_all(const bool save_on_all)
    {
      save_on_exit = save_on_all;
      save_on_interrupt = save_on_all;
    }
    
  };
}

#include <csignal>

namespace AFFPiCS
{
  /*! \brief We use global memory just to ensure save on interrupt.
  !*/
  namespace global
  {
#ifdef DEBUG
  #if DEBUG_TO_LOG
    std::ofstream log("AFFPiCS.log");
  #else
    auto& log = std::cout;
  #endif
#else
    
    struct AllEater
    {    
      template <class ... Args>
      void textual_output(const Args&... args) const
      {
      }
      template <class ... Args>
      void textual_input(const Args&... args)
      {
      }
      template <class ... Args>
      void binary_output(const Args&... args) const
      {
      }
      template <class ... Args>
      void binary_input(const Args&... args)
      {
      }     
      
    };
    
    template <class Obj>
    AllEater& operator<< (AllEater& eat, const Obj& obj)
    {
      return eat;
    }
    
    template <class A = char, class B = std::char_traits<A>>
    AllEater& operator<< (AllEater& eat, std::basic_ostream<A,B>& (*pf)(std::basic_ostream<A,B>&))
    {
      return eat;
    }
    
    AllEater log;
    
#endif
    inline Saver *running = nullptr;
    //It's inline to ensure compatibility
    //across multiple translation units.
  }
  
  inline void interrupt_backup(int signal)
  //A function to handle saving on interrupt.
  {
    if(global::running != nullptr)
      {
        if (global::running->get_save_on_interrupt())
          {
            global::running->save(global::running->get_name() + StrType("_interrupt"));
            global::running->set_save_on_all(false);
          }
      }
    exit(0);
  }
      
  inline void establish_backup()
  //This, when called, sets up the necessary callback
  //to save on an interrupt.
  {
    signal(SIGINT, interrupt_backup);
  }
  
}

#if AFFPICS_SKIP_INHERITANCE_CHECK
#define AFFPICS_INHERITANCE_HACK_CHECK() sizeof(char) > 0
#else
#define AFFPICS_INHERITANCE_HACK_CHECK() sizeof(char) == 0
#endif

#endif