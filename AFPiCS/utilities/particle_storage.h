#ifndef AFFPICS_PARTICLES
#define AFFPICS_PARTICLES

/*!
  \file particle_storage.h
  
  \brief The class that enables storage of a compile-time specified
         collection of different particle types.
         (In essence, a simplified version of a std::tuple...)
  
  \author Nuno Fernandes
*/

#include "../header.h"

namespace AFFPiCS
{
  
  template <class parallelism, class particle>
  struct particle_storage_part
  {
    particle_holder<parallelism, particle> particles;
    
    template <class stream> void save(stream &s, bool binary = Defaults::data_i_o_as_binary) const
    {
      if (binary)
        {
          particles.binary_output(s);
        }
      else
        {
          particles.textual_output(s);
        }
    }
    
    template <class stream> void load(stream &s, bool binary = Defaults::data_i_o_as_binary)
    {
      if (binary)
        {
          particles.binary_input(s);
        }
      else
        {
          particles.textual_input(s);
        }
    }
  };

  template <class parallelism, class ... particles>
  struct particle_storage : public particle_storage_part<parallelism, particles>...
  {
    
    private:
    
    template <class stream, class Arg, class ... Args>
    static void save_helper(stream &s, bool binary, const Arg& one, const Args& ... others)
    {
      one.save(s, binary);
      save_helper(s, binary, others...);
    }
    
    template <class stream, class Arg> static void save_helper(stream &s, bool binary, const Arg& one)
    {
      one.save(s, binary);
    }
    
    template <class stream, class Arg, class ... Args>
    static void load_helper(stream &s, bool binary, Arg& one, Args& ... others)
    {
      one.load(s, binary);
      load_helper(s, binary, others...);
    }
    
    template <class stream, class Arg> static void load_helper(stream &s, bool binary, Arg& one)
    {
      one.load(s, binary);
    }
  
    template <class Arg, class ... Args>
    indexer size_helper(const Arg& one, const Args& ... others) const
    {
      const indexer s1 = one.particles.size();
      const indexer s2 = size_helper(others...);
      if (s2 > s1)
        {
          return s2;
        }
      else
        {
          return s1;
        }
    }
    
    template <class Arg>
    indexer size_helper(const Arg& one) const
    {
      return one.particles.size();
    }
    
    public:
    
    template <class stream> void save(stream &s, bool binary = Defaults::data_i_o_as_binary) const
    {
      save_helper(s, binary, static_cast<const particle_storage_part<parallelism, particles>&>(*this)...);
    }
    
    
    template <class stream> void load(stream &s, bool binary = Defaults::data_i_o_as_binary)
    {
      load_helper(s, binary, static_cast<particle_storage_part<parallelism, particles>&>(*this)...);
    }
    
    
    indexer size() const
    {
      return size_helper(static_cast<const particle_storage_part<parallelism, particles>&>(*this)...);
    }
    
    template <class particle>
    particle_holder<parallelism, particle>& get_particles()
    {
      return static_cast<particle_storage_part<parallelism, particle>*>(this)->particles;
    }
    
    template <class particle>
    const particle_holder<parallelism, particle>& get_particles() const
    {
      return static_cast<const particle_storage_part<parallelism, particle>*>(this)->particles;
    }
  };
  
}

#endif