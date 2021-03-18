#ifndef AFFPICS_UNIT_SYSTEM
#define AFFPICS_UNIT_SYSTEM

/*!
  \file unit_system.h
  
  \brief A quick and easy way to define a system of units to be used in the simulation.
  
  \author Nuno Fernandes
*/

#include "../header.h"

namespace AFFPiCS
{
  /*!
    \brief A simple and fast approach to unit systems.
    
    \remark We skip candela and mole since they shouldn't be too relevant.
  */
  class UnitSystem
  {
    private:
    
    FLType length, time, mass, current, temperature;
    
    public:
    
    CUDA_HOS_DEV constexpr FLType length_unit() const
    {
      return length;
    }
    CUDA_HOS_DEV constexpr FLType time_unit() const
    {
      return time;
    }
    CUDA_HOS_DEV constexpr FLType mass_unit() const
    {
      return mass;
    }
    CUDA_HOS_DEV constexpr FLType current_unit() const
    {
      return current;
    }
    CUDA_HOS_DEV constexpr FLType temperature_unit() const
    {
      return temperature;
    }
    
    CUDA_HOS_DEV constexpr FLType charge_unit() const
    {
      return current * time;
    }
    
    CUDA_HOS_DEV static constexpr FLType SI_c()
    {
      return 299792458;
    }
    
    CUDA_HOS_DEV constexpr FLType c() const
    {
      return SI_c() * time / length;
    }
    
    CUDA_HOS_DEV static constexpr FLType SI_epsilon_zero()
    {
      return 8.8541878128e-12;
    }
    
    CUDA_HOS_DEV constexpr FLType epsilon_zero() const
    {
      return  SI_epsilon_zero() * mass * g24_lib::fastpow(length, 3)
                / g24_lib::fastpow(current, 2) / g24_lib::fastpow(time, 4);
      //Farads/m
    }
    
    CUDA_HOS_DEV static constexpr FLType SI_mu_zero()
    {
      return 1.25663706212e-6;
    }
    
    CUDA_HOS_DEV constexpr FLType mu_zero() const
    {
      return SI_mu_zero() * g24_lib::fastpow(time, 2) * g24_lib::fastpow(current, 2) / mass / length;
    }
    
    CUDA_HOS_DEV static constexpr FLType SI_q_e()
    {
      return 1.602176634e-19;
    }
        
    CUDA_HOS_DEV constexpr FLType q_e() const
    {
      return SI_q_e() / current / time;
    }
    
    CUDA_HOS_DEV static constexpr FLType SI_Planck()
    {
      return 6.62607015e-34;
    }
    
    CUDA_HOS_DEV constexpr FLType Planck() const
    {
      return SI_h_bar() / time / mass / length / length;
    }
    
    CUDA_HOS_DEV static constexpr FLType SI_h_bar()
    {
      return SI_Planck() / g24_lib::pi<FLType>;
    }
    
    CUDA_HOS_DEV constexpr FLType h_bar() const
    {
      return Planck() / g24_lib::pi<FLType>;
    }
    
    CUDA_HOS_DEV static constexpr FLType SI_k_B()
    {
      return 1.380649e-23;
    }
    
    CUDA_HOS_DEV constexpr FLType k_B() const
    {
      return SI_k_B() * temperature * time * time / mass / length / length;
    }
    
    
    CUDA_HOS_DEV static constexpr FLType fine_structure()
    {
      return 0.0072973525693;
    }
    
    CUDA_HOS_DEV static constexpr FLType SI_m_e()
    {
      return 9.1093837015e-31;
    }
    
    CUDA_HOS_DEV constexpr FLType m_e() const
    {
      return SI_m_e() / mass;
    }
    
    CUDA_HOS_DEV static constexpr FLType SI_m_p()
    {
      return 1.67262192369e-27;
    }
    
    CUDA_HOS_DEV constexpr FLType m_p() const
    {
      return SI_m_p() / mass;
    }
    
    CUDA_HOS_DEV static constexpr FLType SI_m_n()
    {
      return 1.67492749804e-27;
    }
    
    CUDA_HOS_DEV constexpr FLType m_n() const
    {
      return SI_m_n() / mass;
    }
    
    /*!
      \brief Specifies the units to be used in SI values.
    */
    CUDA_HOS_DEV constexpr UnitSystem(const FLType l_unit = 1, const FLType t_unit = 1,
                                       const FLType m_unit = 1, const FLType I_unit = 1,
                                       const FLType Temp_unit = 1):
    length(l_unit), time(t_unit), mass(m_unit), current(I_unit), temperature(Temp_unit)
    {
    }
    
    
  };
  
  namespace DefaultUnits
  {
    inline static constexpr UnitSystem SI = UnitSystem{};
    
    inline static constexpr UnitSystem NuclearPhysics =
                            UnitSystem ( UnitSystem::SI_c()*UnitSystem::SI_h_bar()/UnitSystem::SI_q_e(),
                                         UnitSystem::SI_h_bar()/UnitSystem::SI_q_e(),
                                         UnitSystem::SI_q_e()/UnitSystem::SI_c()/UnitSystem::SI_c(),
                                         UnitSystem::SI_q_e()*constexpr_sqrt( UnitSystem::SI_epsilon_zero()*UnitSystem::SI_c()/
                                                                              UnitSystem::SI_h_bar()                   ),
                                         UnitSystem::SI_q_e()/UnitSystem::SI_k_B()                                        );
    //Things in eV, with the usual hbar = c = epsilon_zero = 1 shenanigans.
    
  }
  
}

#endif
