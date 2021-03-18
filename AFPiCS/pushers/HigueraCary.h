#ifndef AFFPICS_PUSHERS_HIGUERA_CARY
#define AFFPICS_PUSHERS_HIGUERA_CARY


/*!
  \file HigueraCary.h
  
  \brief The Higuera-Cary Pusher.
  
  \author Nuno Fernandes
*/

#include "../header.h"
#include "simple_pusher.h"

namespace AFFPiCS
{

  namespace PusherFunctors
  {
    struct HigueraCary
    {
      template<class part_arr, class E_arr, class B_arr, class S_Info>
      CUDA_HOS_DEV void operator() ( part_arr& parts,
                                     const indexer i,
                                     const E_arr & E_fields,
                                     const B_arr & B_fields,
                                     const FLType dt,
                                     const S_Info & info       ) const
      {
        auto&& particle = parts[i];
        
        const FLType q_dt_m_factor = particle.charge(info) * dt/2/particle.mass(info);
        //(q dt)/(2 m)
        
        //particle.move(particle.u(info).element_divide(info.cell_sizes())/particle.gamma(info) * dt/2);
        //Half update with the previous velocity;
        
        
        const auto u_minus = particle.u(info) + info.E_gather(E_fields, particle)
                                  * q_dt_m_factor;
        
        
        const auto tau = info.B_gather(B_fields, particle) * q_dt_m_factor;
        
        const FLType u_star = AFFPiCS::dot_product(u_minus, tau)/info.units().c();
        
        using namespace std;
        
        
        const FLType sigma = FLType(1) + u_minus.square_norm2()/info.units().c()/info.units().c() - tau.square_norm2();
        
        const auto t_vec = tau/sqrt( (sigma + sqrt(sigma * sigma+4*(tau.square_norm2()+u_star * u_star)))/2 );
        
        const auto u_plus = (u_minus + t_vec * AFFPiCS::dot_product(u_minus, t_vec) +
                              AFFPiCS::cross_product(u_minus, t_vec)                  )/(1 + t_vec.square_norm2());
        
        particle.set_u(u_plus + q_dt_m_factor * info.E_gather(E_fields, particle) +
                        AFFPiCS::cross_product(u_minus, t_vec), info                                            );
        
        //particle.move(particle.u(info).element_divide(info.cell_sizes())/particle.gamma(info) * dt/2, info);
        //Half update with the next velocity;
        
        parts[i] = particle;
      }
    };
  }
  
  namespace Pushers
  {
    template <class system_info, indexer num_dims, template <indexer> class ... particles>
    using HigueraCary = SimplePusher<PusherFunctors::HigueraCary, system_info, num_dims, particles...>;
  }
}

#endif