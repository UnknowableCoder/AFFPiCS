#ifndef AFFPICS_PUSHERS_VAY
#define AFFPICS_PUSHERS_VAY


/*!
  \file Vay.h
  
  \brief The Vay Pusher.
  
  \author Nuno Fernandes
*/

#include "../header.h"
#include "simple_pusher.h"

namespace AFFPiCS
{

  namespace PusherFunctors
  {
    struct Vay
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
        
        //particle.move(particle.u(info).element_divide(info.cell_sizes())/particle.gamma(info) * dt/2);
        //Half update with the previous velocity;
        
        const FLType q_dt_m_factor = particle.charge(info) * dt/2/particle.mass(info);
        //(q dt)/(2 m)
        
        const auto u_half = particle.u(info) +
                            (info.E_gather(E_fields, particle) +
                             AFFPiCS::cross_product(particle.u(info)/particle.gamma(info),
                                                       info.B_gather(B_fields, particle)
                                                      )
                            ) * q_dt_m_factor;
        
        const auto u_prime = u_half + info.E_gather(E_fields, particle) * q_dt_m_factor;
        
        const auto tau = info.B_gather(B_fields, particle) * q_dt_m_factor;
        
        const FLType u_star = AFFPiCS::dot_product(u_prime, tau)/info.units().c();
        
        using namespace std;
                
        const FLType sigma = FLType(1) + u_prime.square_norm2()/info.units().c()/info.units().c() - tau.square_norm2();
                
        
        const auto t_vec = tau/sqrt( (sigma + sqrt(sigma * sigma+4*(tau.square_norm2()+u_star * u_star)))/2 );
        
        particle.set_u((u_prime + t_vec * AFFPiCS::dot_product(u_prime, t_vec) + 
                        AFFPiCS::cross_product(u_prime, t_vec)                   )/
                            (FLType(1)+t_vec.square_norm2()), info                         );
                
        //particle.move(particle.u(info).element_divide(info.cell_sizes())/particle.gamma(info) * dt/2, info);
        //Half update with the next velocity;
        
        parts[i] = particle;
      }
    };
  }
  
  namespace Pushers
  {
    template <class system_info, indexer num_dims, template <indexer> class ... particles>
    using Vay = SimplePusher<PusherFunctors::Vay, system_info, num_dims, particles...>;
  }
}

#endif