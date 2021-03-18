#ifndef AFFPICS_PUSHERS_BORIS
#define AFFPICS_PUSHERS_BORIS


/*!
  \file Boris.h
  
  \brief The Boris Pusher.
  
  \author Nuno Fernandes
*/

#include "../header.h"
#include "simple_pusher.h"

namespace AFFPiCS
{

  namespace PusherFunctors
  {
    struct Boris
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
                
        using namespace std;
        
        const auto t_vec = info.B_gather(B_fields, particle)
                                * q_dt_m_factor/sqrt( FLType(1) + u_minus.square_norm2()/
                                                      info.units().c()/info.units().c()    );
                
        const auto u_plus = u_minus + AFFPiCS::cross_product(u_minus + AFFPiCS::cross_product(u_minus, t_vec),
                                                                   2*t_vec/(1+t_vec.square_norm2())                 );
                
        particle.set_u(u_plus + info.E_gather(E_fields, particle) * q_dt_m_factor, info );
        
        //particle.move(particle.u(info).element_divide(info.cell_sizes())/particle.gamma(info) * dt/2, info);
        //Half update with the next velocity;
        
        parts[i] = particle;
      }
    };
  }
  
  namespace Pushers
  {
    template <class system_info, indexer num_dims, template <indexer> class ... particles>
    using Boris = SimplePusher<PusherFunctors::Boris, system_info, num_dims, particles...>;
  }
}

#endif