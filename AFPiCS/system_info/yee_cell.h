#ifndef AFFPICS_SYSTEM_INFO_YEE_CELL
#define AFFPICS_SYSTEM_INFO_YEE_CELL

/*!
  \file yee_cell.h
  
  \brief Implements the staggered grid of the Yee cell in the field gathering algorithms.
  
  \author Nuno Fernandes
*/

#include "../header.h"
#include "system_info_base.h"

namespace AFFPiCS
{
  namespace SystemDefinitions
  {
    template <indexer num_dims, class derived = void>
    class YeeMethodGrid
    //Yee cell:
    //
    // 1D:
    //
    //         |               |
    //              B_y,z(x)
    //         |       A       |
    //      --< >-----(X)-----< >--
    //         |       V       |
    //        E_x(x)        E_x(x+dx)
    //         |               |
    //
    //   y  A
    //      |
    //     (X)-------> x
    //     z
    //
    //
    // 2D:
    //            E_x(x,y+dy)
    //          +-----<>-----+
    //          |            |
    //          |   H_z(x,y) |
    //          A     \/     A E_y(x+dx,y)
    // E_y(x,y) V     /\     V 
    //          |            |
    //          |            |
    //          +-----<>-----+
    //             E_x(x,y) 
    //
    //   y  A
    //      |
    //      |
    //     (X)---> x
    //     z
    //
    // 3D:
    //
    //  (1): E_y (x,y,z)
    //  (2): E_x (x,y+dy,z)
    //  (3): E_y (x+dx,y,z)
    //  (4): E_x (x,y,z)
    //  (5): H_z (x,y,z)
    //  (6): E_z (x+dx,y,z)
    //  (7): E_z (x+dx,y+dy,z)
    //  (8): E_z (x,y+dy,z)
    //  (9): E_z (x,y,z)
    // (10): E_x (x,y+dy,z+dz)
    // (11): E_x (x,y,z+dz)
    // (12): H_y (x,y,z)
    // (13): H_x (x+dx,y,z)
    // (14): H_y (x,y+dy,z)
    // (15): H_x (x,y,z)
    // (16): H_z (x,y,z+dz)
    // (17): E_y (x,y,z+dz)
    // (18): E_y (x+dx,y,z+dz)
    // 
    // 
    //                            (17)   
    //                     +-------<>-------+
    //                    /|               /|
    //                   / | (16)/        / |
    //                  T  |  --/--      T(1|0)
    //             (11)L   A   /   \/   L   A
    //                / (9)V       /\  /    V(8)
    //               /     |(18)  (15)/     | 
    //              +-------<>-------+ (14) |
    //              |  \/  |     (1) |  \/  |
    //            (1|2)/\  +-------<>|--/\--+
    //              |     / (13)     |     /
    //              A    / \/    /   A(7) /
    //           (6)V   T  /\ --/--  V   T
    //              |  L       / (5) |  L (2)
    //              | /(4)           | /
    //              |/               |/
    //              +-------<>-------+
    //                      (3)
    //   z  A
    //      |
    //      |
    //      |
    //      +------->  y
    //     /
    //    /
    //   L  x
    //
    //
    //
    //
    // So, all in all, E_i is measured at (0 , ... , 0) + d x_i/2 and B_i at (0.5 , ... , 0.5) - d x_i/2.
    // 
    {
      private:
      
      using deriv_t = std::conditional_t<std::is_void_v<derived>, YeeMethodGrid, derived>;
      
     public:
      
      CUDA_HOS_DEV vector_type<FLType, num_dims> E_measurement(const indexer dim) const
      {
        vector_type<FLType, num_dims> ret(FLType(0));
        ret[dim] = FLType(0.5);
        return ret;
      }
      
      CUDA_HOS_DEV vector_type<FLType, num_dims> B_measurement(const indexer dim) const
      {
        vector_type<FLType, num_dims> ret(FLType(0.5));
        if constexpr (num_dims == 3)
          {
            ret[dim] = 0;
          }
        //In 1 and 2-dimensional simulations, it's simply at the middle.
        return ret;
      }
      
    };
  }
}

#endif