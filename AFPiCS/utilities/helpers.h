#ifndef AFFPICS_HELPERS
#define AFFPICS_HELPERS

#include "../header.h"

namespace AFFPiCS
{
  template <indexer num_dims_1, indexer num_dims_2>
  inline auto cross_product(const vector_type<FLType, num_dims_1> &a, const vector_type<FLType, num_dims_2> &b)
  {
    if constexpr (num_dims_1 == 1 && num_dims_2 == 1)
      {
        return vector_type<FLType, 1> {FLType(0)};
        //One-dimensional simulations don't have cross product?
      }
    else if constexpr (num_dims_1 == 1 && num_dims_2 == 2)
      {
        return vector_type<FLType, 2>{-a[0]*b[1],a[0]*b[0]};
        //2-d in a plane x 1-d in the perpendicular gives a vector in the plane
      }
    else if constexpr (num_dims_1 == 2 && num_dims_2 == 1)
      {
        return vector_type<FLType, 2>{a[1]*b[0], -a[0]*b[0]};
        //2-d in a plane x 1-d in the perpendicular gives a vector in the plane
      }
    else if constexpr (num_dims_1 == 2 && num_dims_2 == 2)
      {
        return vector_type<FLType, 1> {a[0]*b[1]-a[1]*b[0]};
        //2d-cross product gives a vector in the perpendicular direction.
      }
    else
      {
        return a.crossp(b);
        //Valid if both are 3-dimensional, else ERROR!
      }
  }
  
  template <indexer num_dims_1, indexer num_dims_2>
  inline FLType dot_product(const vector_type<FLType, num_dims_1> &a, const vector_type<FLType, num_dims_2> &b)
  {
    if constexpr ((num_dims_1 == 1 && num_dims_2 == 2) || (num_dims_1 == 2 && num_dims_2 == 1))
      {
        return FLType(0);
        //2-d in a plane dot 1-d in the perpendicular gives 0
      }
    else
      {
        return a.dotp(b);
      }
  }
  
  template <indexer dim_out, indexer num_dims, class ArrT, class BoundFunc>
  inline vector_type<FLType, dim_out> curl( const vector_type<indexer, num_dims>& cell,
                                            const ArrT& arr,
                                            const BoundFunc & f,
                                            const vector_type<FLType, num_dims>& cell_size )
  {
    if constexpr (num_dims == 1)
      {
        return vector_type<FLType, dim_out>(FLType(0));
        //If calculating curl E, it will give 0 since there is only x dependency (and no d/dx Ex).
        //If calculating curl B, it will give 0 in the x direction, thus useless.
      }
    else if constexpr (num_dims == 2)
      {
        if constexpr (dim_out == 1)
        //So we are taking the curl of a 2-D function.
        //( The function lives in the plane, 
        //  the curl will live in the normal. )
          {
            vector_type<FLType, dim_out> ret{0};
            
            //(rot F)_z = d F_y / dx - d F_x / dy
            
            //d F_y / dx
            ret[0] += (f(cell.add(0,1), arr)[1]-f(cell.subtract(0,1), arr)[1])/cell_size[0];
            
            //- d E_x / dy
            ret[0] -= (f(cell.add(1,1), arr)[0]-f(cell.subtract(1,1), arr)[0])/cell_size[1];
            
            return ret/2;
          }
        else if constexpr (dim_out == 2)
        //So we are taking the curl of a 1-D function.
        //( The function lives in the normal,
        //  the curl will live in the plane. )
          {
            vector_type<FLType, dim_out> ret;
            
            //(rot F)_x = d F_z / dy
            ret[0] = (f(cell.add(1,1), arr)[0]-f(cell.subtract(1,1), arr)[0])/cell_size[1];
              
            //(rot F)_y = -d F_z / dx
            ret[1] = -(f(cell.add(0,1), arr)[0]-f(cell.subtract(0,1), arr)[0])/cell_size[0];
            
            return ret/2;
          }
      }
    else if constexpr (num_dims == 3)
    //Everything full 3d.
      {
        vector_type<FLType, dim_out> ret{0,0,0};
        
        //(rot F)_x = d F_z / dy - d F_y / dz
        ret[0] += (f(cell.add(1,1), arr)[2]-f(cell.subtract(1,1), arr)[2])/cell_size[1];
        ret[0] -= (f(cell.add(2,1), arr)[1]-f(cell.subtract(2,1), arr)[1])/cell_size[2];
        //(rot F)_y = d F_x / dz - d F_z / dx
        ret[1] += (f(cell.add(2,1), arr)[0]-f(cell.subtract(2,1), arr)[0])/cell_size[2];
        ret[1] -= (f(cell.add(0,1), arr)[2]-f(cell.subtract(0,1), arr)[2])/cell_size[0];
        //(rot F)_z = d F_y / dx - d F_x / dy
        ret[2] += (f(cell.add(0,1), arr)[1]-f(cell.subtract(0,1), arr)[1])/cell_size[0];
        ret[2] -= (f(cell.add(1,1), arr)[0]-f(cell.subtract(1,1), arr)[0])/cell_size[1];
        
        return ret/2;
      }
    else
      {
        static_assert(num_dims <= 3, "Simulations with more than 3 dimensions not supported!");
        return  vector_type<FLType, dim_out>(0);
      }
  }
  
  CUDA_HOS_DEV inline constexpr FLType constexpr_sqrt(const FLType val)
  {
    if (val < 0)
    {
      using namespace std;
      return NAN;
      //To be CUDA-compatible and all.
    }
    else
    {
      FLType prev = 0, curr = val;
      while (curr != prev)
        {
          prev = curr;
          curr = (curr + val/curr)/2;
        }
      return curr;
    }
  }
}


#endif
