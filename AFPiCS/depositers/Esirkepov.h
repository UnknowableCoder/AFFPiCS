#ifndef AFFPICS_DEPOSITERS_ESIRKEPOV
#define AFFPICS_DEPOSITERS_ESIRKEPOV

/*!
  \file Esirkepov.h
  
  \brief Esirkepov charge deposition algorithm.
  
  \author Nuno Fernandes
*/

#include "../header.h"
#include "../utilities/particle_storage.h"

namespace AFFPiCS
{
  namespace Depositers
  {
    template <class system_info, indexer num_dims, template <indexer> class ... particles>
    class Esirkepov
    {
      
      template <class parallelism>
      struct calc_W_functor
      {
        private:
        
        struct neighbour_functor
        {        
          template <class particle, class TempArr, class S_Info>
          CUDA_HOS_DEV void operator() (const indexer idx,
                                        const vector_type<indexer, num_dims> &this_cell,
                                        const vector_type<bool, num_dims> &mirrored,
                                        const particle& part,
                                        TempArr& temp_W,
                                        const FLType dt,
                                        const S_Info &I) const
          {
            const vector_type<FLType, num_dims> mirror_sign = vector_type<FLType, num_dims>(1) - vector_type<FLType, num_dims>(mirrored) * 2;
            //So mirror_sign is 1 in the dimensions where mirrored is false
            //and -1 where it is true.
            
            const vector_type<FLType, num_dims>
                      flux_factor = part.charge(I) * part.vel(I).element_multiply(I.cell_sizes()).element_multiply(mirror_sign);
            //A factor of q v that appears
            //(Note the sign from the velocity part because of mirrored.)
            
            const vector_type<FLType, num_dims> p_i = part.pos(I).element_multiply(mirror_sign) +
                                                      vector_type<FLType, num_dims>(part.cell(I)) -
                                                      vector_type<FLType, num_dims>(this_cell) +
                                                      vector_type<FLType, num_dims>(mirrored);
            //The initial position in relation to this cell.
            //When it is mirrored in a given direction,
            //pos[dim] must become 1 - pos[dim].
            //We element_multiply by mirror_sign to get -pos[dim] in the right places
            //and add mirrored since it's 1 in the mirrored dimensions and 0 otherwise.
            
            const vector_type<FLType, num_dims> dp = part.vel(I).element_multiply(mirror_sign) * dt;
            //The variation in position.
            //(Note the sign from the velocity part because of mirrored.)
            
            auto S = &S_Info::template particle_fraction<particle>;
            //To shorten the notation.
            
            if constexpr (num_dims == 1)
              {
                const FLType W_x = (I.*S)(part, p_i + dp) - (I.*S)(part, p_i);
                //S(x+dx) - S(x)
                parallelism::atomics::add(temp_W[idx][0], flux_factor[0] * W_x);
              }
            else if constexpr (num_dims == 2)
              {
                const FLType W_general = ((I.*S)(part, p_i + dp) - (I.*S)(part, p_i))/2;
                //S(x+dx, y+dy)/2 - S(x, y)/2
                const FLType W_x = W_general + ((I.*S)(part, p_i + dp.set(1, 0)) - (I.*S)(part, p_i + dp.set(0, 0)))/2;
                //W_x = S(x+dx, y+dy)/2 + S(x+dx, y)/2 - S(x, y+dy)/2 - S(x, y)/2
                const FLType W_y = W_general + ((I.*S)(part, p_i + dp.set(0, 0)) - (I.*S)(part, p_i + dp.set(1, 0)))/2;
                //W_y = S(x+dx, y+dy) + S(x, y+dy) - S(x+dx, y) - S(x, y)
                parallelism::atomics::add(temp_W[idx][0], flux_factor[0] * W_x);
                parallelism::atomics::add(temp_W[idx][1], flux_factor[1] * W_y);
              }
            else if constexpr (num_dims == 3)
              {
                const FLType W_general = ( 2 * (I.*S)(part, p_i + dp) + (I.*S)(part, p_i + dp.set(0, 0)) + 
                                           (I.*S)(part, p_i + dp.set(1, 0)) + (I.*S)(part, p_i + dp.set(2, 0)) -
                                           (I.*S)(part, p_i + dp.set(0, 0).set(1, 0)) - 
                                           (I.*S)(part, p_i + dp.set(0, 0).set(2, 0)) - 
                                           (I.*S)(part, p_i + dp.set(1, 0).set(2, 0)) -
                                           2 * (I.*S)(part, p_i)                                              )/6;
                //If one carefully compares the several factors for W_x, W_y and W_z,
                //these are the ones that are common to the several dimensions.
                
                const FLType W_x = W_general + ((I.*S)(part, p_i + dp.set(1, 0).set(2, 0)) - (I.*S)(part, p_i + dp.set(0, 0)))/2;
                
                const FLType W_y = W_general + ((I.*S)(part, p_i + dp.set(0, 0).set(2, 0)) - (I.*S)(part, p_i + dp.set(1, 0)))/2;
                
                const FLType W_z = W_general + ((I.*S)(part, p_i + dp.set(0, 0).set(1, 0)) - (I.*S)(part, p_i + dp.set(2, 0)))/2;
                
                parallelism::atomics::add(temp_W[idx][0], flux_factor[0] * W_x);
                parallelism::atomics::add(temp_W[idx][1], flux_factor[1] * W_y);
                parallelism::atomics::add(temp_W[idx][2], flux_factor[2] * W_z);
              }
            else
              {
                //We could try to generalize the previous patterns,
                //but it'd be too much work for no gain at all since
                //we are dealing at most with 3d space...
              
                static_assert(num_dims < 3, "Esirkepov method unspecified for dimensions greater than 3!");
                //And the other parts wouldn't work anyway given the need for cross products.
              }
            
          }
        };
        
        template <class particle, class TempArr, class S_Info>
        CUDA_HOS_DEV void deposit_interior(const particle& part,
                                           TempArr & temp_W,
                                           const FLType dt,
                                           const S_Info &info       ) const
        {
          info.template for_all_neighbours<true>( info.template particle_cell_radius<particle>(part) + 1,
                                                   part.cell(info), neighbour_functor{},
                                                   part, temp_W, dt, info );
        }
        
        template <class particle, class TempArr, class S_Info>
        CUDA_HOS_DEV void deposit_border(const particle& part,
                                           TempArr & temp_W,
                                           const FLType dt,
                                           const S_Info &info       ) const
        {
          //Border conditions will always be tricky and entail branching...
                    
          FLType time_to_border = -1;
          
          for (indexer j = 0; j < info.dimensions(); ++j)
            {
              FLType new_time = -1;
              if (part.vel(info)[j] > 0)
                {
                  new_time = (info.num_cells(j) - part.cell(info)[j] - part.pos(info)[j]) / part.vel(info)[j];
                }
              else if (part.vel(info)[j] < 0)
                {
                  new_time = (part.cell(info)[j] + part.pos(info)[j]) / part.vel(info)[j];
                }
                
              if (new_time >= 0 && ( new_time < time_to_border || time_to_border < 0 ) )
                {
                  time_to_border = new_time;
                }
            }
          
          {
            using namespace std;
            
            time_to_border = nextafter(time_to_border, 2*time_to_border);
            //We increase time_to_border just slightly so that we can be sure to cross the boundary.
          }
          
          if (time_to_border >= dt)
          //In this case, the particle never reaches the border
          //and thus we just need to compute W as usual and don't worry about anything else.
            {
              info.template for_all_neighbours<true>( info.template particle_cell_radius<particle>(part) + 1,
                                                       part.cell(info), neighbour_functor{},
                                                       part, temp_W, dt, info );
            }
          else
            {
              //First, we accumulate W for the particle to travel towards the boundary
              info.template for_all_neighbours<true>( info.template particle_cell_radius<particle>(part) + 1,
                                                       part.cell(info), neighbour_functor{},
                                                       part, temp_W, time_to_border, info );
              particle temp = part;
              temp.set_pos(temp.pos(info) + time_to_border * temp.vel(info), info);
              
              info.boundary_particles(temp, true);
              
              //As a good approximation, we disregard the possibility
              //that the particle, after being treated according to the boundary conditions,
              //will reach the border again this timestep.
              info.template for_all_neighbours<true>( info.template particle_cell_radius<particle>(part) + 1,
                                                       temp.cell(info), neighbour_functor{},
                                                       temp, temp_W, dt - time_to_border, info );
              
            }
          
        }
        
        public:
        
        template <class PartArr, class TempArr, class S_Info>
        CUDA_HOS_DEV void operator() (const PartArr& parts,
                                      const indexer i,
                                      TempArr& temp_W,
                                      const FLType dt,
                                      const S_Info &info      ) const
        {
          using particle = g24_lib::value_type<PartArr>;
          const indexer radius = info.template particle_cell_radius<particle>(particle{}) + 1;
          
          bool is_border = false;
          
          for (indexer j = 0; j < parts[i].cell(info).size(); ++j)
            {
              if (parts[i].cell(info)[j] < radius || parts[i].cell(info)[j] >= info.num_cells(j) - radius)
                {
                  is_border = true;
                  break;
                }
            }
            
          //Though the Esirkepov method's formulation is essentially branchless,
          //we need to account for possible boundary conditions,
          //hence this check for the cells that are at the borders of the system.
          //
          //Possible (slightly more complex) alternative,
          //possibly better suited to GPU parallelization:
          //first do a loop over all particles that we know for sure are inside,
          //only then do for those outside.
          //This is admitting the particle array is sorted,
          //which would also be an important improvement
          //in the GPU case for memory coalescence.
          //(And even in the CPU case it might bring benefits...)
          
          if (is_border)
            {
              deposit_border(parts[i], temp_W, dt, info);
            }
          else
            {
              deposit_interior(parts[i], temp_W, dt, info);
            }
        }
      };
      
      struct calc_J_functor
      {
        template <class CurrArr, class TempArr, class S_Info>
        CUDA_HOS_DEV void operator() (CurrArr & currents,
                                      const indexer idx,
                                      const TempArr &temp_W,
                                      const FLType dt,
                                      const indexer radius,
                                      const S_Info &info)const
        {
          const auto cell = info.to_cell(idx);
          for (indexer dim = 0; dim < info.dimensions(); ++dim)
            {
              indexer j;
              for (j = -radius; j <= radius && cell[dim] + j < 0; ++j)
                {
                  currents[idx] += g24_lib::sign(j) * info.boundary_J(cell.add(dim, j), temp_W);
                  //Since the currents are linear combinations of W, 
                  //the same boundary conditions (namely if periodic)
                  //should apply.
                }
              for (; j <= radius && cell[dim] + j < info.num_cells(dim); ++j)
                {
                  currents[idx] += g24_lib::sign(j) * temp_W[info.to_index(cell.add(dim, j))];
                }
              for (; j <= radius; ++j)
                {
                  currents[idx] += g24_lib::sign(j) * info.boundary_J(cell.add(dim, j), temp_W);
                }
            }
        }
      };
      
      struct W_J_reset_functor
      {
        template <class W_arr>
        CUDA_HOS_DEV void operator() (W_arr &arr, const indexer i) const
        {
          arr[i].set_all(0);
        }
      };
      
      private:
      template <class parallelism>
      using particle_storage_type = particle_storage<parallelism, particles<num_dims>...>;
      
      public:
      
      template <class parallelism> struct storage
      {
        typename parallelism::kernel_size_type calc_W_kernel[sizeof...(particles)],
                                               calc_J_kernel[sizeof...(particles)],
                                               W_J_reset_kernel;
        
        current_holder<parallelism, num_dims> temp_W;
        
        private:
        
        template <indexer idx, class part, class ... parts>
        void initialize_in(const particle_storage_type<parallelism> &part_store)
        {
          initialize_single<idx, part>(part_store);
          if constexpr (sizeof...(parts) > 0)
            {
              initialize_in<idx + 1, parts...>(part_store);
            }
        }
        
        template <indexer idx, class part>
        void initialize_single(const particle_storage_type<parallelism> &part_store)
        {
          calc_W_kernel[idx] = parallelism::template estimate_loop_kernel_size
                                  < particle_holder<parallelism, part>,
                                    calc_W_functor<parallelism>,
                                    current_holder<parallelism, num_dims>,
                                    FLType, system_info                     >
                                (part_store.template get_particles<part>().size());
                                
          calc_J_kernel[idx] = parallelism::template estimate_loop_kernel_size
                                  < current_holder<parallelism, num_dims>,
                                    calc_J_functor,
                                    current_holder<parallelism, num_dims>,
                                    FLType, indexer, system_info                     >
                                (temp_W.size());
        }
        
        public:
        
        void initialize( const particle_storage_type<parallelism> &part_store,
                         const current_holder<parallelism, num_dims> &currents,
                         const system_info& info                                )
        {
          temp_W.resize(currents.size());
          
          initialize_in<0, particles<num_dims>...>(part_store);
          
          W_J_reset_kernel = parallelism::template estimate_loop_kernel_size
                                    <current_holder<parallelism, num_dims>, W_J_reset_functor>
                                (currents.size());
        }
        
        template <class stream> void save(stream &s, bool binary = Defaults::data_i_o_as_binary) const
        {
          //We do not need to save the temporary because it only holds values
          //that are relevant at mid step.
        }
        
        template <class stream> void load(stream &s, bool binary = Defaults::data_i_o_as_binary)
        {
        }
        
      };
     
      private:
      
      template <indexer idx, class parallelism, class part, class ... parts>
      static void deposit_impl(storage<parallelism> &store,
                               current_holder<parallelism, num_dims> &currents,
                               const particle_storage_type<parallelism>& part_storage,
                               const FLType dt,
                               const system_info& info)
      {
        deposit_impl_single<idx, parallelism, part>(store, currents, part_storage, dt, info);
        if constexpr (sizeof...(parts) > 0)
          {
            deposit_impl<idx + 1, parallelism, parts...>(store, currents, part_storage, dt, info);
          }
      }
      
      template <indexer idx, class parallelism, class part>
      static void deposit_impl_single(storage<parallelism> &store,
                                      current_holder<parallelism, num_dims> &currents,
                                      const particle_storage_type<parallelism>& part_storage,
                                      const FLType dt,
                                      const system_info& info)
      {
        parallelism::loop( store.W_J_reset_kernel, store.temp_W, W_J_reset_functor{});
        
        parallelism::loop( store.calc_W_kernel[idx], part_storage.template get_particles<part>(),
                           calc_W_functor<parallelism>{}, store.temp_W, dt, info                  );
        
        parallelism::loop( store.calc_J_kernel[idx], currents, calc_J_functor{},
                           store.temp_W, dt, info.template particle_cell_radius<part>(part{}) + 1, info );
      }
      
      public:
      
      struct results {};
      //There's no need to return anything from here.
      //Might be useful for debugging, but later...
      
      template <class parallelism>
      static results deposit(storage<parallelism> &store,
                             current_holder<parallelism, num_dims> &currents,
                             const particle_storage_type<parallelism>& part_storage,
                             const FLType dt,
                             const system_info& info)
      {
        parallelism::loop( store.W_J_reset_kernel, currents, W_J_reset_functor{} );
        deposit_impl<0, parallelism, particles<num_dims>...>(store, currents, part_storage, dt, info);
        return results{};
      }
    };
  }
}

#endif