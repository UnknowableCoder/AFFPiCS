#ifndef AFFPICS_DIFFSOLVE
#define AFFPICS_DIFFSOLVE

/*!
  \file diffsolve.h
  
  \brief Differential equation solver (currently, just Poisson and/or Laplace).
  
  \author Nuno Fernandes
*/

#include "header.h"

namespace AFFPiCS
{

  struct diffsolve_results
  //Could be implemented with an std::tuple or similar,
  //but this seems more expressive and expansible.
  {
    FLType error;
    indexer iterations;
  };
    
  /*!
    \brief A differential equation solver.

    \tparam parallelism_type A valid type of parallelism, that is, one that satisfies `g24_lib::is_parallelism`,
                             with which to perform the computations.
  */
  template<class parallelism_type = g24_lib::Parallelism::OpenMP>
  class diffsolve
  {
    public:
          
      using solver = g24_lib::NonMatrixLinearSolvers::KrylovMethods::ConjugateGradient<parallelism_type>;
      
      template <class Obj> using storage = typename solver::template storage<Obj>;
    
    
      using parallel_type = typename solver::parallel_type;
    
    private:
    
    
      struct matrix_functor
      {
        template <class Vec1, class Vec2, class NDView, class Cell_Sizes>
        CUDA_HOS_DEV inline auto operator() ( const Vec1& wanted, const indexer i, const Vec2& other_side,
                                              const NDView &view, const Cell_Sizes & cellsize,
                                              const indexer accuracy) const
        {
          auto ret = g24_lib::laplacian(wanted, i, view, cellsize, accuracy);
          return ret;
        }
      };
    
    public:
    
      template <class Obj, class BoolArr = g24_lib::bool_vector_parallel<parallel_type, indexer>,
                class NDView = g24_lib::ndview<indexer>, class Cell_Sizes = g24_lib::fspoint<FLType>>
      static void resize_temporaries(const indexer size, storage<Obj> &store)
      {
      
        solver::template resize_temporaries
            <matrix_functor, Obj, Obj, BoolArr, NDView, Cell_Sizes, indexer> (size, store);
      }
      
      template<class ValArr, class FunArr, class BoolArr, class NDView, class Cell_Sizes>
      static auto solve_poisson(storage<ValArr> &store,
                                //The array that stores the temporaries needed for the method.
                                ValArr &wanted,
                                //The array that discretizes the function to be found
                                const FunArr& other_side,
                                //The array that discretizes the other side of the Poisson equation
                                const BoolArr& unchange,
                                //The points whose value is given as a boundary condition 
                                const NDView &view,
                                //The ndview that specifies the dimensions of the arrays
                                const Cell_Sizes &cellsize,
                                //The size of each array cell
                                const indexer max_iter = AFFPiCS::Defaults::max_iterations,
                                const FLType precision = AFFPiCS::Defaults::precision,
                                const indexer accuracy = AFFPiCS::Defaults::derivative_accuracy )
      {
        FLType delta = 2*precision;
        
        indexer count = 0;
        
#if DEBUG_LEVEL > 1
        std::cout << "Poisson Equation Solver:\n"
          "Precision threshold: " << precision <<
          "\nMaximum iterations: " << max_iter << std::endl;
#endif

        solver::pre_iteration (store, matrix_functor{}, wanted,
                                other_side, unchange, view, cellsize, accuracy);
                
        while(count < max_iter)
          //While we're within the maximum number of iterations...
          {
            auto res = lineq_solver::iterate(store, matrix_functor{}, wanted, other_side, unchange, view, cellsize, accuracy);
            ++count;
#if DEBUG_LEVEL > 1
            std::cout << count << ": error = " << res.error
                      << " (" << res.error/res.average*100 << "%)"
                      << " [average is " << res.average << "] " << std::endl;
#endif
            using namespace std;
            if(abs(res.error/res.average) < precision)
            //If the residual is already small enough
            {
              delta = fabs(res.error/res.average);
              break;
            }
          }
        diffsolve_results ret;
        ret.error = delta;
        ret.iterations = count;
        return ret;
      }
      
      template<class ValArr, class BoolArr, class NDView, class Cell_Sizes>
      static auto solve_laplace(storage<ValArr> &store,
                                //The array that stores the temporaries needed for the method.
                                ValArr &wanted,
                                //The array that discretizes the function to be found
                                const BoolArr& unchange,
                                //The points whose value is given as a boundary condition 
                                const NDView &view,
                                //The ndview that specifies the 3d nature of the arrays
                                const Cell_Sizes &cellsize,
                                //The size of each array cell
                                const indexer max_iter = AFFPiCS::Defaults::max_iterations,
                                const FLType precision = AFFPiCS::Defaults::precision,
                                const indexer accuracy = AFFPiCS::Defaults::derivative_accuracy )
      {
        g24_lib::fixed_return_array<typename g24_lib::value_type<ValArr>> dummy(wanted.size(), typename g24_lib::value_type<ValArr>(0.0));
        return solve_poisson(store, wanted, dummy, unchange, view, cellsize, max_iter, precision, accuracy);
      }
      
      template<class ValArr, class FunArr, class BoolArr, class NDView, class Cell_Sizes>
      static auto solve_poisson(ValArr &wanted,
                                //The array that discretizes the function to be found
                                const FunArr& other_side,
                                //The array that discretizes the other side of the Poisson equation
                                const BoolArr& unchange,
                                //The points whose value is given as a boundary condition 
                                const NDView &view,
                                //The ndview that specifies the 3d nature of the arrays
                                const Cell_Sizes &cellsize,
                                //The size of each array cell
                                const indexer max_iter = AFFPiCS::Defaults::max_iterations,
                                const FLType precision = AFFPiCS::Defaults::precision,
                                const indexer accuracy = AFFPiCS::Defaults::derivative_accuracy )
      {
        storage<ValArr> store;
        resize_temporaries(wanted.size(), store);
        return solve_poisson(store, wanted, other_side, unchange, view, cellsize, max_iter, precision, accuracy);
        
      }
      
      template<class ValArr, class BoolArr, class NDView, class Cell_Sizes>
      static auto solve_laplace(ValArr &wanted,
                                //The array that discretizes the function to be found
                                const BoolArr& unchange,
                                //The points whose value is given as a boundary condition 
                                const NDView &view,
                                //The ndview that specifies the 3d nature of the arrays
                                const Cell_Sizes &cellsize,
                                //The size of each array cell
                                const indexer max_iter = AFFPiCS::Defaults::max_iterations,
                                const FLType precision = AFFPiCS::Defaults::precision,
                                const indexer accuracy = AFFPiCS::Defaults::derivative_accuracy )
      {
        storage<ValArr> store;
        resize_temporaries(wanted.size(), store);
        return solve_laplace(store, wanted, unchange, view, cellsize, max_iter, precision, accuracy);
      }
      
    private:
      struct gradient_array_functor
      {
        template <class retT, class ArrT, class BoolArr, class NDView, class Cell_Sizes, class Type>
        CUDA_HOS_DEV inline auto operator() (retT &output,
                                             //The array to which we want to write the result
                                             const indexer i,
                                             //The index of the loop
                                             const ArrT &arr,
                                             //The array upon which we want to operate
                                             const BoolArr & unchange,
                                             //This indicates which points should be the default value
                                             const vector_type<Type, NDView::dimensions()>& unspecified,
                                             //The default value
                                             const NDView &view,
                                             //The ndview that specifies the 3d nature of the arrays
                                             const Cell_Sizes &cellsize,
                                             //The discretization dimensions
                                             const FLType factor = 1.0,
                                             //A multiplying factor for each term (including the unspecified ones)
                                             const indexer accuracy = AFFPiCS::Defaults::derivative_accuracy
                                             //The accuracy of the derivative
                                             )
        {
          if(unchange[i])
            {
              output[i] = unspecified * factor;
            }
          else
            {
              output[i] = g24_lib::gradient(arr, i, view, cellsize, accuracy) * factor;
            }
        }
      };
      
      struct curl_array_functor
      {
        template <class retT, class ArrT, class BoolArr, class NDView, class Cell_Sizes, class Type>
        CUDA_HOS_DEV inline auto operator() (retT &output,
                                             //The array to which we want to write the result
                                             const indexer i,
                                             //The index of the loop
                                             const ArrT &arr,
                                             //The array upon which we want to operate
                                             const BoolArr & unchange,
                                             //This indicates which points should be the default value
                                             const Type unspecified,
                                             //The default value
                                             const NDView &view,
                                             //The ndview that specifies the 3d nature of the arrays
                                             const Cell_Sizes &cellsize,
                                             //The discretization dimensions
                                             const FLType factor = 1.0,
                                             //A multiplying factor for each term (including the unspecified ones)
                                             const indexer accuracy = AFFPiCS::Defaults::derivative_accuracy
                                             //The accuracy of the derivative
                                             )
        {
          if(unchange[i])
            {
              output[i] = unspecified * factor;
            }
          else
            {
              output[i] = g24_lib::curl(arr, i, view, cellsize, accuracy) * factor;
            }
        }
      };
      
    public:

      template <class retT, class ArrT, class BoolArr, class NDView, class Cell_Sizes, class Type>
      //We assume Type is at least convertible to the result of ArrT::operator[].
      static void gradient_array(retT &output,
                                 //The array to which we want to write the result
                                 const ArrT &arr,
                                 //The array upon which we want to operate
                                 const BoolArr & unchange,
                                 //This indicates which points should be the default value
                                 const vector_type<Type, NDView::dimensions()>& unspecified,
                                 //The default value
                                 const NDView &view,
                                 //The ndview that specifies the 3d nature of the arrays
                                 const Cell_Sizes &cellsize,
                                 //The discretization dimensions
                                 const FLType factor = 1.0,
                                 //A multiplying factor for each term (including the unspecified ones)
                                 const indexer accuracy = AFFPiCS::Defaults::derivative_accuracy
                                 //The accuracy of the derivative
                                 )
      {
        solver::parallel_type::loop(output, gradient_array_functor{},
                                    arr, unchange, unspecified, view, cellsize, factor, accuracy);
      }

      template <class retT, class ArrT, class BoolArr, class NDView, class Cell_Sizes, class Type> 
      //We assume Type is at least convertible to the result of retT::operator[].
      static void curl_array(retT &output,
                             //The array to which we want to write the result
                             const ArrT &arr,
                             //The array upon which we want to operate
                             const BoolArr & unchange,
                             //This indicates which points should be the default value
                             const Type& unspecified,
                             //The default value
                             const NDView &view,
                             //The ndview that specifies the 3d nature of the arrays
                             const Cell_Sizes &cellsize,
                             //The discretization dimensions
                             const FLType factor = 1.0,
                             //A multiplying factor for each term (including the unspecified ones)
                             const indexer accuracy = AFFPiCS::Defaults::derivative_accuracy
                             //The accuracy of the derivative
                             )
      {
        solver::parallel_type::loop(output, curl_array_functor{},
                                    arr, unchange, unspecified, view, cellsize, factor, accuracy);
      }

  };
  
}  

#endif
