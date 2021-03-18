#ifndef AFFPICS_DIAGNOSTIC_HANDLER
#define AFFPICS_DIAGNOSTIC_HANDLER

/*!
  \file diagnostic_handler.h
  
  \brief Helps dealing with diagnostic information output during the simulation.
  
  \author Nuno Fernandes
*/

#include "../header.h"

namespace AFFPiCS
{
  /*!
    \brief Just checks a (possible) diagnostics class
           for the relevant static member functions.
           No runtime functionality need.
  */
  template <class diagnostic>
  class diagnostic_handler
  {
    private:
    
    //These check for the existence of the specified static functions.
    //(No macro magic involved, this just serves to speed up the writing
    //of the relevant SFINAE pattern - see g24_lib/preliminary_macros.h).

    G24_LIB_FUNC_CHECKER(pre_step);
    G24_LIB_FUNC_CHECKER(post_step);
    
    G24_LIB_FUNC_CHECKER(before_pusher);
    G24_LIB_FUNC_CHECKER(before_evolver);
    G24_LIB_FUNC_CHECKER(before_depositer);
    G24_LIB_FUNC_CHECKER(before_mover);
    
    G24_LIB_FUNC_CHECKER(after_pusher);
    G24_LIB_FUNC_CHECKER(after_evolver);
    G24_LIB_FUNC_CHECKER(after_depositer);
    G24_LIB_FUNC_CHECKER(after_mover);
    
    public:
    
    static constexpr bool pre_step = pre_step_f_exists<diagnostic>;
    static constexpr bool post_step = post_step_f_exists<diagnostic>;
    
    static constexpr bool before_pusher = before_pusher_f_exists<diagnostic>;
    static constexpr bool before_evolver = before_evolver_f_exists<diagnostic>;
    static constexpr bool before_depositer = before_depositer_f_exists<diagnostic>;
    static constexpr bool before_mover = before_mover_f_exists<diagnostic>;
    
    static constexpr bool after_pusher = after_pusher_f_exists<diagnostic>;
    static constexpr bool after_evolver = after_evolver_f_exists<diagnostic>;
    static constexpr bool after_depositer = after_depositer_f_exists<diagnostic>;
    static constexpr bool after_mover = after_mover_f_exists<diagnostic>;
    
    
  };
}

#endif