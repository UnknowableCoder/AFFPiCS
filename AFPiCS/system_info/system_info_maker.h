#ifndef AFFPICS_SYSTEM_INFO_MAKER
#define AFFPICS_SYSTEM_INFO_MAKER

#include "../header.h"
#include "system_info_base.h"

namespace AFFPiCS
{
  namespace SystemDefinitions
  {
  
    
  
#define AFFPICS_COMMA_TRICK(...) __VA_ARGS__

    
  
    template <indexer num_dims, class end, class base, class ... options> class SystemInfoMaker;
    
#define AFFPICS_SYSINFO_MAKER_FUNC(PRE, RET_TYPE, NAME, ARGUMENTS, POST, ARG_NAMES, ARG_TYPES, EXTRAID) \
private:                                                              \
template<class T, class ... Args_> inline static constexpr            \
auto NAME ## EXTRAID ## _checker(T*) -> decltype(std::declval<T>().NAME(std::declval<Args_>()...)); \
template<class T, class ... Args_> inline static constexpr            \
auto NAME ## EXTRAID ## _checker(T*)                                  \
-> std::enable_if_t<sizeof...(Args_)== 1 &&                           \
                    std::is_void_v<std::common_type_t<Args_...>>,     \
                    decltype(std::declval<T>().NAME())>;              \
template<class T, class ... Args_> inline static constexpr            \
InvalidReturn NAME ## EXTRAID ## _checker(...);                       \
template <class T> inline static constexpr                            \
bool NAME ## EXTRAID ## _exists = std::is_same_v<decltype( NAME ## EXTRAID ## _checker<T, ARG_TYPES>(nullptr) ), RET_TYPE>; \
public:                                                               \
inline static constexpr bool has_ ## NAME ## EXTRAID = NAME ## EXTRAID ## _exists<option> || \
                                                       before::has_ ## NAME ## EXTRAID; \
PRE RET_TYPE NAME ARGUMENTS POST                                      \
{                                                                     \
  if constexpr (before::has_ ## NAME ## EXTRAID)                      \
    {                                                                 \
     return static_cast<POST before*>(this)->NAME(ARG_NAMES);         \
    }                                                                 \
  else if constexpr (NAME ## EXTRAID ## _exists<option>)              \
    {                                                                 \
     return static_cast<POST option*>(this)->NAME(ARG_NAMES);         \
    }                                                                 \
  else                                                                \
    {                                                                 \
     return static_cast<POST base*>(this)->NAME(ARG_NAMES);           \
    }                                                                 \
}                                                                     \

#define AFFPICS_SYSINFO_MAKER_TEMPLATE_FUNC(TEMPLATE, PRE, RET_TYPE, NAME, ARGUMENTS, POST, ARG_NAMES, ARG_TYPES, TEMP_ARGS, EXTRAID) \
private:                                                              \
template TEMPLATE struct NAME ## EXTRAID ## _checker                  \
{                                                                     \
  template<class T, class ... Args_> inline static constexpr          \
  auto checker(T*) -> decltype(std::declval<T>().template NAME<TEMP_ARGS>(std::declval<Args_>()...)); \
  template<class T, class ... Args_> inline static constexpr          \
  InvalidReturn checker(...);                                         \
  template <class T> inline static constexpr                          \
  bool exists = std::is_same_v<decltype(checker<T, ARG_TYPES>(nullptr)), RET_TYPE>;\
};                                                                    \
public:                                                               \
template TEMPLATE                                                     \
inline static constexpr bool has_ ## NAME ## EXTRAID = NAME ## EXTRAID ## _checker<TEMP_ARGS>::template exists<option> || \
                            before::template has_ ## NAME ## EXTRAID <TEMP_ARGS>; \
template TEMPLATE PRE RET_TYPE NAME ARGUMENTS POST                    \
{                                                                     \
  if constexpr (before::template has_ ## NAME ## EXTRAID <TEMP_ARGS>) \
    {                                                                 \
     return static_cast<POST before*>(this)->template NAME <TEMP_ARGS>(ARG_NAMES); \
    }                                                                 \
  else if constexpr (NAME ## EXTRAID ## _checker<TEMP_ARGS>::template exists<option>) \
    {                                                                 \
     return static_cast<POST option*>(this)->template NAME <TEMP_ARGS>(ARG_NAMES); \
    }                                                                 \
  else                                                                \
    {                                                                 \
     return static_cast<POST base*>(this)->template NAME <TEMP_ARGS>(ARG_NAMES); \
    }                                                                 \
}                                                                     \

    template <indexer num_dims, class end, class base, class option, class ... options>
    class SystemInfoMaker<num_dims, end, base, option, options...>:
    public SystemInfoMaker<num_dims, end, base, options...>,
    public option
    {
      using before = SystemInfoMaker<num_dims, end, base, options...>;
      
      struct InvalidReturn {};
      //For SFINAE.
      
      public:
      using SystemInfoMaker<num_dims, end, base, options...>::SystemInfoMaker;
      using option::option;
      
      AFFPICS_SYSINFO_MAKER_FUNC(CUDA_HOS_DEV, FLType, mu, (const indexer i), const, i, const indexer, 1)
     
      AFFPICS_SYSINFO_MAKER_FUNC(CUDA_HOS_DEV, FLType, mu, (const vector_type<indexer, num_dims> &p), const, p, AFFPICS_COMMA_TRICK(const vector_type<indexer, num_dims>), 2)
     
      AFFPICS_SYSINFO_MAKER_FUNC(CUDA_HOS_DEV, FLType, epsilon, (const indexer i), const, i, indexer, 1)
     
      AFFPICS_SYSINFO_MAKER_FUNC(CUDA_HOS_DEV, FLType, epsilon, (const vector_type<indexer, num_dims> &p), const, p, AFFPICS_COMMA_TRICK(const vector_type<indexer, num_dims>), 2)
     
      AFFPICS_SYSINFO_MAKER_FUNC(CUDA_HOS_DEV, const UnitSystem&, units, (), const, , void , 1)
     
      AFFPICS_SYSINFO_MAKER_FUNC(CUDA_HOS_DEV constexpr, indexer, dimensions, (), const, , void ,1)
     
      AFFPICS_SYSINFO_MAKER_FUNC(CUDA_HOS_DEV, AFFPICS_COMMA_TRICK(vector_type<indexer, num_dims>), num_cells, (), const, , void, 1)
     
      AFFPICS_SYSINFO_MAKER_FUNC(CUDA_HOS_DEV, indexer, num_cells, (const indexer i), const, i, const indexer, 2)
     
      AFFPICS_SYSINFO_MAKER_FUNC(CUDA_HOS_DEV, indexer, total_cells, (), const, , void, 1)
     
      AFFPICS_SYSINFO_MAKER_FUNC(CUDA_HOS_DEV, AFFPICS_COMMA_TRICK(vector_type<indexer, num_dims>), to_cell, (const indexer i), const, i, const indexer, 1)
     
      AFFPICS_SYSINFO_MAKER_FUNC(CUDA_HOS_DEV, indexer, to_index, (const vector_type<indexer, num_dims> &cell), const, cell, AFFPICS_COMMA_TRICK(const vector_type<indexer, num_dims> &), 1)
     
      AFFPICS_SYSINFO_MAKER_TEMPLATE_FUNC(
          <class E_holder>,
          CUDA_HOS_DEV,
          B_field_type<num_dims>,
          E_curl,
          (const E_holder &E_fields, const indexer i),
          const,
          AFFPICS_COMMA_TRICK(E_fields, i),
          AFFPICS_COMMA_TRICK(const E_holder&, const indexer),
          E_holder,
          1)
          
      AFFPICS_SYSINFO_MAKER_TEMPLATE_FUNC(
          <class E_holder>,
          CUDA_HOS_DEV,
          B_field_type<num_dims>,
          E_curl,
          (const E_holder &E_fields, const vector_type<indexer, num_dims> &p),
          const,
          AFFPICS_COMMA_TRICK(E_fields, p),
          AFFPICS_COMMA_TRICK(const E_holder&, const vector_type<indexer, num_dims>),
          E_holder,
          2)
          
      AFFPICS_SYSINFO_MAKER_TEMPLATE_FUNC(
          <class B_holder>,
          CUDA_HOS_DEV,
          E_field_type<num_dims>,
          B_curl,
          (const B_holder &B_fields, const indexer i),
          const,
          AFFPICS_COMMA_TRICK(B_fields, i),
          AFFPICS_COMMA_TRICK(const B_holder&, const indexer),
          B_holder,
          1)
          
      AFFPICS_SYSINFO_MAKER_TEMPLATE_FUNC(
          <class B_holder>,
          CUDA_HOS_DEV,
          E_field_type<num_dims>,
          B_curl,
          (const B_holder &B_fields, const vector_type<indexer, num_dims> &p),
          const,
          AFFPICS_COMMA_TRICK(B_fields, p),
          AFFPICS_COMMA_TRICK(const B_holder&, const vector_type<indexer, num_dims>&),
          B_holder,
          2)
          
      AFFPICS_SYSINFO_MAKER_TEMPLATE_FUNC(
          AFFPICS_COMMA_TRICK(<class ArrT, class part>),
          CUDA_HOS_DEV,
          E_field_type<num_dims>,
          E_gather,
          (const ArrT &E_arr, const part& particle),
          const,
          AFFPICS_COMMA_TRICK(E_arr, particle),
          AFFPICS_COMMA_TRICK(const ArrT&, const part&),
          AFFPICS_COMMA_TRICK(ArrT, part),
          1)
          
      AFFPICS_SYSINFO_MAKER_TEMPLATE_FUNC(
          AFFPICS_COMMA_TRICK(<class ArrT, class part>),
          CUDA_HOS_DEV,
          B_field_type<num_dims>,
          B_gather,
          (const ArrT &B_arr, const part& particle),
          const,
          AFFPICS_COMMA_TRICK(B_arr, particle),
          AFFPICS_COMMA_TRICK(const ArrT&, const part&),
          AFFPICS_COMMA_TRICK(ArrT, part),
          1)
          
      AFFPICS_SYSINFO_MAKER_FUNC(CUDA_HOS_DEV, AFFPICS_COMMA_TRICK(vector_type<FLType, num_dims>), cell_sizes, (), const, , void, 1)
      
      AFFPICS_SYSINFO_MAKER_FUNC(CUDA_HOS_DEV, bool, is_border, (const indexer i), const, i, const indexer, 1)
     
      AFFPICS_SYSINFO_MAKER_FUNC(CUDA_HOS_DEV, bool, is_border, (const vector_type<indexer, num_dims> &p), const, p, AFFPICS_COMMA_TRICK(const vector_type<indexer, num_dims>&), 2)
      
      AFFPICS_SYSINFO_MAKER_FUNC(CUDA_HOS_DEV, bool, is_outside, (const indexer i), const, i, const indexer, 1)
     
      AFFPICS_SYSINFO_MAKER_FUNC(CUDA_HOS_DEV, bool, is_outside, (const vector_type<indexer, num_dims> &p), const, p, AFFPICS_COMMA_TRICK(const vector_type<indexer, num_dims>&), 2)
      
      AFFPICS_SYSINFO_MAKER_TEMPLATE_FUNC(
          <class part_type>,
          CUDA_HOS_DEV constexpr,
          indexer,
          particle_cell_radius,
          (const part_type& part),
          const,
          part,
          const part_type&,
          part_type,
          1)

      AFFPICS_SYSINFO_MAKER_TEMPLATE_FUNC(
          <class part_type>,
          CUDA_HOS_DEV,
          FLType,
          particle_fraction,
          (const part_type& part, const vector_type<FLType, num_dims> &pos),
          const,
          AFFPICS_COMMA_TRICK(part, pos),
          AFFPICS_COMMA_TRICK(const part_type&, const vector_type<FLType, num_dims>&),
          part_type,
          1)
          
      AFFPICS_SYSINFO_MAKER_FUNC(CUDA_HOS_DEV, AFFPICS_COMMA_TRICK(vector_type<FLType, num_dims>), E_measurement, (const indexer dim), const, dim, const indexer, 1)
      
      AFFPICS_SYSINFO_MAKER_FUNC(CUDA_HOS_DEV, AFFPICS_COMMA_TRICK(vector_type<FLType, num_dims>), B_measurement, (const indexer dim), const, dim, const indexer, 1)
      
      AFFPICS_SYSINFO_MAKER_TEMPLATE_FUNC(
          AFFPICS_COMMA_TRICK(<bool check_for_border, class Func, class ... Args>),
          CUDA_HOS_DEV,
          void,
          for_all_neighbours,
          (const indexer radius, const indexer index, Func && func, Args&& ... args),
          const,
          AFFPICS_COMMA_TRICK(radius, index, std::forward<Func>(func), std::forward<Args>(args)...),
          AFFPICS_COMMA_TRICK(const indexer, const indexer, Func, Args...),
          AFFPICS_COMMA_TRICK(check_for_border, Func, Args...),
          1)
      
      AFFPICS_SYSINFO_MAKER_TEMPLATE_FUNC(
          AFFPICS_COMMA_TRICK(<bool check_for_border, class Func, class ... Args>),
          CUDA_HOS_DEV,
          void,
          for_all_neighbours,
          (const indexer radius, const vector_type<indexer, num_dims> &cell, Func && func, Args&& ... args),
          const,
          AFFPICS_COMMA_TRICK(radius, cell, std::forward<Func>(func), std::forward<Args>(args)...),
          AFFPICS_COMMA_TRICK(const indexer, const vector_type<indexer, num_dims>&, Func, Args...),
          AFFPICS_COMMA_TRICK(check_for_border, Func, Args...),
          2)
          
      AFFPICS_SYSINFO_MAKER_TEMPLATE_FUNC(
          <class particle>,
          CUDA_HOS_DEV,
          void,
          boundary_particles,
          (particle& part, const bool force_apply = false),
          const,
          AFFPICS_COMMA_TRICK(part, force_apply),
          AFFPICS_COMMA_TRICK(particle&, const bool),
          particle,
          1)
          
      AFFPICS_SYSINFO_MAKER_TEMPLATE_FUNC(
          <class ArrT>,
          CUDA_HOS_DEV,
          E_field_type<num_dims>,
          boundary_E,
          (const vector_type<indexer, num_dims> &cell, const ArrT & E_fields),
          const,
          AFFPICS_COMMA_TRICK(cell, E_fields),
          AFFPICS_COMMA_TRICK(const vector_type<indexer, num_dims>&, const ArrT&),
          ArrT,
          1)
          
      AFFPICS_SYSINFO_MAKER_TEMPLATE_FUNC(
          <class ArrT>,
          CUDA_HOS_DEV,
          B_field_type<num_dims>,
          boundary_B,
          (const vector_type<indexer, num_dims> &cell, const ArrT & B_fields),
          const,
          AFFPICS_COMMA_TRICK(cell, B_fields),
          AFFPICS_COMMA_TRICK(const vector_type<indexer, num_dims>&, const ArrT&),
          ArrT,
          1)
      
      AFFPICS_SYSINFO_MAKER_TEMPLATE_FUNC(
          <class ArrT>,
          CUDA_HOS_DEV,
          current_type<num_dims>,
          boundary_J,
          (const vector_type<indexer, num_dims> &cell, const ArrT & currents),
          const,
          AFFPICS_COMMA_TRICK(cell, currents),
          AFFPICS_COMMA_TRICK(const vector_type<indexer, num_dims>&, const ArrT&),
          ArrT,
          1)
          
      AFFPICS_SYSINFO_MAKER_TEMPLATE_FUNC(
          AFFPICS_COMMA_TRICK(<class EArr, class BArr, class JArr, class PartStorage>),
          CUDA_HOS_DEV,
          void,
          initial_condition,
          (EArr& E_fields, BArr & B_fields, JArr & currents, PartStorage& particles),
          const,
          AFFPICS_COMMA_TRICK(E_fields, B_fields, currents, particles),
          AFFPICS_COMMA_TRICK(EArr&, BArr&, JArr&, PartStorage&),
          AFFPICS_COMMA_TRICK(EArr, BArr, JArr, PartStorage),
          1)
    };
    
#undef AFFPICS_SYSINFO_MAKER_FUNC
#undef AFFPICS_SYSINFO_MAKER_TEMPLATE_FUNC

#define AFFPICS_SYSINFO_MAKER_FUNC(PRE, RET_TYPE, NAME, ARGUMENTS, POST, ARG_NAMES, ARG_TYPES, EXTRAID) \
private:                                                              \
template<class T, class ... Args_> inline static constexpr            \
auto NAME ## EXTRAID ## _checker(T*) -> decltype(std::declval<T>().NAME(std::declval<Args_>()...)); \
template<class T, class ... Args_> inline static constexpr            \
auto NAME ## EXTRAID ## _checker(T*)                                  \
-> std::enable_if_t<sizeof...(Args_)== 1 &&                           \
                    std::is_void_v<std::common_type_t<Args_...>>,     \
                    decltype(std::declval<T>().NAME())>;              \
template<class T, class ... Args_> inline static constexpr            \
InvalidReturn NAME ## EXTRAID ## _checker(...);                       \
template <class T> inline static constexpr                            \
bool NAME ## EXTRAID ## _exists = std::is_same_v<decltype( NAME ## EXTRAID ## _checker<T, ARG_TYPES>(nullptr) ), RET_TYPE>; \
public:                                                               \
inline static constexpr bool has_ ## NAME ## EXTRAID = NAME ## EXTRAID ## _exists<option>; \
PRE RET_TYPE NAME ARGUMENTS POST                                      \
{                                                                     \
  if constexpr (NAME ## EXTRAID ## _exists<option>)                   \
    {                                                                 \
     return static_cast<POST option*>(this)->NAME(ARG_NAMES);         \
    }                                                                 \
  else                                                                \
    {                                                                 \
     return static_cast<POST base*>(this)->NAME(ARG_NAMES);           \
    }                                                                 \
}                                                                     \

#define AFFPICS_SYSINFO_MAKER_TEMPLATE_FUNC(TEMPLATE, PRE, RET_TYPE, NAME, ARGUMENTS, POST, ARG_NAMES, ARG_TYPES, TEMP_ARGS, EXTRAID) \
private:                                                              \
template TEMPLATE struct NAME ## EXTRAID ## _checker                  \
{                                                                     \
  template<class T, class ... Args_> inline static constexpr          \
  auto checker(T*) -> decltype(std::declval<T>.template NAME<TEMP_ARGS>(std::declval<Args_>()...)); \
  template<class T, class ... Args_> inline static constexpr          \
  InvalidReturn checker(...);                                         \
  template <class T> inline static constexpr                          \
  bool exists = std::is_same_v<decltype(checker<T, ARG_TYPES>(nullptr)), RET_TYPE>;\
};                                                                    \
public:                                                               \
template TEMPLATE                                                     \
inline static constexpr bool has_ ## NAME ## EXTRAID = NAME ## EXTRAID ## _checker<TEMP_ARGS>::template exists<option>; \
template TEMPLATE PRE RET_TYPE NAME ARGUMENTS POST                    \
{                                                                     \
  if constexpr (NAME ## EXTRAID ## _checker<TEMP_ARGS>::template exists<option>) \
    {                                                                 \
     return static_cast<POST option*>(this)->template NAME <TEMP_ARGS>(ARG_NAMES); \
    }                                                                 \
  else                                                                \
    {                                                                 \
     return static_cast<POST base*>(this)->template NAME <TEMP_ARGS>(ARG_NAMES); \
    }                                                                 \
}                                                                     \

    template <indexer num_dims, class end, class base, class option>
    class SystemInfoMaker<num_dims, end, base, option>:
    public base,
    public option
    {      
      struct InvalidReturn {};
      //For SFINAE.
      
      public:
      using base::base;
      using option::option;
      
      AFFPICS_SYSINFO_MAKER_FUNC(CUDA_HOS_DEV, FLType, mu, (const indexer i), const, i, const indexer, 1)
     
      AFFPICS_SYSINFO_MAKER_FUNC(CUDA_HOS_DEV, FLType, mu, (const vector_type<indexer, num_dims> &p), const, p, AFFPICS_COMMA_TRICK(const vector_type<indexer, num_dims>), 2)
     
      AFFPICS_SYSINFO_MAKER_FUNC(CUDA_HOS_DEV, FLType, epsilon, (const indexer i), const, i, indexer, 1)
     
      AFFPICS_SYSINFO_MAKER_FUNC(CUDA_HOS_DEV, FLType, epsilon, (const vector_type<indexer, num_dims> &p), const, p, AFFPICS_COMMA_TRICK(const vector_type<indexer, num_dims>), 2)
     
      AFFPICS_SYSINFO_MAKER_FUNC(CUDA_HOS_DEV, const UnitSystem&, units, (), const, , void , 1)
     
      AFFPICS_SYSINFO_MAKER_FUNC(CUDA_HOS_DEV constexpr, indexer, dimensions, (), const, , void ,1)
     
      AFFPICS_SYSINFO_MAKER_FUNC(CUDA_HOS_DEV, AFFPICS_COMMA_TRICK(vector_type<indexer, num_dims>), num_cells, (), const, , void, 1)
     
      AFFPICS_SYSINFO_MAKER_FUNC(CUDA_HOS_DEV, indexer, num_cells, (const indexer i), const, i, const indexer, 2)
     
      AFFPICS_SYSINFO_MAKER_FUNC(CUDA_HOS_DEV, indexer, total_cells, (), const, , void, 1)
     
      AFFPICS_SYSINFO_MAKER_FUNC(CUDA_HOS_DEV, AFFPICS_COMMA_TRICK(vector_type<indexer, num_dims>), to_cell, (const indexer i), const, i, const indexer, 1)
     
      AFFPICS_SYSINFO_MAKER_FUNC(CUDA_HOS_DEV, indexer, to_index, (const vector_type<indexer, num_dims> &cell), const, cell, AFFPICS_COMMA_TRICK(const vector_type<indexer, num_dims> &), 1)
     
      AFFPICS_SYSINFO_MAKER_TEMPLATE_FUNC(
          <class E_holder>,
          CUDA_HOS_DEV,
          B_field_type<num_dims>,
          E_curl,
          (const E_holder &E_fields, const indexer i),
          const,
          AFFPICS_COMMA_TRICK(E_fields, i),
          AFFPICS_COMMA_TRICK(const E_holder&, const indexer),
          E_holder,
          1)
          
      AFFPICS_SYSINFO_MAKER_TEMPLATE_FUNC(
          <class E_holder>,
          CUDA_HOS_DEV,
          B_field_type<num_dims>,
          E_curl,
          (const E_holder &E_fields, const vector_type<indexer, num_dims> &p),
          const,
          AFFPICS_COMMA_TRICK(E_fields, p),
          AFFPICS_COMMA_TRICK(const E_holder&, const vector_type<indexer, num_dims>),
          E_holder,
          2)
          
      AFFPICS_SYSINFO_MAKER_TEMPLATE_FUNC(
          <class B_holder>,
          CUDA_HOS_DEV,
          E_field_type<num_dims>,
          B_curl,
          (const B_holder &B_fields, const indexer i),
          const,
          AFFPICS_COMMA_TRICK(B_fields, i),
          AFFPICS_COMMA_TRICK(const B_holder&, const indexer),
          B_holder,
          1)
          
      AFFPICS_SYSINFO_MAKER_TEMPLATE_FUNC(
          <class B_holder>,
          CUDA_HOS_DEV,
          E_field_type<num_dims>,
          B_curl,
          (const B_holder &B_fields, const vector_type<indexer, num_dims> &p),
          const,
          AFFPICS_COMMA_TRICK(B_fields, p),
          AFFPICS_COMMA_TRICK(const B_holder&, const vector_type<indexer, num_dims>&),
          B_holder,
          2)
          
      AFFPICS_SYSINFO_MAKER_TEMPLATE_FUNC(
          AFFPICS_COMMA_TRICK(<class ArrT, class part>),
          CUDA_HOS_DEV,
          E_field_type<num_dims>,
          E_gather,
          (const ArrT &E_arr, const part& particle),
          const,
          AFFPICS_COMMA_TRICK(E_arr, particle),
          AFFPICS_COMMA_TRICK(const ArrT&, const part&),
          AFFPICS_COMMA_TRICK(ArrT, part),
          1)
          
      AFFPICS_SYSINFO_MAKER_TEMPLATE_FUNC(
          AFFPICS_COMMA_TRICK(<class ArrT, class part>),
          CUDA_HOS_DEV,
          B_field_type<num_dims>,
          B_gather,
          (const ArrT &B_arr, const part& particle),
          const,
          AFFPICS_COMMA_TRICK(B_arr, particle),
          AFFPICS_COMMA_TRICK(const ArrT&, const part&),
          AFFPICS_COMMA_TRICK(ArrT, part),
          1)
          
      AFFPICS_SYSINFO_MAKER_FUNC(CUDA_HOS_DEV, AFFPICS_COMMA_TRICK(vector_type<FLType, num_dims>), cell_sizes, (), const, , void, 1)
      
      AFFPICS_SYSINFO_MAKER_FUNC(CUDA_HOS_DEV, bool, is_border, (const indexer i), const, i, const indexer, 1)
     
      AFFPICS_SYSINFO_MAKER_FUNC(CUDA_HOS_DEV, bool, is_border, (const vector_type<indexer, num_dims> &p), const, p, AFFPICS_COMMA_TRICK(const vector_type<indexer, num_dims>&), 2)
      
      AFFPICS_SYSINFO_MAKER_FUNC(CUDA_HOS_DEV, bool, is_outside, (const indexer i), const, i, const indexer, 1)
     
      AFFPICS_SYSINFO_MAKER_FUNC(CUDA_HOS_DEV, bool, is_outside, (const vector_type<indexer, num_dims> &p), const, p, AFFPICS_COMMA_TRICK(const vector_type<indexer, num_dims>&), 2)
      
      AFFPICS_SYSINFO_MAKER_TEMPLATE_FUNC(
          <class part_type>,
          CUDA_HOS_DEV constexpr,
          indexer,
          particle_cell_radius,
          (const part_type& part),
          const,
          part,
          const part_type&,
          part_type,
          1)

      AFFPICS_SYSINFO_MAKER_TEMPLATE_FUNC(
          <class part_type>,
          CUDA_HOS_DEV,
          FLType,
          particle_fraction,
          (const part_type& part, const vector_type<FLType, num_dims> &pos),
          const,
          AFFPICS_COMMA_TRICK(part, pos),
          AFFPICS_COMMA_TRICK(const part_type&, const vector_type<FLType, num_dims>&),
          part_type,
          1)
          
      AFFPICS_SYSINFO_MAKER_FUNC(CUDA_HOS_DEV, AFFPICS_COMMA_TRICK(vector_type<FLType, num_dims>), E_measurement, (const indexer dim), const, dim, const indexer, 1)
      
      AFFPICS_SYSINFO_MAKER_FUNC(CUDA_HOS_DEV, AFFPICS_COMMA_TRICK(vector_type<FLType, num_dims>), B_measurement, (const indexer dim), const, dim, const indexer, 1)
      
      AFFPICS_SYSINFO_MAKER_TEMPLATE_FUNC(
          AFFPICS_COMMA_TRICK(<bool check_for_border, class Func, class ... Args>),
          CUDA_HOS_DEV,
          void,
          for_all_neighbours,
          (const indexer radius, const indexer index, Func && func, Args&& ... args),
          const,
          AFFPICS_COMMA_TRICK(radius, index, std::forward<Func>(func), std::forward<Args>(args)...),
          AFFPICS_COMMA_TRICK(const indexer, const indexer, Func, Args...),
          AFFPICS_COMMA_TRICK(check_for_border, Func, Args...),
          1)
      
      AFFPICS_SYSINFO_MAKER_TEMPLATE_FUNC(
          AFFPICS_COMMA_TRICK(<bool check_for_border, class Func, class ... Args>),
          CUDA_HOS_DEV,
          void,
          for_all_neighbours,
          (const indexer radius, const vector_type<indexer, num_dims> &cell, Func && func, Args&& ... args),
          const,
          AFFPICS_COMMA_TRICK(radius, cell, std::forward<Func>(func), std::forward<Args>(args)...),
          AFFPICS_COMMA_TRICK(const indexer, const vector_type<indexer, num_dims>&, Func, Args...),
          AFFPICS_COMMA_TRICK(check_for_border, Func, Args...),
          2)
          
      AFFPICS_SYSINFO_MAKER_TEMPLATE_FUNC(
          <class particle>,
          CUDA_HOS_DEV,
          void,
          boundary_particles,
          (particle& part, const bool force_apply = false),
          const,
          AFFPICS_COMMA_TRICK(part, force_apply),
          AFFPICS_COMMA_TRICK(particle&, const bool),
          particle,
          1)
          
      AFFPICS_SYSINFO_MAKER_TEMPLATE_FUNC(
          <class ArrT>,
          CUDA_HOS_DEV,
          E_field_type<num_dims>,
          boundary_E,
          (const vector_type<indexer, num_dims> &cell, const ArrT & E_fields),
          const,
          AFFPICS_COMMA_TRICK(cell, E_fields),
          AFFPICS_COMMA_TRICK(const vector_type<indexer, num_dims>&, const ArrT&),
          ArrT,
          1)
          
      AFFPICS_SYSINFO_MAKER_TEMPLATE_FUNC(
          <class ArrT>,
          CUDA_HOS_DEV,
          B_field_type<num_dims>,
          boundary_B,
          (const vector_type<indexer, num_dims> &cell, const ArrT & B_fields),
          const,
          AFFPICS_COMMA_TRICK(cell, B_fields),
          AFFPICS_COMMA_TRICK(const vector_type<indexer, num_dims>&, const ArrT&),
          ArrT,
          1)
      
      AFFPICS_SYSINFO_MAKER_TEMPLATE_FUNC(
          <class ArrT>,
          CUDA_HOS_DEV,
          current_type<num_dims>,
          boundary_J,
          (const vector_type<indexer, num_dims> &cell, const ArrT & currents),
          const,
          AFFPICS_COMMA_TRICK(cell, currents),
          AFFPICS_COMMA_TRICK(const vector_type<indexer, num_dims>&, const ArrT&),
          ArrT,
          1)
          
      AFFPICS_SYSINFO_MAKER_TEMPLATE_FUNC(
          AFFPICS_COMMA_TRICK(<class EArr, class BArr, class JArr, class PartStorage>),
          CUDA_HOS_DEV,
          void,
          initial_condition,
          (EArr& E_fields, BArr & B_fields, JArr & currents, PartStorage& particles),
          const,
          AFFPICS_COMMA_TRICK(E_fields, B_fields, currents, particles),
          AFFPICS_COMMA_TRICK(EArr&, BArr&, JArr&, PartStorage&),
          AFFPICS_COMMA_TRICK(EArr, BArr, JArr, PartStorage),
          1)
    };
    
    template <indexer num_dims, class end, class ... options>
    using SystemInfo = SystemInfoMaker<num_dims, end, BaseSystemInfo<num_dims, end>, options...>;
  }
}

#undef AFFPICS_COMMA_TRICK
#undef AFFPICS_SYSINFO_MAKER_FUNC
#undef AFFPICS_SYSINFO_MAKER_TEMPLATE_FUNC

#endif