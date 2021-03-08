#pragma once

#include <type_traits>

#include "../Types.hpp"
#include "../ScalarTraits.hpp"


namespace LSLOpt {

namespace Implementation {

///! @brief Remove CV and reference from type.
template<typename T>
struct remove_cvref {
    ///! @brief The type `T` without CV and reference.
    using type = typename std::remove_cv<typename std::remove_reference<T>::type>::type;
};

/**
 * @brief Traits for the problem.
 *
 * This class is used to check which features the problem
 * implements.
 *
 * This is used for better error messages using static_assert.
 *
 * It uses the SFINAE concept.
 */
template<typename Problem, typename Scalar>
class problem_traits {
    static constexpr Vector<Scalar>* const dummy_vec = nullptr;
    static constexpr Problem* const dummy_problem = nullptr;

    template<typename U>
    static auto value_probe(U& u) -> decltype(u.value(*dummy_vec));
    static void value_probe(...);

    template<typename U>
    static auto gradient_probe(U& u) -> decltype(u.gradient(*dummy_vec));
    static void gradient_probe(...);

    template<typename U>
    static auto lower_bound_probe(U& u) -> decltype(u.lower_bounds());
    static void lower_bound_probe(...);

    template<typename U>
    static auto upper_bound_probe(U& u) -> decltype(u.upper_bounds());
    static void upper_bound_probe(...);

    template<typename U>
    static auto initial_step_length_probe(U& u)
        -> decltype(u.initial_step_length(*dummy_vec, *dummy_vec));
    static void initial_step_length_probe(...);

    template<typename U>
    static auto change_acceptable_probe(U& u)
        -> decltype(u.change_acceptable(*dummy_vec, *dummy_vec));
    static void change_acceptable_probe(...);

    template<typename U>
    static auto is_check_acceptable_first_probe(U& u) -> decltype(u.is_check_acceptable_first());
    static void is_check_acceptable_first_probe(...);

    template<typename U>
    static auto reparameterize_probe(U& u) -> decltype(u.reparameterize(*dummy_vec, *dummy_vec));
    static void reparameterize_probe(...);

public:
    /// return type of the `value` function; `void` if not provided
    using value_type = decltype(value_probe(*dummy_problem));
    /// check if the problem has the `value` function
    static constexpr const bool has_value
        = std::is_same<Scalar, typename remove_cvref<value_type>::type>::value;

    /// return type of the `gradient` function; `void` if not provided
    using gradient_type = decltype(gradient_probe(*dummy_problem));
    /// check if the problem has the `gradient` function
    static constexpr const bool has_gradient
        = std::is_same<Vector<Scalar>,
                       typename remove_cvref<gradient_type>::type>::value;

    /// return type of the `lower_bounds` function; `void` if not provided
    using lower_bound_type = decltype(lower_bound_probe(*dummy_problem));
    /// check if the problem has the `lower_bounds` function
    static constexpr const bool has_lower_bound
        = std::is_same<Vector<Scalar>,
                       typename remove_cvref<lower_bound_type>::type>::value;

    /// return type of the `upper_bounds` function; `void` if not provided
    using upper_bound_type = decltype(upper_bound_probe(*dummy_problem));
    /// check if the problem has the `upper_bounds` function
    static constexpr const bool has_upper_bound
        = std::is_same<Vector<Scalar>,
                       typename remove_cvref<upper_bound_type>::type>::value;

    /// return type of the `initial_step_length` function; `void` if not provided
    using initial_step_length_type = decltype(initial_step_length_probe(*dummy_problem));
    /// check if the problem has the `initial_step_length` function
    static constexpr const bool has_initial_step_length
        = std::is_same<Scalar,
                       typename remove_cvref<initial_step_length_type>::type>::value;

    /// return type of the `change_acceptable` function; `void` if not provided
    using change_acceptable_type = decltype(change_acceptable_probe(*dummy_problem));
    /// check if the problem has the `change_acceptable` function
    static constexpr const bool has_change_acceptable
        = std::is_same<Scalar,
                       typename remove_cvref<change_acceptable_type>::type>::value;

    /// return type of the `check_acceptable_first` function; `void` if not provided
    using is_check_acceptable_first_type = decltype(is_check_acceptable_first_probe(*dummy_problem));
    /// check if the problem has the `check_acceptable_first` function
    static constexpr const bool has_is_check_acceptable_first
        = std::is_same<bool,
                       typename remove_cvref<is_check_acceptable_first_type>::type>::value;

    /// return type of the `reparameterize` function; `void` if not provided
    using reparameterize_type = decltype(reparameterize_probe(*dummy_problem));
    /// check if the problem has the `reparameterize` function
    static constexpr const bool has_reparameterize
        = std::is_same<bool,
                       typename remove_cvref<reparameterize_type>::type>::value;
};

template<typename Problem, typename Scalar>
/**
 * @brief The max step length identifier.
 */
struct MaxStepLengthIdentifierTrait {
  /**
   * @brief Get the initial step length from problem.
   * @param problem The optimization problem.
   * @param x Current value.
   * @param p Search direction.
   * @returns The initial step length.
   */
  static Scalar initial_step_length(
      Problem& problem,
      const Vector<Scalar>& x,
      const Vector<Scalar>& p)
  {
    static_assert(problem_traits<Problem, Scalar>::has_initial_step_length,
        "Problem must provide initial step length function: Scalar initial_step_length(...)");

    return problem.initial_step_length(x, p);
  }

  /**
   * @brief Check if suggested step is acceptable.
   * @param problem The optimization problem.
   * @param x0 Current value.
   * @param x Suggested new value.
   * @returns If change is acceptable.
   */
  static Scalar change_acceptable(
      Problem& problem,
      const Vector<Scalar>& x0,
      const Vector<Scalar>& x)
  {
    static_assert(problem_traits<Problem, Scalar>::has_change_acceptable,
        "Problem must provide change acceptable function: Scalar change_acceptable(...)");

    return problem.change_acceptable(x0, x);
  }

  /**
   * @brief Check if change acceptability should be checked first.
   * @param problem The optimization problem.
   * @returns `true` If acceptability is to be checked first.
   */
  static bool is_check_acceptable_first(
      Problem& problem)
  {
    return _is_check_acceptable_first(problem);
  }

  private:

  template<typename U,
    typename std::enable_if<!problem_traits<U, Scalar>::has_is_check_acceptable_first,
      U>::type* = nullptr>
  static constexpr bool _is_check_acceptable_first(
    U&)
  {
    return false;
  }

  template<typename U,
    typename std::enable_if<problem_traits<U, Scalar>::has_is_check_acceptable_first,
      void>::type* = nullptr>
  static bool _is_check_acceptable_first(
      U& problem)
  {
    return problem.is_check_acceptable_first();
  }

};

/**
 * @brief The no step length identifier.
 */
template<typename Problem, typename Scalar>
struct NoMaxStepLengthIdentifierTrait {
  /**
   * @brief Get the intial step length from problem.
   * @returns \f$ \infty \f$.
   */
  static constexpr Scalar initial_step_length(
      Problem&,
      const Vector<Scalar>&,
      const Vector<Scalar>&)
  {
    return scalar_traits<Scalar>::infinity();
  }

  /**
   * @brief Check if change is acceptable.
   * @returns `Scalar{0}`
   */
  static constexpr Scalar change_acceptable(
      Problem&,
      const Vector<Scalar>&,
      const Vector<Scalar>&)
  {
    return Scalar{0};
  }

  /**
   * @brief Check if change acceptability should be checked first.
   * @returns `false`
   */
  static constexpr bool is_check_acceptable_first(
      Problem&)
  {
    return false;
  }
};

template<typename Problem, typename Scalar>
/**
 * @brief Reparameterizer.
 */
struct ReparameterizeTrait {
  /**
   * @brief Reparemeterize x and recalculate the gradient.
   * @param problem The optimization problem.
   * @param x The current x vector, later holding the reparameterized result.
   * @param grad The gradient storing the current and later the reparameterized result.
   * @returns `true` If a change was performed, `false` otherwise.
   */
  static bool reparameterize(
      Problem& problem,
      Vector<Scalar>& x,
      Vector<Scalar>& grad)
  {
    return _reparameterize(problem, x, grad);
  }

  private:

  template<typename U,
    typename std::enable_if<!problem_traits<U, Scalar>::has_reparameterize,
      U>::type* = nullptr>
  static constexpr bool _reparameterize(
    U&,
    Vector<Scalar>& x,
    Vector<Scalar>& grad)
  {
    return false;
  }

  template<typename U,
    typename std::enable_if<problem_traits<U, Scalar>::has_reparameterize,
    U>::type* = nullptr>
  static constexpr bool _reparameterize(
    U& problem,
    Vector<Scalar>& x,
    Vector<Scalar>& grad)
  {
    return problem.reparameterize(x, grad);
  }
};

}

}
