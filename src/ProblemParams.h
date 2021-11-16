/* Arkadijs Slobodkins
 * Summer 2021
 */

#ifndef PROBLEM_PARAMS_H
#define PROBLEM_PARAMS_H
#include <cmath>

namespace KGE
{
   #define pow_double(x,y) pow(double(x), double(y))

   constexpr double square(double x)
   {
      return x*x;
   }

   constexpr bool is_even(int x)
   {
      return x % 2 == 0;
   }

   enum BC {DIRICHLET, NEUMANN, SOMMERFELD};
   constexpr enum BC bc_global {SOMMERFELD};
   constexpr unsigned int dim_global {3};
   constexpr unsigned int fe_degree_global {7};

   constexpr double wave_speed_global {2.0};
   constexpr double alpha_global {4.0};

   constexpr double initial_time_global {0.0};
   constexpr double final_time_global {5.0};

   // left and right boundaries of the line/square/cube
   // should be integers so that boundary conditions
   // of the analytical solution match.
   constexpr int left_g {-7};
   constexpr int right_g {8};
   constexpr double left_global  {double(left_g)};
   constexpr double right_global {double(right_g)};
   constexpr unsigned int output_timestep_skip_global {200};

   constexpr double right_sign_cos = (is_even(right_g) ? 1.0 : -1.0);
   constexpr double left_sign_cos = (is_even(left_g) ?  1.0 : -1.0);
   constexpr double zero_sin = 0.0;

   const double bound_tol = pow_double(10.0, -12.0);
}

#endif
