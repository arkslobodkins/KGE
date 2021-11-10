#ifndef PROBLEM_PARAMS_H
#define PROBLEM_PARAMS_H

namespace KGE
{
   constexpr double square(double x)
   {
      return x*x;
   }

   enum BC {DIRICHLET, NEUMANN, SOMMERFELD};
   constexpr enum BC bc_global                        = DIRICHLET;
   constexpr unsigned int dim_global                  = 1;
   constexpr unsigned int fe_degree_global            = 6;

   constexpr double wave_speed_global                 = 1.0;
   constexpr double alpha_global                      = 4.0;

   constexpr double initial_time_global               = 0.0;
   constexpr double final_time_global                 = 10.0;

   constexpr double left_global                       = -10.0;
   constexpr double right_global                      = 10.0;
   constexpr unsigned int output_timestep_skip_global = 200;

}

#endif
