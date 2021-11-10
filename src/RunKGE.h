#ifndef DABSOLVER_H
#define DABSOLVER_H

#include "Conditions.h"
#include "KleinGordonProblem.h"
#include "ProblemParams.h"

namespace KGE
{
   using namespace dealii;

   template<int dim, int fe_degree>
   void RunKGE()
   {
      KGENonHomogenous<dim> kge(wave_speed_global, alpha_global, bc_global, 1, 0.0);

      KleinGordonProblem<dim, fe_degree>
      KGE_problem(kge.analytical_U, kge.analytical_V,
                  *kge.initial_U, *kge.initial_V,
                  *kge.boundary_U, *kge.boundary_V,
                  left_global, right_global,
                  wave_speed_global, alpha_global, bc_global,
                  initial_time_global, final_time_global, output_timestep_skip_global);

      KGE_problem.run();
   }

} // namespace KGE

#endif
