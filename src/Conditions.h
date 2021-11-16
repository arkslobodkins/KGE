/* Arkadijs Slobodkins
 * Summer 2021
 */


#ifndef COND_NONHOMOGENOUS_H
#define COND_NONHOMOGENOUS_H

#include "ProblemParams.h"

#include <iostream>
#include <stdexcept>
#include <cmath>
#include <cassert>
#include <deal.II/base/function.h>

namespace KGE
{
   using namespace dealii;

   // forward class declarations;
   template <int dim> class AnalyticalSolutionU;
   template <int dim> class AnalyticalSolutionV;
   template <int dim> class BoundaryConditionU;
   template <int dim> class BoundaryConditionV;
   template <int dim> class InitialConditionU;
   template <int dim> class InitialConditionV;

   template <int dim>
   class  KGEConditions
   {
   public:
      AnalyticalSolutionU<dim> *analytical_U;
      AnalyticalSolutionV<dim> *analytical_V;
      BoundaryConditionU<dim> *boundary_U;
      BoundaryConditionV<dim> *boundary_V;
      InitialConditionU<dim> *initial_U;
      InitialConditionV<dim> *initial_V;

      KGEConditions() = delete;
      KGEConditions(const double _wave_speed,
                    const double _alpha,
                    const enum BC _bc = DIRICHLET,
                    const unsigned int n_components = 1,
                    const double time = 0.)
      {
         static_assert(dim <= 3, "dim <= 3 condition is not satisfied");
         analytical_U = new AnalyticalSolutionU<dim>(_wave_speed, _alpha, _bc, n_components, time);
         analytical_V = new AnalyticalSolutionV<dim>(_wave_speed, _alpha, _bc, n_components, time);
         boundary_U   = new BoundaryConditionU<dim>(_wave_speed, _alpha, _bc, n_components, time);
         boundary_V   = new BoundaryConditionV<dim>(_wave_speed, _alpha, _bc, n_components, time);
         initial_U    = new InitialConditionU<dim>(_wave_speed, _alpha, _bc, n_components);
         initial_V    = new InitialConditionV<dim>(_wave_speed, _alpha, _bc, n_components);
      }

      ~KGEConditions()
      {
         delete analytical_U;
         delete analytical_V;
         delete boundary_U;
         delete boundary_V;
         delete initial_U;
         delete initial_V;
      }
   };

   template<int dim>
   class BoundaryConditionU: public Function<dim>
   {
   public:
      BoundaryConditionU(const double _wave_speed,
                         const double _alpha,
                         const enum BC _bc = DIRICHLET,
                         const unsigned int n_components = 1,
                         const double time = 0.)
         : Function<dim>(n_components, time),
         wave_speed(_wave_speed),
         alpha(_alpha),
         bc(_bc)
      {}

      virtual double value(const Point<dim> &p, const unsigned int /*component*/) const override
      {
         using namespace std;
         double pi = numbers::PI;
         double t  = this->get_time();

         if(dim == 1)
         {
            double k  = square(wave_speed) * square(pi) + square(alpha);
            enum class side {xleft, xright} s;
            if(p[0] > right_global - bound_tol && p[0] < right_global + bound_tol)
               s = side::xright;
            else if (p[0] < left_global + bound_tol  && p[0] > left_global - bound_tol)
               s = side::xleft;
            else
               throw range_error("point is not on the boundary");

            switch(bc)
            {
               case(DIRICHLET):
               {
                  double G = sqrt( square(wave_speed) + square(alpha) / square(pi) );
                  return sin(pi * (p[0] - G*t));
               }
               case(NEUMANN):
               {
                  if(s == side::xright)
                     return 1.0/sqrt(k) * sin(sqrt(k)*t) * pi * right_sign_cos;
                  else if (s == side::xleft)
                     return -1.0/sqrt(k) * sin(sqrt(k)*t) * pi * left_sign_cos;
                  [[fallthrough]];
               }
               case(SOMMERFELD):
               {
                  if(s == side::xright) {
                     double gradient = 1.0/sqrt(k) * sin(sqrt(k)*t) * pi * ( right_sign_cos - zero_sin );
                     double du_dt = cos(sqrt(k)*t) * (zero_sin + right_sign_cos);
                     return du_dt + gradient;
                  }
                  else if (s == side::xleft) {
                     double gradient = 1.0/sqrt(k) * sin(sqrt(k)*t) * pi * ( left_sign_cos - zero_sin );
                     double du_dt = cos(sqrt(k)*t) * (zero_sin + left_sign_cos);
                     return du_dt - gradient;
                  }
                  [[fallthrough]];
               }
               default:
                  throw invalid_argument("Inappropriate boundary conditions were specified");
            }
         }

         else if(dim == 2)
         {
            double k  = 2.0 * square(wave_speed) * square(pi) + square(alpha);
            switch(bc)
            {
               case(DIRICHLET):  return 1.0/sqrt(k) * sin(sqrt(k)*t) * sin(pi*p[0]) * sin(pi*p[1]);
               case(NEUMANN):    return 0.0;
               case(SOMMERFELD): return cos(sqrt(k)*t) * cos(pi*p[0]) * cos(pi*p[1]);
               default:
                  throw invalid_argument("Inappropriate boundary conditions were specified");
            }
         }

         else if(dim == 3)
         {
            double k  = 3.0 * square(wave_speed) * square(pi) + square(alpha);
            switch(bc)
            {
               case(DIRICHLET):  return 1.0/sqrt(k) * sin(sqrt(k)*t) * sin(pi*p[0]) * sin(pi*p[1]) * sin(pi*p[2]);
               case(NEUMANN):    return 0.0;
               case(SOMMERFELD): return cos(sqrt(k)*t) * cos(pi*p[0]) * cos(pi*p[1]) * cos(pi*p[2]);
               default:
                  throw invalid_argument("Inappropriate boundary conditions were specified");
            }
         }

      }

   private:
      const double wave_speed;
      const double alpha;
      enum BC bc;
   };



   template<int dim>
   class BoundaryConditionV: public Function<dim>
   {
   public:
      BoundaryConditionV(const double _wave_speed,
                         const double _alpha,
                         const enum BC _bc = DIRICHLET,
                         const unsigned int n_components = 1,
                         const double time = 0.)
         : Function<dim>(n_components, time),
         wave_speed(_wave_speed),
         alpha(_alpha),
         bc(_bc)
      {}

      virtual double value(const Point<dim> &p, const unsigned int /*component*/) const override
      {
         using namespace std;
         double pi = numbers::PI;
         double t  = this->get_time();

         if(dim == 1)
            switch(bc)
            {
               case(DIRICHLET):
               {
                  double G = sqrt( square(wave_speed) + square(alpha) / square(pi) );
                  return -pi * G * cos(pi * (p[0] - G*t));
               }

               // Computation of jumps for NEUMANN and SOMMERFELD
               // does not require conditions for V
               case(NEUMANN)   : return 0.0;
               case(SOMMERFELD): return 0.0;
               default: {
                  throw invalid_argument("Inappropriate boundary conditions were specified");
               }
            }

         else if(dim == 2)
         {
            double k  = 2.0 * square(wave_speed) * square(pi) + square(alpha);
            switch(bc)
            {
               case(DIRICHLET):
                  return cos(sqrt(k)*t) * sin(pi*p[0]) * sin(pi*p[1]);

               // Computation of jumps for NEUMANN and SOMMERFELD
               // does not require conditions for V
               case(NEUMANN):    return 0.0;
               case(SOMMERFELD): return 0.0;
               default:
                  throw invalid_argument("Inappropriate boundary conditions were specified");
            }
         }

         else if(dim == 3)
         {
            double k  = 3.0 * square(wave_speed) * square(pi) + square(alpha);
            switch(bc)
            {
               case(DIRICHLET):
                  return cos(sqrt(k)*t) * sin(pi*p[0]) * sin(pi*p[1]) * sin(pi*p[2]);

               // Computation of jumps for NEUMANN and SOMMERFELD
               // does not require conditions for V
               case(NEUMANN):    return 0.0;
               case(SOMMERFELD): return 0.0;
               default:
                  throw invalid_argument("Inappropriate boundary conditions were specified");
            }
         }
      }

   private:
      const double wave_speed;
      const double alpha;
      enum BC bc;
   };



   template <int dim>
   class InitialConditionU : public Function<dim>
   {
   public:
      InitialConditionU(const double _wave_speed,
                        const double _alpha,
                        const enum BC _bc = DIRICHLET,
                        const unsigned int n_components = 1)
         : Function<dim>(n_components),
         wave_speed(_wave_speed),
         alpha(_alpha),
         bc(_bc)
      {}

      virtual double value(const Point<dim> &p,
                           const unsigned int /*component*/) const override
      {
         using namespace std;

         if(dim == 1)
            switch(bc)
            {
               case(DIRICHLET):  return sin(numbers::PI * p[0]);
               case(NEUMANN):    return 0.0;
               case(SOMMERFELD): return 0.0;
               default:
                  throw invalid_argument("Inappropriate boundary conditions were specified");
            }

         else if(dim == 2)
            switch(bc)
            {
               case(DIRICHLET):  return 0.0;
               case(NEUMANN):    return 0.0;
               case(SOMMERFELD): return 0.0;
               default:
                  throw invalid_argument("Inappropriate boundary conditions were specified");
            }

         else if(dim == 3)
            switch(bc)
            {
               case(DIRICHLET):  return 0.0;
               case(NEUMANN):    return 0.0;
               case(SOMMERFELD): return 0.0;
               default:
                  throw invalid_argument("Inappropriate boundary conditions were specified");
            }
      }

   private:
      const double wave_speed;
      const double alpha;
      enum BC bc;
   };



   template <int dim>
   class InitialConditionV : public Function<dim>
   {
   public:
      InitialConditionV(const double _wave_speed,
                        const double _alpha,
                        const enum BC _bc = DIRICHLET,
                        const unsigned int n_components = 1)
         : Function<dim>(n_components),
         wave_speed(_wave_speed),
         alpha(_alpha),
         bc(_bc)
      {}

      virtual double value(const Point<dim> &p,
                           const unsigned int /*component*/) const override
      {
         using namespace std;
         double result = 0.0;
         double pi = numbers::PI;

         if(dim == 1)
            switch(bc)
            {
               case(DIRICHLET):
               {
                  double G = sqrt( square(wave_speed) + square(alpha) / square(pi) );
                  result   = -pi * G * cos(pi*p[0]);
                  break;
               }
               case(NEUMANN):
               {
                  result = sin(pi*p[0]);
                  break;
               }
               case(SOMMERFELD):
               {
                  result = sin(pi*p[0]) + cos(pi*p[0]);
                  break;
               }
               default:
                  throw invalid_argument("Inappropriate boundary conditions were specified");
            }

         else if(dim == 2)
            switch(bc)
            {
               case(DIRICHLET):
                  result = sin(pi*p[0]) * sin(pi*p[1]);
                  break;
               case(NEUMANN):
                  result = cos(pi*p[0]) * cos(pi*p[1]);
                  break;
               case(SOMMERFELD):
                  result = cos(pi*p[0]) * cos(pi*p[1]);
                  break;
               default:
                  throw invalid_argument("Inappropriate boundary conditions were specified");
            }

         else if(dim == 3)
            switch(bc)
            {
               case(DIRICHLET):
                  result = sin(pi*p[0]) * sin(pi*p[1]) * sin(pi*p[2]);
                  break;
               case(NEUMANN):
                  result = cos(pi*p[0]) * cos(pi*p[1]) * cos(pi*p[2]);
                  break;
               case(SOMMERFELD):
                  result = cos(pi*p[0]) * cos(pi*p[1]) * cos(pi*p[2]);
                  break;
               default:
                  throw invalid_argument("Inappropriate boundary conditions were specified");
            }

         return result;
      }

   private:
      const double wave_speed;
      const double alpha;
      enum BC bc;
   };



   template <int dim>
   class AnalyticalSolutionU : public Function<dim>
   {
   public:
      AnalyticalSolutionU(const double _wave_speed,
                          const double _alpha,
                          const enum BC _bc = DIRICHLET,
                          const unsigned int n_components = 1,
                          const double time = 0.)
         :  Function<dim>(n_components, time),
         wave_speed(_wave_speed),
         alpha(_alpha),
         bc(_bc)
      {}

      virtual double value(const Point<dim> &p,
                           const unsigned int /*component*/) const override
      {
         using namespace std;
         double pi = numbers::PI;
         double t  = this->get_time();
         double result = 0.0;

         if(dim == 1)
         {
            double k  = square(wave_speed) * square(pi) + square(alpha);
            switch(bc)
            {
               case(DIRICHLET):
               {
                  double G = sqrt( square(wave_speed) + square(alpha) / square(pi) );
                  result   = sin(pi * (p[0] - G*t));
                  break;
               }
               case(NEUMANN):
               {
                  result = 1.0/sqrt(k) * sin(sqrt(k)*t) * sin(pi*p[0]);
                  break;
               }
               case(SOMMERFELD):
               {
                  result = 1.0/sqrt(k) * sin(sqrt(k)*t) * ( sin(pi*p[0]) + cos(pi*p[0]) );
                  break;
               }
               default:
                  throw invalid_argument("Inappropriate boundary conditions were specified");
            }
         }

         else if(dim == 2)
         {
            double k  = 2.0 * square(wave_speed) * square(pi) + square(alpha);
            switch(bc)
            {
               case(DIRICHLET):
                  result = 1.0/sqrt(k) * sin(sqrt(k)*t) * sin(pi*p[0]) * sin(pi*p[1]);
                  break;
               case(NEUMANN):
                  result = 1.0/sqrt(k) * sin(sqrt(k)*t) * cos(pi*p[0]) * cos(pi*p[1]);
                  break;
               case(SOMMERFELD):
                  result = 1.0/sqrt(k) * sin(sqrt(k)*t) * cos(pi*p[0]) * cos(pi*p[1]);
                  break;
               default:
                  throw invalid_argument("Inappropriate boundary conditions were specified");
            }
         }

         else if(dim == 3)
         {
            double k  = 3.0 * square(wave_speed) * square(pi) + square(alpha);
            switch(bc)
            {
               case(DIRICHLET):
                  result = 1.0/sqrt(k) * sin(sqrt(k)*t) * sin(pi*p[0]) * sin(pi*p[1]) * sin(pi*p[2]);
                  break;
               case(NEUMANN):
                  result = 1.0/sqrt(k) * sin(sqrt(k)*t) * cos(pi*p[0]) * cos(pi*p[1]) * cos(pi*p[2]);
                  break;
               case(SOMMERFELD):
                  result = 1.0/sqrt(k) * sin(sqrt(k)*t) * cos(pi*p[0]) * cos(pi*p[1]) * cos(pi*p[2]);
                  break;
               default:
                  throw invalid_argument("Inappropriate boundary conditions were specified");
            }
         }

         return result;
      }

      private:
         const double wave_speed;
         const double alpha;
         enum BC bc;
   };



   template <int dim>
   class AnalyticalSolutionV : public Function<dim>
   {
   public:
      AnalyticalSolutionV(const double _wave_speed,
                          const double _alpha,
                          const enum BC _bc = DIRICHLET,
                          const unsigned int n_components = 1,
                          const double time = 0.)
         : Function<dim>(n_components, time),
         wave_speed(_wave_speed),
         alpha(_alpha),
         bc(_bc)
      {}

      virtual double value(const Point<dim> &p,
                           const unsigned int /*component*/) const override
      {
         using namespace std;
         double pi = numbers::PI;
         double t  = this->get_time();
         double result = 0.0;

         if(dim == 1)
         {
            double k  = square(wave_speed) * square(pi) + square(alpha);
            switch(bc)
            {
               case(DIRICHLET):
               {
                  double G = sqrt( square(wave_speed) + square(alpha) / square(pi) );
                  result   = -pi * G * cos(pi * (p[0] - G*t));
                  break;
               }
               case(NEUMANN):
               {
                  result = cos(sqrt(k)*t) * sin(pi*p[0]);
                  break;
               }
               case(SOMMERFELD):
               {
                  result = cos(sqrt(k)*t) * ( sin(pi*p[0]) + cos(pi*p[0]) );
                  break;
               }
               default:
                  throw invalid_argument("Inappropriate boundary conditions were specified");
            }
         }

         else if(dim == 2)
         {
            double k = 2.0 * square(wave_speed) * square(pi) + square(alpha);
            switch(bc)
            {
               case(DIRICHLET):
                  result = cos(sqrt(k)*t) * sin(pi*p[0]) * sin(pi*p[1]);
                  break;
               case(NEUMANN):
                  result = cos(sqrt(k)*t) * cos(pi*p[0]) * cos(pi*p[1]);
                  break;
               case(SOMMERFELD):
                  result = cos(sqrt(k)*t) * cos(pi*p[0]) * cos(pi*p[1]);
                  break;
               default:
                  throw invalid_argument("Inappropriate boundary conditions were specified");
            }
         }

         else if(dim == 3)
         {
            double k  = 3.0 * square(wave_speed) * square(pi) + square(alpha);
            switch(bc)
            {
               case(DIRICHLET):
                  result = cos(sqrt(k)*t) * sin(pi*p[0]) * sin(pi*p[1]) * sin(pi*p[2]);
                  break;
               case(NEUMANN):
                  result = cos(sqrt(k)*t) * cos(pi*p[0]) * cos(pi*p[1]) * cos(pi*p[2]);
                  break;
               case(SOMMERFELD):
                  result = cos(sqrt(k)*t) * cos(pi*p[0]) * cos(pi*p[1]) * cos(pi*p[2]);
                  break;
               default:
                  throw invalid_argument("Inappropriate boundary conditions were specified");
            }
         }

         return result;
      }

   private:
      const double wave_speed;
      const double alpha;
      enum BC bc;
   };

} // namespace KGE

#endif
