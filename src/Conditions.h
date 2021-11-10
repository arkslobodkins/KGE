#ifndef COND_NONHOMOGENOUS_H
#define COND_NONHOMOGENOUS_H

#include "ProblemParams.h"

#include <cmath>
#include <deal.II/base/function.h>

namespace KGE
{
   using namespace dealii;

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
         if(dim != 1)
            AssertThrow(false, ExcNotImplemented());

         using namespace std;
         double pi = numbers::PI;
         double k  = square(wave_speed) * square(pi) + square(alpha);
         double t  = this->get_time();
         double result = 0.0;

         switch(bc)
         {
            case(DIRICHLET):
            {
               double G = sqrt( square(wave_speed) + square(alpha) / square(pi) );
               result = sin(pi * (p[0] - G*t));
               break;
            }
            case(NEUMANN):
            {
               if( p[0] > right_global - pow(10, -12) )
                  result = 1.0/sqrt(k) * sin(sqrt(k)*t) * pi;
               else
                  result = -1.0/sqrt(k) * sin(sqrt(k)*t) * pi;
               break;
            }
            case(SOMMERFELD):
            {
               if( p[0] > right_global - pow(10, -12) )
                  result = 1.0/sqrt(k) * sin(sqrt(k)*t) * pi + cos(sqrt(k)*t) * cos(pi*p[0]);
               else
                  result = -1.0/sqrt(k) * sin(sqrt(k)*t) * pi + cos(sqrt(k)*t) * cos(pi*p[0]);
               break;
            }
         }

         return result;
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
         if(dim != 1)
            AssertThrow(false, ExcNotImplemented());

         using namespace std;
         double pi = numbers::PI;
         double t  = this->get_time();

         switch(bc)
         {
            case(DIRICHLET):
            {
               double G = sqrt( square(wave_speed) + square(alpha) / square(pi) );
               return -pi * G * cos(pi * (p[0] - G*t));
            }

            //Computation of jumps for NEUMANN and SOMMERFELD
            //does not require conditions for V
            case(NEUMANN)   : return 0.0;
            case(SOMMERFELD): return 0.0;
            default:          return 0.0;
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
         if(dim != 1)
            AssertThrow(false, ExcNotImplemented());

         switch(bc)
         {
            case(DIRICHLET):  return std::sin(numbers::PI * p[0]);
            case(SOMMERFELD): return 0.0;
            case(NEUMANN):    return 0.0;
            default:          return 0.0;
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
         if(dim != 1)
            AssertThrow(false, ExcNotImplemented());

         using namespace std;
         double result = 0.0;
         double pi = numbers::PI;

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
         if(dim != 1)
            AssertThrow(false, ExcNotImplemented());

         using namespace std;
         double pi = numbers::PI;
         double k  = square(wave_speed) * square(pi) + square(alpha);
         double t  = this->get_time();
         double result = 0.0;

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
               result = 1.0/sqrt(k) * sin(sqrt(k)*t) * sin(pi*p[0])
                      + 1.0/sqrt(k) * sin(sqrt(k)*t) * cos(pi*p[0]);
               break;
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
         if(dim != 1)
            AssertThrow(false, ExcNotImplemented());

         using namespace std;
         double pi = numbers::PI;
         double k  = square(wave_speed) * square(pi) + square(alpha);
         double t  = this->get_time();
         double result = 0.0;

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
               result = cos(sqrt(k)*t) * sin(pi*p[0])
                      + cos(sqrt(k)*t) * cos(pi*p[0]);
               break;
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
   class  KGENonHomogenous
   {
   public:
      AnalyticalSolutionU<dim> *analytical_U;
      AnalyticalSolutionV<dim> *analytical_V;
      BoundaryConditionU<dim> *boundary_U;
      BoundaryConditionV<dim> *boundary_V;
      InitialConditionU<dim> *initial_U;
      InitialConditionV<dim> *initial_V;

      KGENonHomogenous(const double _wave_speed,
                       const double _alpha,
                       const enum BC _bc = DIRICHLET,
                       const unsigned int n_components = 1,
                       const double time = 0.)
      {
         analytical_U = new AnalyticalSolutionU<dim>(_wave_speed, _alpha, _bc, n_components, time);
         analytical_V = new AnalyticalSolutionV<dim>(_wave_speed, _alpha, _bc, n_components, time);
         boundary_U   = new BoundaryConditionU<dim>(_wave_speed, _alpha, _bc, n_components, time);
         boundary_V   = new BoundaryConditionV<dim>(_wave_speed, _alpha, _bc, n_components, time);
         initial_U    = new InitialConditionU<dim>(_wave_speed, _alpha, _bc, n_components);
         initial_V    = new InitialConditionV<dim>(_wave_speed, _alpha, _bc, n_components);
      }

      ~KGENonHomogenous()
      {
         delete analytical_U;
         delete analytical_V;
         delete boundary_U;
         delete boundary_V;
         delete initial_U;
         delete initial_V;
      }
   };


} // namespace KGE

#endif
