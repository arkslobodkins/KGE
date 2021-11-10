#ifndef KGP_H
#define KGP_H

#include "KleinGordonOperation.h"

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/base/timer.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/base/time_stepping.h>

namespace KGE
{
   using namespace dealii;
   #define num_cells 64


   template <int dim, int fe_degree>
   class KleinGordonProblem
   {
   public:
      KleinGordonProblem(
            Function<dim> *_analytical_solution_U
            , Function<dim> *_analytical_solution_V
            , Function<dim> &_initial_condition_U
            , Function<dim> &_initial_condition_V
            , Function<dim> &_boundary_U
            , Function<dim> &_boundary_V
            , const double _left_end
            , const double _right_end
            , const double _wave_speed
            , const double _alpha
            , const enum BC _bc
            , const double _final_time
            , double _time
            , const unsigned int _output_timestep_skip);

      void run();
      double getStepSize();

   private:
      ConditionalOStream pcout;

      void make_grid_and_dofs();
      void output_results(const unsigned int timestep_number);

#ifdef DEAL_II_WITH_P4EST
      parallel::distributed::Triangulation<dim> triangulation;
#else
      Triangulation<dim> triangulation;
#endif
      Function<dim> *analytical_solution_U;
      Function<dim> *analytical_solution_V;
      Function<dim> &initial_condition_U;
      Function<dim> &initial_condition_V;
      Function<dim> &boundary_U;
      Function<dim> &boundary_V;
      const double left_end, right_end;
      const double wave_speed;
      const double alpha;
      const enum BC bc;

      FE_DGQ<dim> fe;
      DoFHandler<dim> dof_handler;
      IndexSet locally_relevant_dofs;
      MatrixFree<dim, double, VectorizedArray<double>> matrix_free_data;
      LinearAlgebra::distributed::Vector<double> solution_U;
      LinearAlgebra::distributed::Vector<double> solution_V;
      bool set_time_step;
      double             time, time_step;
      const double       final_time;
      const double       cfl_number;
      const unsigned int output_timestep_skip;
   };



   template <int dim, int fe_degree>
   KleinGordonProblem<dim, fe_degree>::KleinGordonProblem(
         Function<dim> *_analytical_solution_U
         , Function<dim> *_analytical_solution_V
         , Function<dim> &_initial_condition_U
         , Function<dim> &_initial_condition_V
         , Function<dim> &_boundary_U
         , Function<dim> &_boundary_V
         , const double _left_end
         , const double _right_end
         , double _wave_speed
         , double _alpha
         , enum BC _bc
         , double _time
         , double _final_time
         , const unsigned int _output_timestep_skip)
   : pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      ,
#ifdef DEAL_II_WITH_P4EST
      triangulation(MPI_COMM_WORLD)
         ,
#endif
      analytical_solution_U(_analytical_solution_U)
         , analytical_solution_V(_analytical_solution_V)
         , initial_condition_U(_initial_condition_U)
         , initial_condition_V(_initial_condition_V)
         , boundary_U(_boundary_U)
         , boundary_V(_boundary_V)
         , left_end(_left_end)
         , right_end(_right_end)
         , wave_speed(_wave_speed)
         , alpha(_alpha)
         , bc(_bc)
         , fe(fe_degree)
         , dof_handler(triangulation)
         , set_time_step(false)
         , time(_time)
         , final_time(_final_time)
         , cfl_number(.1 / fe_degree)
         , output_timestep_skip(_output_timestep_skip)
         {}



   template <int dim, int fe_degree>
   void KleinGordonProblem<dim, fe_degree>::make_grid_and_dofs()
   {
      GridGenerator::subdivided_hyper_cube(triangulation, num_cells, left_end, right_end);
      const double cell_diameter = triangulation.last()->diameter() / std::sqrt(dim);
      time_step = 0.25 * cfl_number * cell_diameter;
      set_time_step= true;

      {
         std::string filepath = "results/grid"+ std::to_string(dim) + "d.vtk";
         std::ofstream file(filepath);
         GridOut grid_out;
         grid_out.write_vtk(triangulation, file);
         std::cout << "Grid written to " << filepath << std::endl;

         pcout << "   Number of global active cells for Volume: "
#ifdef DEAL_II_WITH_P4EST
            << triangulation.n_global_active_cells()
#else
            << triangulation.n_active_cells()
#endif
            << std::endl;
         dof_handler.distribute_dofs(fe);
         pcout << "   Number of degrees of freedom for Volume: " << dof_handler.n_dofs()
            << std::endl << std::endl;
      }

      AffineConstraints<double> dummy; dummy.close();
      typename MatrixFree<dim, double, VectorizedArray<double>>::AdditionalData additional_data;
      const MappingQGeneric<dim> mapping(fe_degree);

      additional_data.tasks_parallel_scheme =
         MatrixFree<dim, double, VectorizedArray<double>>::AdditionalData::TasksParallelScheme::none;

      additional_data.mapping_update_flags = (update_gradients | update_JxW_values |
                                              update_quadrature_points | update_values);
      additional_data.mapping_update_flags_inner_faces = (update_JxW_values | update_normal_vectors |
                                                          update_quadrature_points | update_values);
      additional_data.mapping_update_flags_boundary_faces = (update_JxW_values | update_normal_vectors |
                                                             update_quadrature_points | update_values);
      matrix_free_data.reinit(mapping, dof_handler, dummy,
                              QGaussLobatto<1>(fe_degree + 1), additional_data);

      matrix_free_data.initialize_dof_vector(solution_U);
      matrix_free_data.initialize_dof_vector(solution_V);
   }


   template <int dim, int fe_degree>
   void KleinGordonProblem<dim, fe_degree>::output_results(const unsigned int /*timestep_number*/)
   {

      Vector<float> norm_per_cell_U(triangulation.n_active_cells());
      Vector<float> norm_per_cell_V(triangulation.n_active_cells());

      solution_U.update_ghost_values();
      solution_V.update_ghost_values();

      if(analytical_solution_U != nullptr)
      {
         analytical_solution_U->set_time(time);
         analytical_solution_V->set_time(time);
         VectorTools::integrate_difference(
               dof_handler,
               solution_U,
               *analytical_solution_U,
               norm_per_cell_U,
               QGauss<dim>(fe_degree + 1),
               VectorTools::Linfty_norm);
         VectorTools::integrate_difference(
               dof_handler,
               solution_V,
               *analytical_solution_V,
               norm_per_cell_V,
               QGauss<dim>(fe_degree + 1),
               VectorTools::Linfty_norm);

         const double error_norm_U =
            VectorTools::compute_global_error(
                  triangulation,
                  norm_per_cell_U,
                  VectorTools::Linfty_norm);

         const double error_norm_V =
            VectorTools::compute_global_error(
                  triangulation,
                  norm_per_cell_V,
                  VectorTools::Linfty_norm);

         pcout << "Time: " << std::setw(4) << std::setprecision(3) << time
            << ", error norm U: " << std::setprecision(5) << std::setw(7)
            << error_norm_U << std::endl;

         pcout<< "Time: " << std::setw(4) << std::setprecision(3) << time
            << ", error norm V: " << std::setprecision(5) << std::setw(7)
            << error_norm_V << std::endl;

         pcout<< std::endl;
      }
      else
      {
         VectorTools::integrate_difference(
               dof_handler,
               solution_U,
               Functions::ZeroFunction<dim>(),
               norm_per_cell_U,
               QGauss<dim>(fe_degree + 1),
               VectorTools::Linfty_norm);

         VectorTools::integrate_difference(
               dof_handler,
               solution_V,
               Functions::ZeroFunction<dim>(),
               norm_per_cell_V,
               QGauss<dim>(fe_degree + 1),
               VectorTools::Linfty_norm);

         const double solution_norm_U =
            VectorTools::compute_global_error(
                  triangulation,
                  norm_per_cell_U,
                  VectorTools::Linfty_norm);

         const double solution_norm_V =
            VectorTools::compute_global_error(
                  triangulation,
                  norm_per_cell_V,
                  VectorTools::Linfty_norm);

         pcout << "Time: " << std::setw(4) << std::setprecision(3) << time
            << ", norm U: " << std::setprecision(5) << std::setw(7)
            << solution_norm_U<< std::endl;

         pcout << "Time: " << std::setw(4) << std::setprecision(3) << time
            << ", norm V: " << std::setprecision(5) << std::setw(7)
            << solution_norm_V<< std::endl;

         pcout<< std::endl;
      }

   }



   template <int dim, int fe_degree>
   void KleinGordonProblem<dim, fe_degree>::run()
   {

      {
         pcout << "Number of MPI ranks:            "
                << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) << std::endl;
         pcout << "Number of threads on each rank: "
                << MultithreadInfo::n_threads() << std::endl;
         const unsigned int n_vect_doubles = VectorizedArray<double>::size();
         const unsigned int n_vect_bits    = 8 * sizeof(double) * n_vect_doubles;
         pcout << "Vectorization over " << n_vect_doubles
                << " doubles = " << n_vect_bits << " bits ("
                << Utilities::System::get_current_vectorization_level() << ")"
                << std::endl
                << std::endl;
      }

      make_grid_and_dofs();

      const double cell_diameter = triangulation.last()->diameter() / std::sqrt(dim);
      pcout << "   Time step size: " << time_step
             << ", cell width: " << cell_diameter << std::endl
               << std::endl;

      VectorTools::interpolate(dof_handler, initial_condition_U, solution_U);
      VectorTools::interpolate(dof_handler, initial_condition_V, solution_V);
      const unsigned int size = solution_U.size();

      KleinGordonOperation<dim, fe_degree_global> klein_gordon_op(
            boundary_U, boundary_V, wave_speed, alpha,
            matrix_free_data, bc, size);

      unsigned int timestep_number = 0;
      Timer  timer;
      double wtime       = 0;
      double output_time = 0;

      for (; time <= final_time; time += time_step, ++timestep_number)
      {
         timer.restart();
         if (timestep_number % output_timestep_skip == 0)
         {
            output_results(timestep_number / output_timestep_skip);
         }
         output_time += timer.wall_time();

         timer.restart();
         klein_gordon_op.RK4_step(solution_U, solution_V, time, time_step);
         wtime += timer.wall_time();
      }

      timer.restart();
      output_results(timestep_number / output_timestep_skip + 1);
      output_time += timer.wall_time();

      pcout << std::endl << "   Performed " << timestep_number << " time steps." << std::endl;
      pcout << "   Average wallclock time per time step: " << wtime / timestep_number << "s" << std::endl;
      pcout << "   Spent " << output_time << "s on output and " << wtime << "s on computations." << std::endl;
   }



   template <int dim, int fe_degree>
   double KleinGordonProblem<dim, fe_degree>::getStepSize()
   {
      if (set_time_step == false) {
         std::cerr << "Error: function called before step size was computed. "
                      "'run' function should be called first." << std::endl;
         return 0.0;
      }

      return this->time_step;
   }

} // namespace KGE

#endif
