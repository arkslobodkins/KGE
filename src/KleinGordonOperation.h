/* Arkadijs Slobodkins
 * Summer 2021
 * Based on steps 48 and 59 from dealii examples
 */

#ifndef KG_OP
#define KG_OP

#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/base/time_stepping.h>

#include <fstream>
#include <unistd.h>

namespace KGE
{
   using namespace dealii;
   using namespace LinearAlgebra;

   template<int dim>
   VectorizedArray<double> eval_function(const Function<dim> &function,
                                         const Point<dim, VectorizedArray<double>> &p_vectorized,
                                         const unsigned int component)
   {
      VectorizedArray<double> result;
      for(unsigned int v = 0; v < VectorizedArray<double>::size(); ++v)
      {
         Point<dim>p;
         for(unsigned int d = 0; d < dim; ++d) p[d] = p_vectorized[d][v];
         result[v] = function.value(p, component);
      }
      return result;
   }



   template<int dim, int fe_degree>
   class KleinGordonOperation
   {
   public:
      KleinGordonOperation() = delete; // cannot use default constructor
      KleinGordonOperation(
            Function<dim> &_boundary_u,
            Function<dim> &_boundary_v,
            const double _wave_speed,
            const double _alpha,
            const MatrixFree<dim, double, VectorizedArray<double>> &data_in_VOL,
            const enum BC _bc,
            const unsigned int _size);

      ~KleinGordonOperation();

      void RK4_step(
            distributed::Vector<double> & update_U_VOL,
            distributed::Vector<double> & update_V_VOL,
            double current_time, double step_size) const;

   private:
      Function<dim> &boundary_u;
      Function<dim> &boundary_v;
      const double wave_speed;
      const double wave_speed_squared;
      const double alpha;
      const double alpha_squared;
      const MatrixFree<dim, double, VectorizedArray<double>> &data;
      const enum BC bc;
      const unsigned int size;

      TimeStepping::runge_kutta_method method;
      TimeStepping::ExplicitRungeKutta<distributed::Vector<double>> *explicit_runge_kutta;
      distributed::Vector<double> compute_rhs_coupled(
            const double time,
            const distributed::Vector<double> &y) const;

      void local_apply_inverse_mass_matrix(
            const MatrixFree<dim, double, VectorizedArray<double>> &data,
            LinearAlgebra::distributed::Vector<double> &dst,
            const LinearAlgebra::distributed::Vector<double> &src,
            const std::pair<unsigned int, unsigned int> &cell_range) const;

      void apply_cell(
            const MatrixFree<dim, double, VectorizedArray<double>> &data,
            std::vector<distributed::Vector<double> *> &dst,
            const std::vector<distributed::Vector<double> *> &src,
            const std::pair<unsigned int, unsigned int> &cell_range) const;

      void apply_face(
            const MatrixFree<dim, double, VectorizedArray<double>> &data,
            std::vector<distributed::Vector<double> *> &dst,
            const std::vector<distributed::Vector<double> *> &src,
            const std::pair<unsigned int, unsigned int> &face_range) const;

      void apply_boundary(
            const MatrixFree<dim, double, VectorizedArray<double>> &data,
            std::vector<distributed::Vector<double> *> &dst,
            const std::vector<distributed::Vector<double> *> &src,
            const std::pair<unsigned int, unsigned int> &face_range) const;

      double get_penalty_factor() const
      {
         return 1.0 * fe_degree * (fe_degree + 1);
      }

   };



   template <int dim, int fe_degree>
   KleinGordonOperation<dim, fe_degree>::KleinGordonOperation(
         Function<dim> &_boundary_u,
         Function<dim> &_boundary_v,
         const double _wave_speed,
         const double _alpha,
         const MatrixFree<dim, double, VectorizedArray<double>> &data_in_VOL,
         enum BC _bc,
         const unsigned int _size)
   :boundary_u(_boundary_u),
   boundary_v(_boundary_v),
   wave_speed(_wave_speed),
   wave_speed_squared(square(_wave_speed)),
   alpha(_alpha),
   alpha_squared(square(_alpha)),
   data(data_in_VOL),
   bc(_bc),
   size(_size)
   {
      method = TimeStepping::RK_CLASSIC_FOURTH_ORDER;
      explicit_runge_kutta = new TimeStepping::ExplicitRungeKutta<distributed::Vector<double>>(method);
   }



   template<int dim, int fe_degree>
   KleinGordonOperation<dim, fe_degree>::~KleinGordonOperation()
   {
      delete explicit_runge_kutta;
   }



   template <int dim, int fe_degree>
   void KleinGordonOperation<dim, fe_degree>::local_apply_inverse_mass_matrix(
         const MatrixFree<dim, double, VectorizedArray<double>> &,
         LinearAlgebra::distributed::Vector<double> &      dst,
         const LinearAlgebra::distributed::Vector<double> &src,
         const std::pair<unsigned int, unsigned int> &     cell_range) const
   {
      FEEvaluation<dim, fe_degree, fe_degree+1, 1, double > phi(data);
      MatrixFreeOperators::CellwiseInverseMassMatrix<dim, fe_degree> inverse(phi);
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
         phi.reinit(cell);
         phi.read_dof_values(src);
         inverse.apply(phi.begin_dof_values(), phi.begin_dof_values());
         phi.set_dof_values(dst);
      }
   }



   template <int dim, int fe_degree>
   void KleinGordonOperation<dim, fe_degree>::apply_cell(
         const MatrixFree<dim, double, VectorizedArray<double>> &,
         std::vector<distributed::Vector<double> *> &dst,
         const std::vector<distributed::Vector<double> *> &src,
         const std::pair<unsigned int, unsigned int> &cell_range) const
   {
      FEEvaluation<dim, fe_degree, fe_degree+1, 1, double > phi_V(data);
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
         phi_V.reinit(cell);
         phi_V.gather_evaluate((*src[0]), true, true, false);

         for(unsigned int q = 0; q < phi_V.n_q_points; ++q)
            phi_V.submit_value(-alpha_squared * phi_V.get_value(q), q);

         for(unsigned int q = 0; q < phi_V.n_q_points; ++q)
            phi_V.submit_gradient(-1.0 * phi_V.get_gradient(q) * wave_speed_squared, q);

         phi_V.integrate_scatter(true, true, (*dst[1]));
      }
   }



   template<int dim, int fe_degree>
   void KleinGordonOperation<dim, fe_degree>::apply_face(
         const MatrixFree<dim, double, VectorizedArray<double>> &,
         std::vector<distributed::Vector<double> *> &dst,
         const std::vector<distributed::Vector<double> *> &src,
         const std::pair<unsigned int, unsigned int> &face_range) const
   {
      FEFaceEvaluation<dim, fe_degree, fe_degree+1, 1, double> phi_inner_U(data, true);
      FEFaceEvaluation<dim, fe_degree, fe_degree+1, 1, double> phi_outer_U(data, false);
      FEFaceEvaluation<dim, fe_degree, fe_degree+1, 1, double> phi_inner_V(data, true);
      FEFaceEvaluation<dim, fe_degree, fe_degree+1, 1, double> phi_outer_V(data, false);

      for(unsigned int face = face_range.first; face < face_range.second; ++face)
      {
         phi_inner_U.reinit(face); phi_inner_U.gather_evaluate(*src[0], true, true);
         phi_outer_U.reinit(face); phi_outer_U.gather_evaluate(*src[0], true, true);
         phi_inner_V.reinit(face); phi_inner_V.gather_evaluate(*src[1], true, true);
         phi_outer_V.reinit(face); phi_outer_V.gather_evaluate(*src[1], true, true);

         const VectorizedArray<double> inverse_length_normal_to_face =
            0.5 * ( std::abs((phi_inner_U.get_normal_vector(0) * phi_inner_U.inverse_jacobian(0))[dim - 1])
                  + std::abs((phi_outer_U.get_normal_vector(0) * phi_outer_U.inverse_jacobian(0))[dim - 1]) );

         const VectorizedArray<double> sigma = inverse_length_normal_to_face * get_penalty_factor();

         for(unsigned int q = 0; q < phi_inner_U.n_q_points; ++q)
         {
            const VectorizedArray<double> solution_jump_U = (phi_inner_U.get_value(q) - phi_outer_U.get_value(q));
            const VectorizedArray<double> solution_jump_V = (phi_inner_V.get_value(q) - phi_outer_V.get_value(q));
            const VectorizedArray<double> average_normal_derivative_U = (phi_inner_U.get_normal_derivative(q) + phi_outer_U.get_normal_derivative(q)) * 0.5;
            const VectorizedArray<double> what = average_normal_derivative_U - solution_jump_V * 0.5 / wave_speed;
            const VectorizedArray<double> boundary_terms = solution_jump_U * sigma - what;

            phi_inner_U.submit_value(-boundary_terms * wave_speed_squared, q);
            phi_outer_U.submit_value(+boundary_terms * wave_speed_squared, q);
            phi_inner_U.submit_normal_derivative(+solution_jump_U * wave_speed_squared, q);
            phi_outer_U.submit_normal_derivative(+solution_jump_U * wave_speed_squared, q);
         }
         phi_inner_U.integrate_scatter(true, true, *dst[1]);
         phi_outer_U.integrate_scatter(true, true, *dst[1]);
      }
   }



   template<int dim, int fe_degree>
   void KleinGordonOperation<dim, fe_degree>::apply_boundary(
         const MatrixFree<dim, double, VectorizedArray<double>> &,
         std::vector<distributed::Vector<double> *> &dst,
         const std::vector<distributed::Vector<double> *> &src,
         const std::pair<unsigned int, unsigned int> &face_range) const
   {
      FEFaceEvaluation<dim, fe_degree, fe_degree+1> phi_inner_U(data, true);
      FEFaceEvaluation<dim, fe_degree, fe_degree+1> phi_inner_V(data, true);

      for(unsigned int face = face_range.first; face < face_range.second; ++face)
      {
         phi_inner_U.reinit(face); phi_inner_U.gather_evaluate(*src[0], true, true);
         phi_inner_V.reinit(face); phi_inner_V.gather_evaluate(*src[1], true, true);

         const VectorizedArray<double> inverse_length_normal_to_face =
            std::abs((phi_inner_U.get_normal_vector(0) * phi_inner_U.inverse_jacobian(0))[dim - 1]);
         const VectorizedArray<double> sigma = inverse_length_normal_to_face * get_penalty_factor();

         for(unsigned int q = 0; q < phi_inner_U.n_q_points; ++q)
         {
            const VectorizedArray<double> u_inner = phi_inner_U.get_value(q);
            const VectorizedArray<double> v_inner = phi_inner_V.get_value(q);
            const VectorizedArray<double> normal_der_inner_U = phi_inner_U.get_normal_derivative(q);
            VectorizedArray<double> u_outer;
            VectorizedArray<double> v_outer;
            VectorizedArray<double> normal_der_outer_U;

            switch(bc)
            {
               case(DIRICHLET):
                  u_outer = -1.0 * u_inner + 2.0 * eval_function<dim>(boundary_u, phi_inner_U.quadrature_point(q), 0);
                  v_outer = -1.0 * v_inner + 2.0 * eval_function<dim>(boundary_v, phi_inner_V.quadrature_point(q), 0);
                  normal_der_outer_U = normal_der_inner_U + 2.0 / wave_speed * (eval_function<dim>(boundary_v, phi_inner_V.quadrature_point(q), 0) - v_inner);
                  break;

               case(NEUMANN):
                  u_outer = u_inner;
                  v_outer = v_inner + 2.0 * (eval_function<dim>(boundary_u, phi_inner_U.quadrature_point(q), 0) - normal_der_inner_U);
                  normal_der_outer_U = -1.0 * normal_der_inner_U + 2.0 * eval_function<dim>(boundary_u, phi_inner_U.quadrature_point(q), 0);
                  break;

               case(SOMMERFELD):
                  u_outer = u_inner;
                  v_outer = v_inner;
                  normal_der_outer_U = +normal_der_inner_U + 2.0 * (-v_inner - normal_der_inner_U + eval_function<dim>(boundary_u, phi_inner_U.quadrature_point(q), 0));
                  break;
            }

            const VectorizedArray<double> average_normal_derivative_U = 0.5 * (normal_der_inner_U + normal_der_outer_U);
            const VectorizedArray<double> jump_U = (u_inner - u_outer);
            const VectorizedArray<double> jump_V = (v_inner - v_outer);
            const VectorizedArray<double> what = + average_normal_derivative_U - jump_V * 0.5 / wave_speed;
            const VectorizedArray<double> boundary_terms = jump_U * sigma - what;

            phi_inner_U.submit_value(-boundary_terms * wave_speed_squared, q);
            phi_inner_U.submit_normal_derivative(+jump_U * 0.5 * wave_speed_squared, q);
         }
         phi_inner_U.integrate_scatter(true, true, *dst[1]);
      }
   }



   template<int dim, int fe_degree>
   distributed::Vector<double> KleinGordonOperation<dim, fe_degree>::compute_rhs_coupled(
         const double t,const distributed::Vector<double> &y)const
   {
      unsigned int i;
      distributed::Vector<double> update_U, update_V, current_U, current_V;
      current_U.reinit(size); current_V.reinit(size);
      update_U.reinit(size); update_V.reinit(size);

      for(i = 0; i < size; ++i) current_U(i) = y(i);
      for(i = 0; i < size; ++i) current_V(i) = y(size+i);

      std::vector<distributed::Vector<double> *> current({&current_U, &current_V});
      std::vector<distributed::Vector<double> *> update({&update_U, &update_V});

      boundary_v.set_time(t);
      boundary_u.set_time(t);
      data.loop(&KleinGordonOperation::apply_cell,
            &KleinGordonOperation::apply_face,
            &KleinGordonOperation::apply_boundary,
            this,
            update,
            current,
            true,
            MatrixFree<dim, double, VectorizedArray<double>>::DataAccessOnFaces::gradients,
            MatrixFree<dim, double, VectorizedArray<double>>::DataAccessOnFaces::gradients);

      data.cell_loop(&KleinGordonOperation::local_apply_inverse_mass_matrix, this, update_V, update_V);
      update_U = current_V;

      distributed::Vector<double> ret_update; ret_update.reinit(y.size());
      for(i = 0; i < size; ++i) ret_update(i) = update_U(i);
      for(i = 0; i < size; ++i) ret_update(size+i) = update_V(i);

      return ret_update;
   }



   template <int dim, int fe_degree>
   void KleinGordonOperation<dim, fe_degree>::RK4_step(
         distributed::Vector<double> &update_U,
         distributed::Vector<double> &update_V,
         double current_time, double step_size) const
   {
      Assert(update_U.size() == update_V.size(), ExcMessage("Vectors U and V must be of the same length."));
      unsigned int i;

      distributed::Vector<double> update; update.reinit(2*size);
      for(i = 0; i < size; ++i) update(i) = update_U(i);
      for(i = 0; i < size; ++i) update(size + i) = update_V(i);

      explicit_runge_kutta->evolve_one_time_step([this]
            (const double t, const distributed::Vector<double> & y) {
            return this->compute_rhs_coupled(t, y);
            }, current_time, step_size, update);

      for(i = 0; i < size; ++i) update_U(i) = update(i);
      for(i = 0; i < size; ++i) update_V(i) = update(size + i);

   }

} // namespace KGE

#endif
