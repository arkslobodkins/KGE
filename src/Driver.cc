/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2011 - 2020 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 * Author: Katharina Kormann, Martin Kronbichler, Uppsala University, 2011-2012 */

/* Modified by Arkadijs Slobodkins
 * SMU
 * Summer 2021
 */

#if __cplusplus <= 199711L
#error Requires support of C++11 or newer
#else

#include "RunKGE.h"
#include "ProblemParams.h"

#include <fstream>
#include <iostream>
#include <iomanip>


int main(int argc, char **argv)
{
   using namespace KGE;
   using namespace dealii;

   Utilities::MPI::MPI_InitFinalize mpi_initialization(
         argc, argv, numbers::invalid_unsigned_int);

   try
   {
      RunKGE<dim_global, fe_degree_global>();
   }
   catch (std::exception &exc)
   {
      std::cerr << std::endl
         << std::endl
         << "----------------------------------------------------"
         << std::endl;
      std::cerr << "Exception on processing: " << std::endl
         << exc.what() << std::endl
         << "Aborting!" << std::endl
         << "----------------------------------------------------"
         << std::endl;

      return 1;
   }
   catch (...)
   {
      std::cerr << std::endl
         << std::endl
         << "----------------------------------------------------"
         << std::endl;
      std::cerr << "Unknown exception!" << std::endl
         << "Aborting!" << std::endl
         << "----------------------------------------------------"
         << std::endl;
      return 1;
   }

   return 0;
}

#endif
