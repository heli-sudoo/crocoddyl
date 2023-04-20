///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_SOLVERS_FDDP2_HPP_
#define CROCODDYL_CORE_SOLVERS_FDDP2_HPP_

#include <Eigen/Cholesky>
#include <vector>

#include "crocoddyl/core/solvers/fddp.hpp"

namespace crocoddyl {

/**
 * @brief FDDP with improved expected cost change
 */
class SolverFDDP2 : public SolverFDDP {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief Initialize the FDDP solver
   *
   * @param[in] problem  shooting problem
   */
  explicit SolverFDDP2(boost::shared_ptr<ShootingProblem> problem);
  virtual ~SolverFDDP2();

  virtual bool solve(const std::vector<Eigen::VectorXd>& init_xs = DEFAULT_VECTOR,
                     const std::vector<Eigen::VectorXd>& init_us = DEFAULT_VECTOR, const std::size_t maxiter = 100,
                     const bool is_feasible = false, const double regInit = 1e-9);

  void linear_forward_rollout();
  
 private:
  double th_acceptnegstep_;  //!< Threshold used for accepting step along ascent direction
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_SOLVERS_FDDP2_HPP_
