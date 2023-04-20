///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifdef CROCODDYL_WITH_MULTITHREADING
#include <omp.h>
#endif // CROCODDYL_WITH_MULTITHREADING

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/solvers/fddp2.hpp"

namespace crocoddyl
{

  SolverFDDP2::SolverFDDP2(boost::shared_ptr<ShootingProblem> problem)
      : SolverFDDP(problem), th_acceptnegstep_(2) {}

  SolverFDDP2::~SolverFDDP2() {}

  bool SolverFDDP2::solve(const std::vector<Eigen::VectorXd> &init_xs, const std::vector<Eigen::VectorXd> &init_us,
                         const std::size_t maxiter, const bool is_feasible, const double reginit)
  {
    START_PROFILER("SolverFDDP::solve");
    if (problem_->is_updated())
    {
      resizeData();
    }
    xs_try_[0] = problem_->get_x0(); // it is needed in case that init_xs[0] is infeasible
    setCandidate(init_xs, init_us, is_feasible);

    if (std::isnan(reginit))
    {
      xreg_ = reg_min_;
      ureg_ = reg_min_;
    }
    else
    {
      xreg_ = reginit;
      ureg_ = reginit;
    }
    was_feasible_ = false;

    bool recalcDiff = true;
    for (iter_ = 0; iter_ < maxiter; ++iter_)
    {
      while (true)
      {
        try
        {
          computeDirection(recalcDiff);
        }
        catch (std::exception &e)
        {
          recalcDiff = false;
          increaseRegularization();
          if (xreg_ == reg_max_)
          {
            return false;
          }
          else
          {
            continue;
          }
        }
        break;
      }

      if (is_feasible_)
      {
        updateExpectedImprovement();
        d_[0] = dg_;
        d_[1] = dq_;        
      }else
      {
        linear_forward_rollout();
      }
      
      // linear_forward_rollout();
      
      
      // We need to recalculate the derivatives when the step length passes
      recalcDiff = false;
      for (std::vector<double>::const_iterator it = alphas_.begin(); it != alphas_.end(); ++it)
      {
        steplength_ = *it;

        try
        {
          dV_ = tryStep(steplength_);
        }
        catch (std::exception &e)
        {
          continue;
        }
        
        dVexp_ = steplength_ * (d_[0] + 0.5 * steplength_ * d_[1]);
        
        if (dVexp_ >= 0)
        { // descend direction
          if (d_[0] < th_grad_ || dV_ > th_acceptstep_ * dVexp_)
          {
            was_feasible_ = is_feasible_;
            setCandidate(xs_try_, us_try_, (was_feasible_) || (steplength_ == 1));
            cost_ = cost_try_;
            recalcDiff = true;
            break;
          }
        }
        else
        { // reducing the gaps by allowing a small increment in the cost value
          if (dV_ > th_acceptnegstep_ * dVexp_)
          {
            was_feasible_ = is_feasible_;
            setCandidate(xs_try_, us_try_, (was_feasible_) || (steplength_ == 1));
            cost_ = cost_try_;
            recalcDiff = true;
            break;
          }
        }
      }

      if (steplength_ > th_stepdec_)
      {
        decreaseRegularization();
      }
      if (steplength_ <= th_stepinc_)
      {
        increaseRegularization();
        if (xreg_ == reg_max_)
        {
          STOP_PROFILER("SolverFDDP::solve");
          return false;
        }
      }
      stoppingCriteria();

      const std::size_t n_callbacks = callbacks_.size();
      for (std::size_t c = 0; c < n_callbacks; ++c)
      {
        CallbackAbstract &callback = *callbacks_[c];
        callback(*this);
      }

      if (was_feasible_ && stop_ < th_stop_)
      {
        STOP_PROFILER("SolverFDDP::solve");
        return true;
      }
    }
    STOP_PROFILER("SolverFDDP::solve");
    return false;
  }

  void SolverFDDP2::linear_forward_rollout()
  {
    d_.setZero();

    const std::size_t T = problem_->get_T();
    const std::vector<boost::shared_ptr<ActionDataAbstract>> &datas = problem_->get_runningDatas();

    Eigen::VectorXd dx;

    dx.setZero(problem_->get_nx());

    for (std::size_t t = 0; t < T; t++)
    {      
      const boost::shared_ptr<ActionDataAbstract> &d = datas[t];
      
      const auto& At = d->Fx;
      const auto& Bt = d->Fu;      

      const auto& du= -k_[t] - K_[t] * dx;
      const auto& dx_next = At * dx + Bt * du + fs_[t+1];

      d_[0] += d->Lu.transpose()*du;
      d_[0] += d->Lx.transpose()*dx;
      d_[1] += du.transpose() * d->Luu* du;
      d_[1] += dx.transpose() * d->Lxx* dx;
      d_[1] += dx.transpose() * d->Lxu* du;

      dx = dx_next;
    }    

    const boost::shared_ptr<ActionDataAbstract> &d = problem_->get_terminalData();
    d_[0] += d->Lx.transpose()*dx;
    d_[1] += dx.transpose() * d->Lxx* dx;    

    d_ = -d_;
  }
} // namespace crocoddyl
