#include <crocoddyl/core/solver-base.hpp>

namespace crocoddyl {

SolverAbstract::SolverAbstract(ShootingProblem& problem) : problem_(problem), is_feasible_(false),
    cost_(0.), stop_(0.), xreg_(NAN), ureg_(NAN), steplength_(1.), dV_(0.), dVexp_(0.),
    th_acceptstep_(0.1), th_stop_(1e-9), iter_(0) {}

SolverAbstract::~SolverAbstract() {}

void SolverAbstract::setCandidate(const std::vector<Eigen::VectorXd>& xs_warm,
                                  const std::vector<Eigen::VectorXd>& us_warm,
                                  const bool& is_feasible) {
  const long unsigned int& T = problem_.get_T();

  if (xs_warm.size() == 0) {
    for (long unsigned int t = 0; t < T; ++t) {
      xs_[t] = problem_.running_models_[t]->get_state()->zero();
    }
    xs_.back() = problem_.terminal_model_->get_state()->zero();
  } else {
    assert(xs_warm.size()==T+1);
    std::copy(xs_warm.begin(), xs_warm.end(), xs_.begin());
  }

  if(us_warm.size() == 0) {
    for (long unsigned int t = 0; t < T; ++t) {
      const int& nu = problem_.running_models_[t]->get_nu();
      us_[t] = Eigen::VectorXd::Zero(nu);
    }
  } else {
    assert(us_warm.size()==T);
    std::copy(us_warm.begin(), us_warm.end(), us_.begin());
  }
  is_feasible_ = is_feasible;
}

void SolverAbstract::setCallbacks(std::vector<CallbackAbstract*>& callbacks) {
  callbacks_ = callbacks;
}

const bool& SolverAbstract::get_isFeasible() const {
  return is_feasible_;
}

const unsigned int& SolverAbstract::get_iter() const {
  return iter_;
}

const double& SolverAbstract::get_cost() const {
  return cost_;
}

const double& SolverAbstract::get_stop() const {
  return stop_;
}

const Eigen::Vector2d& SolverAbstract::get_d() const {
  return d_;
}

const double& SolverAbstract::get_xreg() const {
  return xreg_;
}

const double& SolverAbstract::get_ureg() const {
  return ureg_;
}

const double& SolverAbstract::get_stepLength() const {
  return steplength_;
}

const double& SolverAbstract::get_dV() const {
  return dV_;
}

const double& SolverAbstract::get_dVexp() const {
  return dVexp_;
}

}  // namespace crocoddyl