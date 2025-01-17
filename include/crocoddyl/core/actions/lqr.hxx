///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

template <typename Scalar>
ActionModelLQRTpl<Scalar>::ActionModelLQRTpl(const std::size_t nx, const std::size_t nu, const bool drift_free)
    : Base(boost::make_shared<StateVector>(nx), nu, 0), drift_free_(drift_free) {
  // TODO(cmastalli): substitute by random (vectors) and random-orthogonal (matrices)
  Fx_ = MatrixXs::Identity(nx, nx);
  Fu_ = MatrixXs::Identity(nx, nu);
  f0_ = VectorXs::Ones(nx);
  Lxx_ = MatrixXs::Identity(nx, nx);
  Lxu_ = MatrixXs::Identity(nx, nu);
  Luu_ = MatrixXs::Identity(nu, nu);
  lx_ = VectorXs::Ones(nx);
  lu_ = VectorXs::Ones(nu);
}

template <typename Scalar>
ActionModelLQRTpl<Scalar>::~ActionModelLQRTpl() {}

template <typename Scalar>
void ActionModelLQRTpl<Scalar>::calc(const boost::shared_ptr<ActionDataAbstract>& data,
                                     const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  Data* d = static_cast<Data*>(data.get());

  if (drift_free_) {
    data->xnext.noalias() = Fx_ * x;
    data->xnext.noalias() += Fu_ * u;
  } else {
    data->xnext.noalias() = Fx_ * x;
    data->xnext.noalias() += Fu_ * u;
    data->xnext += f0_;
  }

  // cost = 0.5 * x^T*Lxx*x + 0.5 * u^T*Luu*u + x^T*Lxu*u + lx^T*x + lu^T*u
  d->Lxx_x_tmp.noalias() = Lxx_ * x;
  data->cost = Scalar(0.5) * x.dot(d->Lxx_x_tmp);
  d->Luu_u_tmp.noalias() = Luu_ * u;
  data->cost += Scalar(0.5) * u.dot(d->Luu_u_tmp);
  d->Lxx_x_tmp.noalias() = Lxu_ * u;
  data->cost += x.dot(d->Lxx_x_tmp);
  data->cost += lx_.transpose() * x;
  data->cost += lu_.transpose() * u;
}

template <typename Scalar>
void ActionModelLQRTpl<Scalar>::calc(const boost::shared_ptr<ActionDataAbstract>& data,
                                     const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  Data* d = static_cast<Data*>(data.get());

  // cost = 0.5 * x^T*Lxx*x + lx^T*x
  d->Lxx_x_tmp.noalias() = Lxx_ * x;
  data->cost = Scalar(0.5) * x.dot(d->Lxx_x_tmp);
  data->cost += lx_.dot(x);
}

template <typename Scalar>
void ActionModelLQRTpl<Scalar>::calcDiff(const boost::shared_ptr<ActionDataAbstract>& data,
                                         const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }

  data->Lx = lx_;
  data->Lx.noalias() += Lxx_ * x;
  data->Lx.noalias() += Lxu_ * u;
  data->Lu = lu_;
  data->Lu.noalias() += Lxu_.transpose() * x;
  data->Lu.noalias() += Luu_ * u;
  data->Fx = Fx_;
  data->Fu = Fu_;
  data->Lxx = Lxx_;
  data->Lxu = Lxu_;
  data->Luu = Luu_;
}

template <typename Scalar>
void ActionModelLQRTpl<Scalar>::calcDiff(const boost::shared_ptr<ActionDataAbstract>& data,
                                         const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }

  data->Lx = lx_;
  data->Lx.noalias() += Lxx_ * x;
  data->Lxx = Lxx_;
}

template <typename Scalar>
boost::shared_ptr<ActionDataAbstractTpl<Scalar>> ActionModelLQRTpl<Scalar>::createData() {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
}

template <typename Scalar>
bool ActionModelLQRTpl<Scalar>::checkData(const boost::shared_ptr<ActionDataAbstract>& data) {
  boost::shared_ptr<Data> d = boost::dynamic_pointer_cast<Data>(data);
  if (d != NULL) {
    return true;
  } else {
    return false;
  }
}

template <typename Scalar>
void ActionModelLQRTpl<Scalar>::print(std::ostream& os) const {
  os << "ActionModelLQR {nx=" << state_->get_nx() << ", nu=" << nu_ << ", drift_free=" << drift_free_ << "}";
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixXs& ActionModelLQRTpl<Scalar>::get_Fx() const {
  return Fx_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixXs& ActionModelLQRTpl<Scalar>::get_Fu() const {
  return Fu_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs& ActionModelLQRTpl<Scalar>::get_f0() const {
  return f0_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs& ActionModelLQRTpl<Scalar>::get_lx() const {
  return lx_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs& ActionModelLQRTpl<Scalar>::get_lu() const {
  return lu_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixXs& ActionModelLQRTpl<Scalar>::get_Lxx() const {
  return Lxx_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixXs& ActionModelLQRTpl<Scalar>::get_Lxu() const {
  return Lxu_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixXs& ActionModelLQRTpl<Scalar>::get_Luu() const {
  return Luu_;
}

template <typename Scalar>
void ActionModelLQRTpl<Scalar>::set_Fx(const MatrixXs& Fx) {
  if (static_cast<std::size_t>(Fx.rows()) != state_->get_nx() ||
      static_cast<std::size_t>(Fx.cols()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "Fx has wrong dimension (it should be " + std::to_string(state_->get_nx()) + "," +
                        std::to_string(state_->get_nx()) + ")");
  }
  Fx_ = Fx;
}

template <typename Scalar>
void ActionModelLQRTpl<Scalar>::set_Fu(const MatrixXs& Fu) {
  if (static_cast<std::size_t>(Fu.rows()) != state_->get_nx() || static_cast<std::size_t>(Fu.cols()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "Fu has wrong dimension (it should be " + std::to_string(state_->get_nx()) + "," +
                        std::to_string(nu_) + ")");
  }
  Fu_ = Fu;
}

template <typename Scalar>
void ActionModelLQRTpl<Scalar>::set_f0(const VectorXs& f0) {
  if (static_cast<std::size_t>(f0.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "f0 has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  f0_ = f0;
}

template <typename Scalar>
void ActionModelLQRTpl<Scalar>::set_lx(const VectorXs& lx) {
  if (static_cast<std::size_t>(lx.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "lx has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  lx_ = lx;
}

template <typename Scalar>
void ActionModelLQRTpl<Scalar>::set_lu(const VectorXs& lu) {
  if (static_cast<std::size_t>(lu.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "lu has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  lu_ = lu;
}

template <typename Scalar>
void ActionModelLQRTpl<Scalar>::set_Lxx(const MatrixXs& Lxx) {
  if (static_cast<std::size_t>(Lxx.rows()) != state_->get_nx() ||
      static_cast<std::size_t>(Lxx.cols()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "Lxx has wrong dimension (it should be " + std::to_string(state_->get_nx()) + "," +
                        std::to_string(state_->get_nx()) + ")");
  }
  Lxx_ = Lxx;
}

template <typename Scalar>
void ActionModelLQRTpl<Scalar>::set_Lxu(const MatrixXs& Lxu) {
  if (static_cast<std::size_t>(Lxu.rows()) != state_->get_nx() || static_cast<std::size_t>(Lxu.cols()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "Lxu has wrong dimension (it should be " + std::to_string(state_->get_nx()) + "," +
                        std::to_string(nu_) + ")");
  }
  Lxu_ = Lxu;
}

template <typename Scalar>
void ActionModelLQRTpl<Scalar>::set_Luu(const MatrixXs& Luu) {
  if (static_cast<std::size_t>(Luu.rows()) != nu_ || static_cast<std::size_t>(Luu.cols()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "Fq has wrong dimension (it should be " + std::to_string(nu_) + "," + std::to_string(nu_) + ")");
  }
  Luu_ = Luu;
}

}  // namespace crocoddyl
