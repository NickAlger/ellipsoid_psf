#pragma once
// SPDX-License-Identifier: MIT
// Part of ellipsoid_psf — https://github.com/NickAlger/ellipsoid_psf

/// @file
/// @brief KernelEvaluator: the complete kernel approximation
/// Phi(y, x) ~ RBF-combine of per-neighbor predictions, single entries and
/// threaded blocks.
///
/// Cols-only mode uses one ImpulseResponseField (impulse responses are
/// "columns" of the kernel, sampled in x). Symmetric mode additionally takes
/// a row field probed with the transpose operator: since
/// Phi(y, x) = Phi^T(x, y), the row field's predictions at (x, y) predict the
/// same kernel entry from samples near y. Two combine rules
/// (EvalConfig::symmetric_combine):
///   - SymmetricCombine::pooled — both prediction sets are pooled in
///     displacement coordinates (sample point minus its query point, so the
///     evaluation point is the origin), near-duplicate centers are averaged,
///     and one RBF interpolation combines them — the paper's construction.
///   - SymmetricCombine::crisscross — one side is SELECTED per entry: the
///     forward (column) interpolation when x is closer to a col-field sample
///     than y is to a row-field sample, the adjoint (row) interpolation when
///     y is closer, the average of both on ties. Fits stay pure, known
///     columns/rows form an exact criss-cross lattice, and with the same
///     field passed twice the result is exactly symmetric by construction.
///
/// There is no special case snapping evaluations onto known columns: with
/// smoothing = 0 the RBF interpolant reproduces its centers, so
/// Phi(y, x_i) is exact at sample points automatically, and with
/// smoothing > 0 that exactness is deliberately traded for noise robustness.
///
/// Entries are scalar today; the planned vector/tensor extension adds
/// vector-valued entry methods additively — see dev/VECTOR_TENSOR_EXTENSION.md.

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "ellipsoid_tree/detail/parallel_for.hpp"
#include "ellipsoid_tree/geometry.hpp"

#include "ellipsoid_psf/config.hpp"
#include "ellipsoid_psf/impulse_response_field.hpp"
#include "ellipsoid_psf/rbf.hpp"

namespace ellipsoid_psf {

/// Evaluates the approximate integral kernel anywhere, by RBF interpolation
/// of the per-neighbor predictions of one (cols-only) or two (symmetric)
/// impulse response fields.
class KernelEvaluator
{
public:
    /// col_field is required; row_field may be null (cols-only) or a field
    /// probed with the transpose operator (symmetric mode; pass the same
    /// field twice for a symmetric operator probed once). Both fields are
    /// validated against `config` here, and the RBF scheme is validated too,
    /// so construction succeeds iff evaluation can run.
    KernelEvaluator( std::shared_ptr<const ImpulseResponseField> col_field,
                     std::shared_ptr<const ImpulseResponseField> row_field = nullptr,
                     EvalConfig config = {},
                     RBFScheme  rbf = {},
                     double     duplicate_tol = 1e-7 )
        : col_field_(std::move(col_field)), row_field_(std::move(row_field)),
          config_(config), rbf_(rbf), duplicate_tol_(duplicate_tol)
    {
        if ( !col_field_ )
        {
            throw std::invalid_argument("ellipsoid_psf::KernelEvaluator: col_field must not be null");
        }
        validate(rbf_);
        col_field_->validate(config_);
        if ( row_field_ )
        {
            // Symmetric mode pools forward centers x_i - x (source space)
            // with adjoint centers x_j - y (target space); that presumes an
            // identification of the two spaces, so all dimensions must agree.
            if ( row_field_->dim_source() != col_field_->dim_source()
                 || row_field_->dim_target() != col_field_->dim_target()
                 || col_field_->dim_source() != col_field_->dim_target() )
            {
                throw std::invalid_argument("ellipsoid_psf::KernelEvaluator: symmetric mode requires equal "
                                            "source and target dimensions across both fields (the "
                                            "forward/adjoint center pooling identifies the two "
                                            "spaces); use cols-only mode for rectangular kernels");
            }
            row_field_->validate(config_);
        }
        if ( !row_field_ && config_.symmetric_combine == SymmetricCombine::crisscross )
        {
            throw std::invalid_argument("ellipsoid_psf::KernelEvaluator: SymmetricCombine::crisscross "
                                        "requires symmetric mode (a row field); pass the same field "
                                        "twice for a symmetric operator probed once");
        }
        if ( !( duplicate_tol_ >= 0.0 ) )
        {
            throw std::invalid_argument("ellipsoid_psf::KernelEvaluator: duplicate_tol must be >= 0");
        }
    }

    const std::shared_ptr<const ImpulseResponseField>& col_field() const { return col_field_; }
    const std::shared_ptr<const ImpulseResponseField>& row_field() const { return row_field_; }
    const EvalConfig& config() const  { return config_; }
    const RBFScheme&  rbf() const     { return rbf_; }
    double duplicate_tol() const      { return duplicate_tol_; }
    bool   symmetric() const          { return static_cast<bool>(row_field_); }
    /// Source-domain dimension (of x and the interpolation centers).
    int    dim_source() const         { return col_field_->dim_source(); }
    /// Target-domain dimension (of y).
    int    dim_target() const         { return col_field_->dim_target(); }

    /// The approximate kernel entry Phi(y, x); zero where there are no
    /// predictions (no samples in reach, or x uninformed). Thread-safe.
    double operator()( const Eigen::Ref<const Eigen::VectorXd>& y,
                       const Eigen::Ref<const Eigen::VectorXd>& x ) const
    {
        if ( row_field_ && config_.symmetric_combine == SymmetricCombine::crisscross )
        {
            const double dx = col_field_->nearest_sample_distance(x);
            const double dy = row_field_->nearest_sample_distance(y);
            if ( dx < dy )
            {
                return evaluate_side(*col_field_, y, x);
            }
            if ( dy < dx )
            {
                return evaluate_side(*row_field_, x, y);
            }
            return 0.5 * ( evaluate_side(*col_field_, y, x)
                           + evaluate_side(*row_field_, x, y) );
        }

        if ( !row_field_ )
        {
            return evaluate_side(*col_field_, y, x);
        }

        // Pool predictions in displacement coordinates (evaluation point at
        // the origin): forward centers x_i - x, adjoint centers x_j - y.
        std::vector<Eigen::VectorXd> centers;
        std::vector<double>          values;

        for ( const Prediction& p : col_field_->predictions(y, x, config_) )
        {
            centers.push_back(p.point - x);
            values.push_back(p.value);
        }
        {
            const int num_fwd = static_cast<int>(centers.size());
            for ( const Prediction& p : row_field_->predictions(x, y, config_) )
            {
                // Average near-duplicates into their forward partner (the
                // x = y case makes the two sets literally coincide).
                const Eigen::VectorXd c = p.point - y;
                bool merged = false;
                for ( int jj = 0; jj < num_fwd; ++jj )
                {
                    if ( ( centers[jj] - c ).norm() <= duplicate_tol_ )
                    {
                        centers[jj] = 0.5 * ( centers[jj] + c );
                        values[jj]  = 0.5 * ( values[jj] + p.value );
                        merged = true;
                        break;
                    }
                }
                if ( !merged )
                {
                    centers.push_back(c);
                    values.push_back(p.value);
                }
            }
        }

        const int k = static_cast<int>(centers.size());
        if ( k == 0 )
        {
            return 0.0;
        }
        // Far-field short circuit: when every prediction is gated to zero the
        // interpolant is identically zero (also under smoothing) — skip the
        // RBF solve entirely. This is the dominant case for off-diagonal
        // blocks in compressed-matrix assembly.
        bool all_zero = true;
        for ( double v : values )
        {
            if ( v != 0.0 )
            {
                all_zero = false;
                break;
            }
        }
        if ( all_zero )
        {
            return 0.0;
        }
        Eigen::MatrixXd C(dim_source(), k);
        Eigen::VectorXd f(k);
        for ( int jj = 0; jj < k; ++jj )
        {
            C.col(jj) = centers[jj];
            f(jj) = values[jj];
        }
        return rbf_interpolate(f, C, Eigen::MatrixXd::Zero(dim_source(), 1), rbf_)(0);
    }

    /// The block of kernel entries [ Phi(yy.col(ii), xx.col(jj)) ]_{ii,jj},
    /// shape (num_y, num_x), evaluated in parallel (num_threads <= 0 uses all
    /// hardware threads).
    ///
    /// In cols-only mode this runs a fixed-source fast path: for each source
    /// point the mesh location, field interpolation, neighbor query, and
    /// scalings are computed once (ImpulseResponseField::
    /// predictions_over_targets), and the RBF solve collapses to one
    /// evaluation-functional weight vector per source (rbf_functional; the
    /// centers do not depend on the target), cached per exclusion pattern —
    /// so each additional target costs a gate test, a mesh lookup, and a dot
    /// product. Results agree with entrywise operator() up to rounding.
    /// Pooled symmetric mode pools row-field predictions looked up near each
    /// target, which breaks the fixed-center structure, and evaluates
    /// entrywise. Crisscross symmetric mode runs the fixed-source fast path
    /// once per side (columns as-is, rows on the transposed roles) and
    /// selects per entry by the nearest-sample distances.
    Eigen::MatrixXd block( const Eigen::Ref<const Eigen::MatrixXd>& yy, // (dim_target, num_y)
                           const Eigen::Ref<const Eigen::MatrixXd>& xx, // (dim_source, num_x)
                           int num_threads = 0 ) const
    {
        if ( yy.rows() != dim_target() || xx.rows() != dim_source() )
        {
            throw std::invalid_argument("ellipsoid_psf::KernelEvaluator::block: yy must have dim_target rows "
                                        "and xx dim_source rows");
        }
        const int ny = static_cast<int>(yy.cols());
        const int nx = static_cast<int>(xx.cols());
        Eigen::MatrixXd out(ny, nx);

        const bool crisscross =
            row_field_ && config_.symmetric_combine == SymmetricCombine::crisscross;

        // The fast path keys its weight cache on a 64-bit exclusion mask.
        if ( crisscross && config_.num_neighbors <= 63 )
        {
            const Eigen::MatrixXd A = block_one_sided(*col_field_, yy, xx, num_threads);
            const Eigen::MatrixXd B = block_one_sided(*row_field_, xx, yy, num_threads);
            Eigen::VectorXd dx(nx), dy(ny);
            for ( int jj = 0; jj < nx; ++jj )
            {
                dx(jj) = col_field_->nearest_sample_distance(xx.col(jj));
            }
            for ( int ii = 0; ii < ny; ++ii )
            {
                dy(ii) = row_field_->nearest_sample_distance(yy.col(ii));
            }
            for ( int jj = 0; jj < nx; ++jj )
            {
                for ( int ii = 0; ii < ny; ++ii )
                {
                    out(ii, jj) = ( dx(jj) < dy(ii) ) ? A(ii, jj)
                                : ( dy(ii) < dx(jj) ) ? B(jj, ii)
                                : 0.5 * ( A(ii, jj) + B(jj, ii) );
                }
            }
            return out;
        }
        if ( row_field_ || config_.num_neighbors > 63 )
        {
            ellipsoid_tree::detail::parallel_for(0, static_cast<std::ptrdiff_t>(ny) * nx,
                [&]( std::ptrdiff_t aa, std::ptrdiff_t bb )
                {
                    for ( std::ptrdiff_t ind = aa; ind < bb; ++ind )
                    {
                        const int ii = static_cast<int>(ind % ny);
                        const int jj = static_cast<int>(ind / ny);
                        out(ii, jj) = ( *this )(yy.col(ii), xx.col(jj));
                    }
                }, num_threads);
            return out;
        }

        return block_one_sided(*col_field_, yy, xx, num_threads);
    }

private:
    /// One-sided entry evaluation: RBF-combine one field's predictions with
    /// centers p.point - source, evaluated at the origin. The cols-only entry
    /// is evaluate_side(*col_field_, y, x); the adjoint (row) side of
    /// crisscross is evaluate_side(*row_field_, x, y).
    double evaluate_side( const ImpulseResponseField& field,
                          const Eigen::Ref<const Eigen::VectorXd>& target,
                          const Eigen::Ref<const Eigen::VectorXd>& source ) const
    {
        const std::vector<Prediction> preds = field.predictions(target, source, config_);
        const int k = static_cast<int>(preds.size());
        if ( k == 0 )
        {
            return 0.0;
        }
        // Far-field short circuit (see operator() comment).
        bool all_zero = true;
        for ( const Prediction& p : preds )
        {
            if ( p.value != 0.0 )
            {
                all_zero = false;
                break;
            }
        }
        if ( all_zero )
        {
            return 0.0;
        }
        const int d = field.dim_source();
        Eigen::MatrixXd C(d, k);
        Eigen::VectorXd f(k);
        for ( int jj = 0; jj < k; ++jj )
        {
            C.col(jj) = preds[jj].point - source;
            f(jj) = preds[jj].value;
        }
        return rbf_interpolate(f, C, Eigen::MatrixXd::Zero(d, 1), rbf_)(0);
    }

    /// The fixed-source fast path for one field (see block() docs); shape
    /// (num_targets, num_sources) with targets/sources in that field's roles.
    Eigen::MatrixXd block_one_sided( const ImpulseResponseField& field,
                                     const Eigen::Ref<const Eigen::MatrixXd>& yy,
                                     const Eigen::Ref<const Eigen::MatrixXd>& xx,
                                     int num_threads ) const
    {
        const int ny = static_cast<int>(yy.cols());
        const int nx = static_cast<int>(xx.cols());
        Eigen::MatrixXd out(ny, nx);
        ellipsoid_tree::detail::parallel_for(0, nx,
            [&]( std::ptrdiff_t aa, std::ptrdiff_t bb )
            {
                const Eigen::VectorXd origin = Eigen::VectorXd::Zero(field.dim_source());
                for ( std::ptrdiff_t jj = aa; jj < bb; ++jj )
                {
                    const ImpulseResponseField::PredictionSweep sweep =
                        field.predictions_over_targets(yy, xx.col(jj), config_);
                    const int k = static_cast<int>(sweep.sample_indices.size());
                    if ( k == 0 )
                    {
                        out.col(jj).setZero();
                        continue;
                    }
                    const Eigen::MatrixXd centers =
                        sweep.sample_points.colwise() - xx.col(jj);

                    // Evaluation-functional weights per exclusion pattern;
                    // usually one pattern (nothing excluded) per source.
                    std::vector<std::pair<std::uint64_t, Eigen::VectorXd>> cache;
                    for ( int ii = 0; ii < ny; ++ii )
                    {
                        std::uint64_t mask = 0;
                        bool all_zero = true;
                        for ( int nn = 0; nn < k; ++nn )
                        {
                            if ( sweep.excluded(nn, ii) )
                            {
                                mask |= ( std::uint64_t(1) << nn );
                            }
                            else if ( sweep.values(nn, ii) != 0.0 )
                            {
                                all_zero = false;
                            }
                        }
                        if ( all_zero )
                        {
                            out(ii, jj) = 0.0; // far field / fully excluded
                            continue;
                        }

                        const Eigen::VectorXd* weights = nullptr;
                        for ( const auto& entry : cache )
                        {
                            if ( entry.first == mask )
                            {
                                weights = &entry.second;
                                break;
                            }
                        }
                        if ( !weights )
                        {
                            int active = 0;
                            Eigen::MatrixXd C(field.dim_source(), k);
                            for ( int nn = 0; nn < k; ++nn )
                            {
                                if ( !( mask & ( std::uint64_t(1) << nn ) ) )
                                {
                                    C.col(active++) = centers.col(nn);
                                }
                            }
                            cache.emplace_back(mask,
                                rbf_functional(C.leftCols(active), origin, rbf_));
                            weights = &cache.back().second;
                        }

                        double value = 0.0;
                        int active = 0;
                        for ( int nn = 0; nn < k; ++nn )
                        {
                            if ( !( mask & ( std::uint64_t(1) << nn ) ) )
                            {
                                value += ( *weights )(active++) * sweep.values(nn, ii);
                            }
                        }
                        out(ii, jj) = value;
                    }
                }
            }, num_threads);
        return out;
    }

public:
    /// Ellipsoids (unit scale, tau folded in) covering the support of the
    /// col-field contribution to the kernel column Phi(., x): those
    /// predictions vanish for y outside the union. In cols-only mode this
    /// covers the whole column; in symmetric mode the row field contributes
    /// too, and the full column support is this union PLUS the targets y
    /// with x inside source_support(y). Requires the ellipsoid support gate.
    std::vector<ellipsoid_tree::Ellipsoid> target_support( const Eigen::Ref<const Eigen::VectorXd>& x ) const
    {
        return col_field_->support_ellipsoids(x, config_);
    }

    /// Symmetric mode only: ellipsoids in the SOURCE domain covering the
    /// support of the row-field contribution to the kernel row Phi(y, .) —
    /// that contribution vanishes for x outside the union. (The col field
    /// contributes to the row elsewhere; its part is covered per-column by
    /// target_support.) Throws in cols-only mode, where no row field exists.
    std::vector<ellipsoid_tree::Ellipsoid> source_support( const Eigen::Ref<const Eigen::VectorXd>& y ) const
    {
        if ( !row_field_ )
        {
            throw std::logic_error("ellipsoid_psf::KernelEvaluator::source_support: requires symmetric mode "
                                   "(a row field); in cols-only mode kernel rows have no computable "
                                   "compact support");
        }
        return row_field_->support_ellipsoids(y, config_);
    }

private:
    std::shared_ptr<const ImpulseResponseField> col_field_;
    std::shared_ptr<const ImpulseResponseField> row_field_;
    EvalConfig config_;
    RBFScheme  rbf_;
    double     duplicate_tol_;
};

} // end namespace ellipsoid_psf
