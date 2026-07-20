// SPDX-License-Identifier: MIT
// Part of psfi — https://github.com/NickAlger/psf_interpolation
//
// Python bindings. Array-layout convention at the Python boundary: POINTS ARE
// ROWS, matching numpy/scipy practice (and the etree bindings) — point sets
// are (n, d), mesh vertices (num_vertices, d), mesh cells (num_cells, d+1),
// covariance stacks (n, d, d). Internally psfi stores points as columns; the
// transpose happens here, once, at the boundary.

#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "psfi/psfi.hpp"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace psfi;

namespace {

using RowsXd = Eigen::Ref<const Eigen::MatrixXd>;
using RowsXi = Eigen::Ref<const Eigen::MatrixXi>;

Eigen::MatrixXd cols_from_rows( const RowsXd& rows )  { return rows.transpose(); }
Eigen::MatrixXi icols_from_rows( const RowsXi& rows ) { return rows.transpose(); }

// (n, d, d) numpy stack -> one matrix per entry.
std::vector<Eigen::MatrixXd> matrices_from_stack( const py::array_t<double>& stack, const char* name )
{
    if ( stack.ndim() != 3 || stack.shape(1) != stack.shape(2) )
    {
        throw std::invalid_argument(std::string(name) + " must have shape (n, d, d)");
    }
    auto A = stack.unchecked<3>();
    const int n = static_cast<int>(stack.shape(0));
    const int d = static_cast<int>(stack.shape(1));
    std::vector<Eigen::MatrixXd> out(n);
    for ( int ii = 0; ii < n; ++ii )
    {
        Eigen::MatrixXd M(d, d);
        for ( int rr = 0; rr < d; ++rr )
        {
            for ( int cc = 0; cc < d; ++cc )
            {
                M(rr, cc) = A(ii, rr, cc);
            }
        }
        out[ii] = std::move(M);
    }
    return out;
}

// (n, d, d) numpy stack -> (d*d, n) matrix with column v = vec(Sigma_v)
// (column-major vec, matching ImpulseResponseField::set_moment_fields).
Eigen::MatrixXd flat_field_from_stack( const py::array_t<double>& stack, const char* name )
{
    if ( stack.ndim() != 3 || stack.shape(1) != stack.shape(2) )
    {
        throw std::invalid_argument(std::string(name) + " must have shape (num_vertices, d, d)");
    }
    auto A = stack.unchecked<3>();
    const int n = static_cast<int>(stack.shape(0));
    const int d = static_cast<int>(stack.shape(1));
    Eigen::MatrixXd out(d * d, n);
    for ( int vv = 0; vv < n; ++vv )
    {
        for ( int cc = 0; cc < d; ++cc )
        {
            for ( int rr = 0; rr < d; ++rr )
            {
                out(rr + cc * d, vv) = A(vv, rr, cc);
            }
        }
    }
    return out;
}

// (d*d, n) flat field -> (n, d, d) numpy stack.
py::array_t<double> stack_from_flat_field( const Eigen::MatrixXd& flat, int d )
{
    const int n = static_cast<int>(flat.cols());
    py::array_t<double> out({n, d, d});
    auto A = out.mutable_unchecked<3>();
    for ( int ii = 0; ii < n; ++ii )
    {
        for ( int cc = 0; cc < d; ++cc )
        {
            for ( int rr = 0; rr < d; ++rr )
            {
                A(ii, rr, cc) = flat(rr + cc * d, ii);
            }
        }
    }
    return out;
}

// one matrix per entry -> (n, d, d) numpy stack.
py::array_t<double> stack_from_matrices( const std::vector<Eigen::MatrixXd>& mats, int d )
{
    const int n = static_cast<int>(mats.size());
    py::array_t<double> out({n, d, d});
    auto A = out.mutable_unchecked<3>();
    for ( int ii = 0; ii < n; ++ii )
    {
        for ( int rr = 0; rr < d; ++rr )
        {
            for ( int cc = 0; cc < d; ++cc )
            {
                A(ii, rr, cc) = mats[ii](rr, cc);
            }
        }
    }
    return out;
}

const char* frame_name( Frame f )
{
    switch ( f )
    {
        case Frame::identity:         return "identity";
        case Frame::translation:      return "translation";
        case Frame::mean_translation: return "mean_translation";
        case Frame::whitened_affine:  return "whitened_affine";
    }
    return "?";
}

const char* scaling_name( Scaling s )
{
    switch ( s )
    {
        case Scaling::none:       return "none";
        case Scaling::volume:     return "volume";
        case Scaling::volume_det: return "volume_det";
    }
    return "?";
}

const char* support_name( Support s )
{
    switch ( s )
    {
        case Support::none:      return "none";
        case Support::ellipsoid: return "ellipsoid";
    }
    return "?";
}

const char* kernel_name( RBFKernel k )
{
    switch ( k )
    {
        case RBFKernel::gaussian:             return "gaussian";
        case RBFKernel::multiquadric:         return "multiquadric";
        case RBFKernel::inverse_multiquadric: return "inverse_multiquadric";
        case RBFKernel::linear:               return "linear";
        case RBFKernel::thin_plate_spline:    return "thin_plate_spline";
        case RBFKernel::cubic:                return "cubic";
    }
    return "?";
}

} // end anonymous namespace

PYBIND11_MODULE(psfi, m)
{
    m.doc() = "psfi: point spread function interpolation — evaluate an integral kernel\n"
              "Phi(y, x) anywhere by interpolating transported impulse responses sampled\n"
              "at scattered points.\n\n"
              "Array convention: points are rows — point sets are (n, d), mesh vertices\n"
              "(num_vertices, d), mesh cells (num_cells, d+1), covariance stacks (n, d, d).";
    m.attr("__version__") = py::str(std::to_string(PSFI_VERSION_MAJOR) + "."
                                    + std::to_string(PSFI_VERSION_MINOR) + "."
                                    + std::to_string(PSFI_VERSION_PATCH));

    py::enum_<Frame>(m, "Frame",
        "Frame map T_i transporting a stored impulse response at x_i to the query point x.")
        .value("identity", Frame::identity, "T_i(y) = y (paper eq. 4.7)")
        .value("translation", Frame::translation, "T_i(y) = y - x + x_i (4.8)")
        .value("mean_translation", Frame::mean_translation, "T_i(y) = y - mu(x) + mu_i (4.9/4.10)")
        .value("whitened_affine", Frame::whitened_affine,
               "T_i(y) = mu_i + Sigma_i^{1/2} Sigma(x)^{-1/2} (y - mu(x))");

    py::enum_<Scaling>(m, "Scaling",
        "Scalar correction s_i applied to the transported impulse response value.")
        .value("none", Scaling::none, "s_i = 1")
        .value("volume", Scaling::volume, "s_i = V(x)/V_i (preserves peak values)")
        .value("volume_det", Scaling::volume_det,
               "s_i = (V(x)/V_i) sqrt(det Sigma_i / det Sigma(x)) (preserves mass)");

    py::enum_<Support>(m, "Support",
        "Support gate for transported points outside the sample's ellipsoid.")
        .value("none", Support::none, "no gate")
        .value("ellipsoid", Support::ellipsoid, "f_i = 0 outside E_i(tau)");

    py::class_<EvalConfig>(m, "EvalConfig",
        "Evaluation configuration: frame map, scaling, support gate, tau, num_neighbors.\n"
        "Different configurations require different data; see ImpulseResponseField.validate.")
        .def(py::init([]( Frame frame, Scaling scaling, Support support, double tau, int num_neighbors )
             {
                 EvalConfig cfg;
                 cfg.frame = frame;
                 cfg.scaling = scaling;
                 cfg.support = support;
                 cfg.tau = tau;
                 cfg.num_neighbors = num_neighbors;
                 return cfg;
             }),
             "frame"_a = Frame::mean_translation, "scaling"_a = Scaling::volume,
             "support"_a = Support::ellipsoid, "tau"_a = 3.0, "num_neighbors"_a = 10)
        .def_readwrite("frame", &EvalConfig::frame)
        .def_readwrite("scaling", &EvalConfig::scaling)
        .def_readwrite("support", &EvalConfig::support)
        .def_readwrite("tau", &EvalConfig::tau)
        .def_readwrite("num_neighbors", &EvalConfig::num_neighbors)
        .def("__repr__", []( const EvalConfig& c )
             {
                 return std::string("EvalConfig(frame=") + frame_name(c.frame)
                     + ", scaling=" + scaling_name(c.scaling)
                     + ", support=" + support_name(c.support)
                     + ", tau=" + std::to_string(c.tau)
                     + ", num_neighbors=" + std::to_string(c.num_neighbors) + ")";
             });

    py::class_<ImpulseResponseField, std::shared_ptr<ImpulseResponseField>>(m, "ImpulseResponseField",
        "Batches of sampled impulse responses on a simplicial mesh; produces\n"
        "per-neighbor kernel predictions for arbitrary target pairs (y, x).\n\n"
        "vertices: (num_vertices, d); cells: (num_cells, d+1) — the TARGET mesh,\n"
        "carrying the batch functions. Optionally pass source_vertices/source_cells\n"
        "for a separate SOURCE mesh carrying the moment fields and locating the\n"
        "query point x (its dimension may differ; see the moment-map discussion in\n"
        "the C++ docs). By default source = target.\n"
        "batches_normalized=True means each stored batch is sum_i phi_i / V_i\n"
        "(the paper's convention); False means batches store raw impulse responses.")
        .def(py::init([]( const RowsXd& vertices, const RowsXi& cells,
                          bool batches_normalized, int num_threads,
                          std::optional<Eigen::MatrixXd> source_vertices,
                          std::optional<Eigen::MatrixXi> source_cells )
             {
                 if ( source_vertices.has_value() != source_cells.has_value() )
                 {
                     throw std::invalid_argument("provide both source_vertices and source_cells, "
                                                 "or neither");
                 }
                 if ( source_vertices )
                 {
                     return ImpulseResponseField(
                         cols_from_rows(vertices), icols_from_rows(cells),
                         source_vertices->transpose(), source_cells->transpose(),
                         batches_normalized, num_threads);
                 }
                 return ImpulseResponseField(cols_from_rows(vertices), icols_from_rows(cells),
                                             batches_normalized, num_threads);
             }),
             "vertices"_a, "cells"_a, "batches_normalized"_a = true, "num_threads"_a = 0,
             "source_vertices"_a = py::none(), "source_cells"_a = py::none())
        .def_property_readonly("dim_source", &ImpulseResponseField::dim_source)
        .def_property_readonly("dim_target", &ImpulseResponseField::dim_target)
        .def_property_readonly("num_source_vertices", &ImpulseResponseField::num_source_vertices)
        .def_property_readonly("num_target_vertices", &ImpulseResponseField::num_target_vertices)
        .def_property_readonly("has_separate_source_mesh",
                               &ImpulseResponseField::has_separate_source_mesh)
        .def_property_readonly("num_batches", &ImpulseResponseField::num_batches)
        .def_property_readonly("num_sample_points", &ImpulseResponseField::num_sample_points)
        .def_property_readonly("batches_normalized", &ImpulseResponseField::batches_normalized)
        .def_property_readonly("has_sample_V", &ImpulseResponseField::has_sample_V)
        .def_property_readonly("has_sample_mu", &ImpulseResponseField::has_sample_mu)
        .def_property_readonly("has_sample_Sigma", &ImpulseResponseField::has_sample_Sigma)
        .def_property_readonly("has_field_V", &ImpulseResponseField::has_field_V)
        .def_property_readonly("has_field_mu", &ImpulseResponseField::has_field_mu)
        .def_property_readonly("has_field_Sigma", &ImpulseResponseField::has_field_Sigma)
        .def_property_readonly("target_mesh_vertices",
             []( const ImpulseResponseField& F )
             { return Eigen::MatrixXd(F.target_mesh().vertices().transpose()); },
             "Target-mesh vertex coordinates, shape (num_target_vertices, dim_target).")
        .def_property_readonly("target_mesh_cells",
             []( const ImpulseResponseField& F )
             { return Eigen::MatrixXi(F.target_mesh().cells().transpose()); },
             "Target-mesh cell vertex indices, shape (num_cells, dim_target+1).")
        .def_property_readonly("source_mesh_vertices",
             []( const ImpulseResponseField& F )
             { return Eigen::MatrixXd(F.source_mesh().vertices().transpose()); },
             "Source-mesh vertex coordinates (the target mesh's in the square case).")
        .def_property_readonly("source_mesh_cells",
             []( const ImpulseResponseField& F )
             { return Eigen::MatrixXi(F.source_mesh().cells().transpose()); },
             "Source-mesh cell vertex indices (the target mesh's in the square case).")
        .def_property_readonly("sample_points",
             []( const ImpulseResponseField& F )
             { return Eigen::MatrixXd(F.sample_points().transpose()); },
             "Sample points, shape (num_sample_points, d).")
        .def_property_readonly("sample_V",
             []( const ImpulseResponseField& F )
             {
                 return Eigen::Map<const Eigen::VectorXd>(F.sample_V().data(),
                                                          static_cast<int>(F.sample_V().size()))
                     .eval();
             },
             "Per-sample masses V_i, shape (num_sample_points,); empty if absent.")
        .def_property_readonly("sample_mu",
             []( const ImpulseResponseField& F )
             {
                 const auto& mu = F.sample_mu();
                 Eigen::MatrixXd out(static_cast<int>(mu.size()), F.dim_target());
                 for ( int ii = 0; ii < static_cast<int>(mu.size()); ++ii )
                 {
                     out.row(ii) = mu[ii].transpose();
                 }
                 return out;
             },
             "Per-sample means mu_i, shape (num_sample_points, d); empty if absent.")
        .def_property_readonly("sample_Sigma",
             []( const ImpulseResponseField& F )
             { return stack_from_matrices(F.sample_Sigma(), F.dim_target()); },
             "Per-sample covariances Sigma_i (symmetrized), shape (num_sample_points, d, d); "
             "empty if absent.")
        .def_property_readonly("point2batch",
             []( const ImpulseResponseField& F )
             {
                 return Eigen::Map<const Eigen::VectorXi>(F.point2batch().data(),
                                                          static_cast<int>(F.point2batch().size()))
                     .eval();
             },
             "Batch index of each sample point, shape (num_sample_points,).")
        .def("batch_range", &ImpulseResponseField::batch_range, "b"_a,
             "Sample index range (start, stop) of batch b.")
        .def("batch_values", &ImpulseResponseField::batch_values, "b"_a,
             "CG1 vertex values of batch b, shape (num_vertices,).")
        .def("set_moment_fields",
             []( ImpulseResponseField& F,
                 std::optional<Eigen::VectorXd> V,
                 std::optional<Eigen::MatrixXd> mu,
                 std::optional<py::array_t<double>> Sigma )
             {
                 const Eigen::VectorXd V_arg  = V ? *V : Eigen::VectorXd();
                 const Eigen::MatrixXd mu_arg = mu ? cols_from_rows(*mu) : Eigen::MatrixXd();
                 const Eigen::MatrixXd Sigma_arg =
                     Sigma ? flat_field_from_stack(*Sigma, "Sigma") : Eigen::MatrixXd();
                 F.set_moment_fields(V_arg, mu_arg, Sigma_arg);
             },
             "V"_a = py::none(), "mu"_a = py::none(), "Sigma"_a = py::none(),
             "Sets the vertex moment fields; pass None for fields you do not have.\n"
             "V: (num_vertices,); mu: (num_vertices, d); Sigma: (num_vertices, d, d).")
        .def("add_batch",
             []( ImpulseResponseField& F,
                 const RowsXd& points,
                 const Eigen::VectorXd& psi,
                 std::optional<Eigen::VectorXd> V,
                 std::optional<Eigen::MatrixXd> mu,
                 std::optional<py::array_t<double>> Sigma,
                 bool rebuild )
             {
                 const Eigen::VectorXd V_arg  = V ? *V : Eigen::VectorXd();
                 const Eigen::MatrixXd mu_arg = mu ? cols_from_rows(*mu) : Eigen::MatrixXd();
                 const std::vector<Eigen::MatrixXd> Sigma_arg =
                     Sigma ? matrices_from_stack(*Sigma, "Sigma") : std::vector<Eigen::MatrixXd>();
                 F.add_batch(cols_from_rows(points), psi, V_arg, mu_arg, Sigma_arg, rebuild);
             },
             "points"_a, "psi"_a, "V"_a = py::none(), "mu"_a = py::none(), "Sigma"_a = py::none(),
             "rebuild"_a = true,
             "Adds one impulse response batch. points: (num_batch_points, d) arbitrary\n"
             "coordinates; psi: (num_vertices,) CG1 vertex values; optional per-sample\n"
             "moments V (num_batch_points,), mu (num_batch_points, d),\n"
             "Sigma (num_batch_points, d, d). Which moments are supplied is fixed by the\n"
             "first batch. With rebuild=False call rebuild_kdtree() before predictions().")
        .def("rebuild_kdtree", &ImpulseResponseField::rebuild_kdtree)
        .def("validate", &ImpulseResponseField::validate, "config"_a,
             "Raises ValueError listing every piece of data `config` needs but the field\n"
             "does not have; returns None iff predictions(y, x, config) can run.")
        .def("predictions",
             []( const ImpulseResponseField& F,
                 const Eigen::VectorXd& y,
                 const Eigen::VectorXd& x,
                 const EvalConfig& config )
             {
                 const std::vector<Prediction> P = F.predictions(y, x, config);
                 const int k = static_cast<int>(P.size());
                 Eigen::VectorXi indices(k);
                 Eigen::MatrixXd points(k, F.dim_source());
                 Eigen::VectorXd values(k);
                 for ( int jj = 0; jj < k; ++jj )
                 {
                     indices(jj)    = P[jj].sample_index;
                     points.row(jj) = P[jj].point.transpose();
                     values(jj)     = P[jj].value;
                 }
                 return std::make_tuple(indices, points, values);
             },
             "y"_a, "x"_a, "config"_a, py::call_guard<py::gil_scoped_release>(),
             "Per-neighbor kernel predictions at the target pair (y, x), both arbitrary\n"
             "coordinates. Returns (sample_indices (k,), sample_points (k, d),\n"
             "values (k,)), nearest sample first; k <= num_neighbors (samples whose\n"
             "transported point leaves the mesh are excluded, and k = 0 when the\n"
             "configuration needs moment fields at an x outside the mesh).");

    // ------------------------------------------------------------------
    //  Moment-data hygiene
    // ------------------------------------------------------------------

    const char* clamp_spd_doc =
        "Symmetrize a covariance stack (n, d, d) and clamp its eigenvalues to at\n"
        "least `floor` (> 0; scalar or per-entry array of shape (n,)). Returns\n"
        "(cleaned_stack, modified_indices). psfi requires strictly positive\n"
        "definite covariances (add_batch and set_moment_fields validate); use this\n"
        "to repair fields corrupted by numerical error. The floor is a modelling\n"
        "choice, not just hygiene — near-singular covariances pass validation but\n"
        "amplify through Sigma(x)^{-1/2} and det Sigma(x); the square of the local\n"
        "mesh spacing is a reasonable default.";
    auto clamp_stack = []( const py::array_t<double>& Sigmas, const Eigen::VectorXd& floors )
    {
        const Eigen::MatrixXd flat = flat_field_from_stack(Sigmas, "Sigmas");
        const int d = static_cast<int>(Sigmas.shape(1));
        std::pair<Eigen::MatrixXd, std::vector<int>> result = clamp_spd_field(flat, d, floors);
        Eigen::VectorXi modified(static_cast<int>(result.second.size()));
        for ( int ii = 0; ii < modified.size(); ++ii )
        {
            modified(ii) = result.second[ii];
        }
        return std::make_pair(stack_from_flat_field(result.first, d), modified);
    };
    m.def("clamp_spd",
          [clamp_stack]( const py::array_t<double>& Sigmas, double floor )
          {
              return clamp_stack(Sigmas,
                                 Eigen::VectorXd::Constant(Sigmas.shape(0), floor));
          },
          "Sigmas"_a, "floor"_a, clamp_spd_doc);
    m.def("clamp_spd", clamp_stack, "Sigmas"_a, "floor"_a, clamp_spd_doc);

    // ------------------------------------------------------------------
    //  Low-rank tools
    // ------------------------------------------------------------------
    // Generic matrix layer: plain rows/columns, no source/target semantics
    // (see the low_rank.hpp file header). Matrices pass through unchanged —
    // the points-are-rows transposition does not apply here.

    py::class_<LowRank>(m, "LowRank",
        "Rank-r factorization A ~ U @ V.T with U (num_rows, r), V (num_cols, r).\n"
        "Scale is folded into U; V is orthonormal when produced by\n"
        "truncated_svd / recompress / randomized_svd.")
        .def(py::init([]( const Eigen::MatrixXd& U, const Eigen::MatrixXd& V )
             {
                 if ( U.cols() != V.cols() )
                 {
                     throw std::invalid_argument("LowRank: U and V must have the same number "
                                                 "of columns (the rank)");
                 }
                 return LowRank{ U, V };
             }),
             "U"_a, "V"_a)
        .def_readwrite("U", &LowRank::U)
        .def_readwrite("V", &LowRank::V)
        .def_property_readonly("rank", &LowRank::rank)
        .def("to_dense", &LowRank::to_dense, "The dense matrix U @ V.T.")
        .def("__repr__", []( const LowRank& F )
             {
                 return "LowRank(num_rows=" + std::to_string(F.U.rows())
                     + ", num_cols=" + std::to_string(F.V.rows())
                     + ", rank=" + std::to_string(F.rank()) + ")";
             });

    m.def("truncated_svd",
          []( const Eigen::MatrixXd& A, double rtol, int max_rank )
          { return truncated_svd(A, rtol, max_rank); },
          "A"_a, "rtol"_a, "max_rank"_a = -1,
          py::call_guard<py::gil_scoped_release>(),
          "Best approximation with the smallest rank whose discarded singular-value\n"
          "tail satisfies ||tail||_F <= rtol * ||A||_F, capped at max_rank\n"
          "(max_rank < 0 means no cap).");

    m.def("recompress",
          []( const LowRank& F, double rtol, int max_rank )
          { return recompress(F, rtol, max_rank); },
          "factors"_a, "rtol"_a, "max_rank"_a = -1,
          py::call_guard<py::gil_scoped_release>(),
          "Recompresses a (possibly redundant) factorization to the smallest rank\n"
          "meeting the relative Frobenius tolerance.");

    py::class_<ACAOptions>(m, "ACAOptions",
        "Options for aca(). The stopping tolerance is aca_safety_factor * rtol and\n"
        "the recompression tolerance recompress_safety_factor * rtol.")
        .def(py::init([]( int max_rank, double aca_safety_factor, double recompress_safety_factor,
                          int required_consecutive_successes, bool recompress, unsigned int seed )
             {
                 ACAOptions o;
                 o.max_rank = max_rank;
                 o.aca_safety_factor = aca_safety_factor;
                 o.recompress_safety_factor = recompress_safety_factor;
                 o.required_consecutive_successes = required_consecutive_successes;
                 o.recompress = recompress;
                 o.seed = seed;
                 return o;
             }),
             "max_rank"_a = -1, "aca_safety_factor"_a = 0.25, "recompress_safety_factor"_a = 0.75,
             "required_consecutive_successes"_a = 10, "recompress"_a = true, "seed"_a = 0)
        .def_readwrite("max_rank", &ACAOptions::max_rank)
        .def_readwrite("aca_safety_factor", &ACAOptions::aca_safety_factor)
        .def_readwrite("recompress_safety_factor", &ACAOptions::recompress_safety_factor)
        .def_readwrite("required_consecutive_successes", &ACAOptions::required_consecutive_successes)
        .def_readwrite("recompress", &ACAOptions::recompress)
        .def_readwrite("seed", &ACAOptions::seed)
        .def("__repr__", []( const ACAOptions& o )
             {
                 return "ACAOptions(max_rank=" + std::to_string(o.max_rank)
                     + ", aca_safety_factor=" + std::to_string(o.aca_safety_factor)
                     + ", recompress_safety_factor=" + std::to_string(o.recompress_safety_factor)
                     + ", required_consecutive_successes="
                     + std::to_string(o.required_consecutive_successes)
                     + ", recompress=" + ( o.recompress ? "True" : "False" )
                     + ", seed=" + std::to_string(o.seed) + ")";
             });

    py::class_<ACAResult>(m, "ACAResult",
        "aca() result: factors plus construction diagnostics. Check hit_max_rank —\n"
        "never let a rank cap bind silently.")
        .def_readonly("factors", &ACAResult::factors)
        .def_readonly("sampled_rows", &ACAResult::sampled_rows)
        .def_readonly("sampled_cols", &ACAResult::sampled_cols)
        .def_readonly("sampled_rank", &ACAResult::sampled_rank)
        .def_readonly("converged", &ACAResult::converged)
        .def_readonly("hit_max_rank", &ACAResult::hit_max_rank)
        .def_readonly("relerr_estimate", &ACAResult::relerr_estimate)
        .def("__repr__", []( const ACAResult& r )
             {
                 return "ACAResult(rank=" + std::to_string(r.factors.rank())
                     + ", sampled_rank=" + std::to_string(r.sampled_rank)
                     + ", converged=" + ( r.converged ? "True" : "False" )
                     + ", hit_max_rank=" + ( r.hit_max_rank ? "True" : "False" ) + ")";
             });

    m.def("aca",
          []( const std::function<Eigen::VectorXd(int)>& get_row,
              const std::function<Eigen::VectorXd(int)>& get_col,
              int num_rows, int num_cols, double rtol, const ACAOptions& options )
          { return aca(get_row, get_col, num_rows, num_cols, rtol, options); },
          "get_row"_a, "get_col"_a, "num_rows"_a, "num_cols"_a, "rtol"_a,
          "options"_a = ACAOptions{},
          "Adaptive cross approximation with partial pivoting and random-restart\n"
          "verification (the GPSF/ymir 'ACA+'). get_row(i) must return row i and\n"
          "get_col(j) column j of the same fixed matrix; only sampled rows and\n"
          "columns are ever evaluated. rtol = 0 runs to exact recovery.\n"
          "Deterministic for a given options.seed.");

    py::class_<RSVDOptions>(m, "RSVDOptions", "Options for randomized_svd().")
        .def(py::init([]( int oversampling, int power_iterations, double rtol, unsigned int seed )
             {
                 RSVDOptions o;
                 o.oversampling = oversampling;
                 o.power_iterations = power_iterations;
                 o.rtol = rtol;
                 o.seed = seed;
                 return o;
             }),
             "oversampling"_a = 10, "power_iterations"_a = 1, "rtol"_a = 0.0, "seed"_a = 0)
        .def_readwrite("oversampling", &RSVDOptions::oversampling)
        .def_readwrite("power_iterations", &RSVDOptions::power_iterations)
        .def_readwrite("rtol", &RSVDOptions::rtol)
        .def_readwrite("seed", &RSVDOptions::seed)
        .def("__repr__", []( const RSVDOptions& o )
             {
                 return "RSVDOptions(oversampling=" + std::to_string(o.oversampling)
                     + ", power_iterations=" + std::to_string(o.power_iterations)
                     + ", rtol=" + std::to_string(o.rtol)
                     + ", seed=" + std::to_string(o.seed) + ")";
             });

    m.def("randomized_svd",
          []( const std::function<Eigen::MatrixXd(const Eigen::Ref<const Eigen::MatrixXd>&)>& apply,
              const std::function<Eigen::MatrixXd(const Eigen::Ref<const Eigen::MatrixXd>&)>& apply_transpose,
              int num_rows, int num_cols, int max_rank, const RSVDOptions& options )
          { return randomized_svd(apply, apply_transpose, num_rows, num_cols, max_rank, options); },
          "apply"_a, "apply_transpose"_a, "num_rows"_a, "num_cols"_a, "max_rank"_a,
          "options"_a = RSVDOptions{},
          "Randomized SVD of a matrix available only through its action:\n"
          "apply(X) = A @ X on (num_cols, k) blocks, apply_transpose(Y) = A.T @ Y on\n"
          "(num_rows, k) blocks. Roughly the best rank-max_rank approximation;\n"
          "deterministic for a given options.seed.");

    // ------------------------------------------------------------------
    //  RBF interpolation
    // ------------------------------------------------------------------

    py::enum_<RBFKernel>(m, "RBFKernel",
        "Radial kernel phi(u) at the locally scaled distance u = shape * r / r0\n"
        "(r0 = diameter of the interpolation point set, recomputed per call).\n"
        "Sign conventions follow scipy.interpolate.RBFInterpolator.")
        .value("gaussian", RBFKernel::gaussian, "exp(-u^2/2)")
        .value("multiquadric", RBFKernel::multiquadric, "-sqrt(1 + u^2)")
        .value("inverse_multiquadric", RBFKernel::inverse_multiquadric, "1/sqrt(1 + u^2)")
        .value("linear", RBFKernel::linear, "-u")
        .value("thin_plate_spline", RBFKernel::thin_plate_spline, "u^2 log u")
        .value("cubic", RBFKernel::cubic, "u^3");

    py::class_<RBFScheme>(m, "RBFScheme",
        "RBF interpolation configuration: kernel, shape (the paper's C_RBF),\n"
        "polynomial-tail degree (-1 none, 0 constant, 1 linear), ridge smoothing.\n"
        "With smoothing = 0 the interpolant reproduces its data at the centers;\n"
        "smoothing has no effect when there are no more centers than tail\n"
        "coefficients (the system degenerates to polynomial interpolation).")
        .def(py::init([]( RBFKernel kernel, double shape, int degree, double smoothing )
             {
                 RBFScheme s;
                 s.kernel = kernel;
                 s.shape = shape;
                 s.degree = degree;
                 s.smoothing = smoothing;
                 validate(s);
                 return s;
             }),
             "kernel"_a = RBFKernel::gaussian, "shape"_a = 3.0, "degree"_a = 1, "smoothing"_a = 0.0)
        .def_readwrite("kernel", &RBFScheme::kernel)
        .def_readwrite("shape", &RBFScheme::shape)
        .def_readwrite("degree", &RBFScheme::degree)
        .def_readwrite("smoothing", &RBFScheme::smoothing)
        .def("__repr__", []( const RBFScheme& s )
             {
                 return std::string("RBFScheme(kernel=") + kernel_name(s.kernel)
                     + ", shape=" + std::to_string(s.shape)
                     + ", degree=" + std::to_string(s.degree)
                     + ", smoothing=" + std::to_string(s.smoothing) + ")";
             });

    m.def("rbf_min_degree", &rbf_min_degree, "kernel"_a,
          "Smallest polynomial-tail degree guaranteeing solvability for this kernel.");

    m.def("rbf_interpolate",
          []( const Eigen::VectorXd& values, const RowsXd& centers, const RowsXd& eval_points,
              const RBFScheme& scheme )
          {
              return rbf_interpolate(values, cols_from_rows(centers).eval(),
                                     cols_from_rows(eval_points).eval(), scheme);
          },
          "values"_a, "centers"_a, "eval_points"_a, "scheme"_a = RBFScheme{},
          py::call_guard<py::gil_scoped_release>(),
          "RBF interpolant of {(centers[i], values[i])} evaluated at eval_points.\n"
          "values: (k,); centers: (k, d); eval_points: (m, d). Returns (m,).\n"
          "One center (or all coincident) gives a constant; the tail degree is\n"
          "lowered automatically when there are fewer centers than coefficients.");

    // ------------------------------------------------------------------
    //  Kernel evaluator
    // ------------------------------------------------------------------

    py::class_<KernelEvaluator>(m, "KernelEvaluator",
        "The complete kernel approximation Phi(y, x): RBF combination of\n"
        "per-neighbor predictions. Cols-only with one field, symmetric with a\n"
        "row field probed with the transpose operator (pass the same field\n"
        "twice for a symmetric operator probed once); in symmetric mode the\n"
        "forward and adjoint prediction sets are pooled in displacement\n"
        "coordinates and near-duplicate centers are averaged. Entries at\n"
        "sample columns are exact when smoothing = 0 (center reproduction —\n"
        "no snapping special case).")
        .def(py::init([]( std::shared_ptr<const ImpulseResponseField> col_field,
                          std::shared_ptr<const ImpulseResponseField> row_field,
                          const EvalConfig& config, const RBFScheme& rbf, double duplicate_tol )
             {
                 return KernelEvaluator(std::move(col_field), std::move(row_field),
                                        config, rbf, duplicate_tol);
             }),
             "col_field"_a, "row_field"_a = nullptr, "config"_a = EvalConfig{},
             "rbf"_a = RBFScheme{}, "duplicate_tol"_a = 1e-7,
             "Validates the fields against the configuration here: construction\n"
             "succeeds iff evaluation can run.")
        .def_property_readonly("dim_source", &KernelEvaluator::dim_source)
        .def_property_readonly("dim_target", &KernelEvaluator::dim_target)
        .def_property_readonly("symmetric", &KernelEvaluator::symmetric)
        .def_property_readonly("duplicate_tol", &KernelEvaluator::duplicate_tol)
        .def_property_readonly("config", []( const KernelEvaluator& K ) { return K.config(); })
        .def_property_readonly("rbf", []( const KernelEvaluator& K ) { return K.rbf(); })
        .def("__call__",
             []( const KernelEvaluator& K, const Eigen::VectorXd& y, const Eigen::VectorXd& x )
             { return K(y, x); },
             "y"_a, "x"_a, py::call_guard<py::gil_scoped_release>(),
             "The approximate kernel entry Phi(y, x); zero where there are no predictions.")
        .def("block",
             []( const KernelEvaluator& K, const RowsXd& yy, const RowsXd& xx, int num_threads )
             {
                 return Eigen::MatrixXd(K.block(cols_from_rows(yy).eval(),
                                                cols_from_rows(xx).eval(), num_threads));
             },
             "yy"_a, "xx"_a, "num_threads"_a = 0, py::call_guard<py::gil_scoped_release>(),
             "Block [Phi(yy[i], xx[j])]_{ij} of shape (num_y, num_x), evaluated in\n"
             "parallel; yy: (num_y, d), xx: (num_x, d). num_threads <= 0 uses all cores.");
}
