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
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
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

    py::class_<ImpulseResponseField>(m, "ImpulseResponseField",
        "Batches of sampled impulse responses on a simplicial mesh; produces\n"
        "per-neighbor kernel predictions for arbitrary target pairs (y, x).\n\n"
        "vertices: (num_vertices, d); cells: (num_cells, d+1).\n"
        "batches_normalized=True means each stored batch is sum_i phi_i / V_i\n"
        "(the paper's convention); False means batches store raw impulse responses.")
        .def(py::init([]( const RowsXd& vertices, const RowsXi& cells,
                          bool batches_normalized, int num_threads )
             {
                 return ImpulseResponseField(cols_from_rows(vertices), icols_from_rows(cells),
                                             batches_normalized, num_threads);
             }),
             "vertices"_a, "cells"_a, "batches_normalized"_a = true, "num_threads"_a = 0)
        .def_property_readonly("dim", &ImpulseResponseField::dim)
        .def_property_readonly("num_vertices", &ImpulseResponseField::num_vertices)
        .def_property_readonly("num_batches", &ImpulseResponseField::num_batches)
        .def_property_readonly("num_sample_points", &ImpulseResponseField::num_sample_points)
        .def_property_readonly("batches_normalized", &ImpulseResponseField::batches_normalized)
        .def_property_readonly("has_sample_V", &ImpulseResponseField::has_sample_V)
        .def_property_readonly("has_sample_mu", &ImpulseResponseField::has_sample_mu)
        .def_property_readonly("has_sample_Sigma", &ImpulseResponseField::has_sample_Sigma)
        .def_property_readonly("has_field_V", &ImpulseResponseField::has_field_V)
        .def_property_readonly("has_field_mu", &ImpulseResponseField::has_field_mu)
        .def_property_readonly("has_field_Sigma", &ImpulseResponseField::has_field_Sigma)
        .def_property_readonly("mesh_vertices",
             []( const ImpulseResponseField& F )
             { return Eigen::MatrixXd(F.mesh().vertices().transpose()); },
             "Mesh vertex coordinates, shape (num_vertices, d).")
        .def_property_readonly("mesh_cells",
             []( const ImpulseResponseField& F )
             { return Eigen::MatrixXi(F.mesh().cells().transpose()); },
             "Mesh cell vertex indices, shape (num_cells, d+1).")
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
                 Eigen::MatrixXd out(static_cast<int>(mu.size()), F.dim());
                 for ( int ii = 0; ii < static_cast<int>(mu.size()); ++ii )
                 {
                     out.row(ii) = mu[ii].transpose();
                 }
                 return out;
             },
             "Per-sample means mu_i, shape (num_sample_points, d); empty if absent.")
        .def_property_readonly("sample_Sigma",
             []( const ImpulseResponseField& F )
             { return stack_from_matrices(F.sample_Sigma(), F.dim()); },
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
                 Eigen::MatrixXd points(k, F.dim());
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
}
