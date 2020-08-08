//! There are currently two meshing algorithms implemented:
//! * Surface Nets
//! * Greedy Quads

mod face;
mod greedy_quads;
mod quad;
mod quad_mesh_factory;
mod surface_nets;

pub use greedy_quads::{greedy_quads, GreedyQuadsVoxel};
pub use quad::Quad;
pub use quad_mesh_factory::{
    make_pos_norm_material_mesh_from_quads, make_pos_norm_tang_tex_mesh_from_quads,
    PosNormMaterialMesh, PosNormTangTexMesh,
};
pub use surface_nets::{surface_nets, SurfaceNetsOutput, SurfaceNetsVoxel};
