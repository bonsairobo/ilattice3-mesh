//! There are currently two meshing algorithms implemented:
//! * Surface Nets
//! * Greedy Quads

mod face;
mod greedy_quads;
mod material_weights;
mod quad;
mod quad_mesh_factory;
mod surface_nets;
mod voxel_traits;

pub use greedy_quads::{greedy_quads, GreedyQuadsVoxel};
pub use material_weights::calculate_material_weights;
pub use quad::Quad;
pub use quad_mesh_factory::{
    make_pos_norm_material_mesh_from_quads, make_pos_norm_tang_tex_mesh_from_quads,
    PosNormMaterialMesh, PosNormTangTexMesh,
};
pub use surface_nets::{surface_nets, SurfaceNetsOutput};
pub use voxel_traits::{MaterialVoxel, SignedDistanceVoxel};
