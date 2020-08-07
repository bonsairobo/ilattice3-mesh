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
    PosNormMaterialMesh, PosNormMaterialQuadMeshFactory, PosNormTangTexMesh,
    PosNormTangTexQuadMeshFactory, QuadMeshFactory,
};
pub use surface_nets::{surface_nets, SurfaceNetsOutput, SurfaceNetsVoxel};
