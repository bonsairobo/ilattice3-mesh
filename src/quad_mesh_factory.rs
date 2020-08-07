use crate::quad::{Quad, QuadCornerInfo};

use ilattice3::{PlaneSpanInfo, Point};
use std::cmp::{Ord, Ordering};
use std::{collections::HashMap, hash::Hash};

const QUAD_VERTEX_PERM: [usize; 6] = [0, 1, 2, 2, 1, 3];

/// A trait to make `greedy_quads` more generic in the kinds of meshes it can produce.
pub trait QuadMeshFactory {
    type Material: Copy + Eq + Hash + Send + Sync;
    type Mesh;

    /// Determines whether two voxels can be part of the same quad.
    fn compatible(m1: &Self::Material, m2: &Self::Material) -> bool;

    /// Transforms `quads`, each labeled with some `Material`, into a mesh.
    fn make_mesh_from_quads(quads: &[(Quad, Self::Material)]) -> Self::Mesh;
}

/// A `QuadMeshFactory` that produces a single `PosNormMaterial` mesh.
pub struct PosNormMaterialQuadMeshFactory<M> {
    marker: std::marker::PhantomData<M>,
}

impl<M> QuadMeshFactory for PosNormMaterialQuadMeshFactory<M>
where
    M: Copy + Eq + Hash + Send + Sync,
{
    type Material = M;
    type Mesh = PosNormMaterialMesh<M>;

    fn compatible(m1: &Self::Material, m2: &Self::Material) -> bool {
        *m1 == *m2
    }

    fn make_mesh_from_quads(quads: &[(Quad, M)]) -> Self::Mesh {
        // Group the quad vertices, keyed by material.
        let mut mesh = PosNormMaterialMesh::default();
        for (quad, material) in quads.iter() {
            Self::add_quad_vertices_to_mesh(quad, material, &mut mesh);
        }

        mesh
    }
}

impl<M> PosNormMaterialQuadMeshFactory<M>
where
    M: Copy,
{
    fn add_quad_vertices_to_mesh(quad: &Quad, material: &M, mesh: &mut PosNormMaterialMesh<M>) {
        let n: Point = quad.normal.into();

        let QuadCornerInfo {
            span: PlaneSpanInfo { u, v },
            min,
            u_corner,
            v_corner,
            max,
            ..
        } = quad.get_corner_info();

        let n_sign = (n.x + n.y + n.z).signum();
        let which_plane = if n_sign > 0 { n } else { [0, 0, 0].into() };

        let min: [f32; 3] = (min + which_plane).into();
        let u_corner: [f32; 3] = (u_corner + u + which_plane).into();
        let v_corner: [f32; 3] = (v_corner + v + which_plane).into();
        let max: [f32; 3] = (max + u + v + which_plane).into();

        // counter-clockwise winding
        let positions = match n_sign.cmp(&0) {
            Ordering::Greater => [min.into(), u_corner.into(), v_corner.into(), max.into()],
            Ordering::Less => [min.into(), v_corner.into(), u_corner.into(), max.into()],
            Ordering::Equal => panic!("Zero normal!"),
        };

        let index_start = mesh.positions.len();
        let indices = QUAD_VERTEX_PERM.iter().map(|i| index_start + i);

        mesh.positions.extend(&positions);
        mesh.normals.extend(&[n.into(); 4]);
        mesh.materials.extend(&[*material; 4]);
        mesh.indices.extend(indices);
    }
}

pub struct PosNormMaterialMesh<M> {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub materials: Vec<M>,
    pub indices: Vec<usize>,
}

// For some reason I can't derive Default unless M: Default.
impl<M> Default for PosNormMaterialMesh<M> {
    fn default() -> Self {
        PosNormMaterialMesh {
            positions: Vec::new(),
            normals: Vec::new(),
            materials: Vec::new(),
            indices: Vec::new(),
        }
    }
}

/// A `QuadMeshFactory` that produces one `PosNormTangTex` mesh per material.
pub struct PosNormTangTexQuadMeshFactory<M> {
    marker: std::marker::PhantomData<M>,
}

impl<M> QuadMeshFactory for PosNormTangTexQuadMeshFactory<M>
where
    M: Copy + Eq + Hash + Send + Sync,
{
    type Material = M;
    type Mesh = HashMap<M, PosNormTangTexMesh>;

    fn compatible(m1: &Self::Material, m2: &Self::Material) -> bool {
        *m1 == *m2
    }

    fn make_mesh_from_quads(quads: &[(Quad, M)]) -> Self::Mesh {
        // Group the quad vertices, keyed by material.
        let mut material_meshes = HashMap::new();
        for (quad, material) in quads.iter() {
            let mesh = material_meshes.entry(*material).or_default();
            Self::add_quad_vertices_to_mesh(quad, mesh);
        }

        material_meshes
    }
}

impl<M> PosNormTangTexQuadMeshFactory<M> {
    fn add_quad_vertices_to_mesh(quad: &Quad, mesh: &mut PosNormTangTexMesh) {
        let n: Point = quad.normal.into();

        let QuadCornerInfo {
            span: PlaneSpanInfo { u, v },
            min,
            u_corner,
            v_corner,
            max,
            tex_max_u,
            tex_max_v,
        } = quad.get_corner_info();

        let n_sign = (n.x + n.y + n.z).signum();
        let which_plane = if n_sign > 0 { n } else { [0, 0, 0].into() };

        let min: [f32; 3] = (min + which_plane).into();
        let u_corner: [f32; 3] = (u_corner + u + which_plane).into();
        let v_corner: [f32; 3] = (v_corner + v + which_plane).into();
        let max: [f32; 3] = (max + u + v + which_plane).into();

        // counter-clockwise winding
        let (positions, tex_coords) = match n_sign.cmp(&0) {
            Ordering::Greater => (
                [min.into(), u_corner.into(), v_corner.into(), max.into()],
                [
                    [0.0, 0.0],
                    [tex_max_u, 0.0],
                    [0.0, tex_max_v],
                    [tex_max_u, tex_max_v],
                ],
            ),
            Ordering::Less => (
                [min.into(), v_corner.into(), u_corner.into(), max.into()],
                [
                    [0.0, 0.0],
                    [0.0, tex_max_v],
                    [tex_max_u, 0.0],
                    [tex_max_u, tex_max_v],
                ],
            ),
            Ordering::Equal => panic!("Zero normal!"),
        };

        let tangent = [u.x as f32, u.y as f32, u.z as f32, 1.0];

        let index_start = mesh.positions.len();
        let indices = QUAD_VERTEX_PERM.iter().map(|i| index_start + i);

        mesh.positions.extend(&positions);
        mesh.tex_coords.extend(&tex_coords);
        mesh.normals.extend(&[n.into(); 4]);
        mesh.tangents.extend(&[tangent; 4]);
        mesh.indices.extend(indices);
    }
}

#[derive(Default)]
pub struct PosNormTangTexMesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub tangents: Vec<[f32; 4]>,
    pub tex_coords: Vec<[f32; 2]>,
    pub indices: Vec<usize>,
}
