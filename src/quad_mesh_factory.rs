use crate::quad::{Quad, QuadCornerInfo};

use ilattice3::{normal::PlaneSpanInfo, Point};
use std::cmp::{Ord, Ordering};
use std::{collections::HashMap, hash::Hash};

const QUAD_VERTEX_PERM: [usize; 6] = [0, 1, 2, 2, 1, 3];

/// Produces a single mesh where each quad has one material assigned.
pub fn make_pos_norm_material_mesh_from_quads<M>(quads: &[(Quad, M)]) -> PosNormMaterialMesh<M>
where
    M: Copy,
{
    // Group the quad vertices, keyed by material.
    let mut mesh = PosNormMaterialMesh::default();
    for (quad, material) in quads.iter() {
        add_quad_vertices_to_pos_norm_material_mesh(quad, material, &mut mesh);
    }

    mesh
}

fn add_quad_vertices_to_pos_norm_material_mesh<M>(
    quad: &Quad,
    material: &M,
    mesh: &mut PosNormMaterialMesh<M>,
) where
    M: Copy,
{
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

/// Produces one `PosNormTangTex` mesh per material.
pub fn make_pos_norm_tang_tex_mesh_from_quads<M>(
    quads: &[(Quad, M)],
) -> HashMap<M, PosNormTangTexMesh>
where
    M: Copy + Eq + Hash,
{
    // Group the quad vertices, keyed by material.
    let mut material_meshes = HashMap::new();
    for (quad, material) in quads.iter() {
        let mesh = material_meshes.entry(*material).or_default();
        add_quad_vertices_to_pos_norm_tang_tex_mesh(quad, mesh);
    }

    material_meshes
}

fn add_quad_vertices_to_pos_norm_tang_tex_mesh(quad: &Quad, mesh: &mut PosNormTangTexMesh) {
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

#[derive(Default)]
pub struct PosNormTangTexMesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub tangents: Vec<[f32; 4]>,
    pub tex_coords: Vec<[f32; 2]>,
    pub indices: Vec<usize>,
}
