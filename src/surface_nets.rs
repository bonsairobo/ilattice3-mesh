// Adapted from the surface-nets crate by khyperia.

use ilattice3 as lat;
use ilattice3::{prelude::*, Extent};
use std::{collections::HashMap, hash::Hash};

pub trait SurfaceNetsVoxel<M: Copy + Eq + Hash> {
    fn distance(&self) -> f32;
    fn material(&self) -> M;
}

#[derive(Default)]
pub struct PosNormMesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub indices: Vec<usize>,
}

impl PosNormMesh {
    fn copy_triangle(&mut self, tri: usize, other: &Self, reindex: &mut [Option<usize>]) {
        let tri_start = 3 * tri;
        let tri_end = tri_start + 3;
        for i in tri_start..tri_end {
            let other_index = other.indices[i];
            let self_index = *reindex[other_index].get_or_insert_with(|| {
                let new_index = self.positions.len();
                self.positions.push(other.positions[other_index]);
                self.normals.push(other.normals[other_index]);

                new_index
            });
            self.indices.push(self_index);
        }
    }
}

/// Returns the map from material ID to (positions, normals, indices) for the isosurface.
pub fn surface_nets<V, T, M>(voxels: &V, extent: &Extent) -> HashMap<M, PosNormMesh>
where
    V: GetWorldRef<T>,
    T: SurfaceNetsVoxel<M>,
    M: Copy + Eq + Hash,
{
    // Find all vertex positions. Addtionally, create a hashmap from grid position to vertex index.
    let (positions, normals, voxel_to_index) = estimate_surface(voxels, extent);

    // Find all triangles (2 per quad), in the form of [index, index, index] triples.
    let (indices, quad_materials) = make_all_quads(voxels, extent, &voxel_to_index, &positions);

    // Assign each triangle a material.
    split_mesh_by_material(
        &quad_materials,
        &PosNormMesh {
            positions,
            normals,
            indices,
        },
    )
}

/// For each triangle, choose a material and copy the triangle into a new mesh.
fn split_mesh_by_material<M>(
    quad_materials: &[M],
    orig_mesh: &PosNormMesh,
) -> HashMap<M, PosNormMesh>
where
    M: Copy + Eq + Hash,
{
    // There's probably a more efficient way to do this, but the code is simple.
    let mut material_to_mesh = HashMap::new();
    let mut material_to_reindex = HashMap::new();
    let num_triangles = orig_mesh.indices.len() / 3;
    for i in 0..num_triangles {
        let material = quad_materials[i / 2];
        // Copy the triangle to the mesh for this material.
        let mesh = material_to_mesh
            .entry(material)
            .or_insert_with(PosNormMesh::default);
        let reindex = material_to_reindex
            .entry(material)
            .or_insert(vec![None; orig_mesh.positions.len()]);
        mesh.copy_triangle(i, &orig_mesh, reindex);
    }

    material_to_mesh
}

fn estimate_surface<V, T, M>(
    voxels: &V,
    extent: &Extent,
) -> (Vec<[f32; 3]>, Vec<[f32; 3]>, HashMap<lat::Point, usize>)
where
    V: GetWorldRef<T>,
    T: SurfaceNetsVoxel<M>,
    M: Copy + Eq + Hash,
{
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut voxel_to_index = HashMap::new();
    for point in extent.add_to_supremum(&[-1, -1, -1].into()) {
        if let Some((surface_point, normal)) = estimate_surface_point(voxels, &point) {
            voxel_to_index.insert(point, positions.len());
            positions.push(surface_point);
            normals.push(normal);
        }
    }

    (positions, normals, voxel_to_index)
}

// List of all edges in a cube.
const CUBE_EDGES: [(usize, usize); 12] = [
    (0b000, 0b001), // ((0, 0, 0), (0, 0, 1)),
    (0b000, 0b010), // ((0, 0, 0), (0, 1, 0)),
    (0b000, 0b100), // ((0, 0, 0), (1, 0, 0)),
    (0b001, 0b011), // ((0, 0, 1), (0, 1, 1)),
    (0b001, 0b101), // ((0, 0, 1), (1, 0, 1)),
    (0b010, 0b011), // ((0, 1, 0), (0, 1, 1)),
    (0b010, 0b110), // ((0, 1, 0), (1, 1, 0)),
    (0b011, 0b111), // ((0, 1, 1), (1, 1, 1)),
    (0b100, 0b101), // ((1, 0, 0), (1, 0, 1)),
    (0b100, 0b110), // ((1, 0, 0), (1, 1, 0)),
    (0b101, 0b111), // ((1, 0, 1), (1, 1, 1)),
    (0b110, 0b111), // ((1, 1, 0), (1, 1, 1)),
];

// Find the vertex position for this grid: it will be somewhere within the cube with coordinates
// [0,1]. Returns the (position, normal, material).
//
// How?
//
// First, for each edge in the cube, find if that edge crosses the SDF boundary - i.e. one
// point is positive, one point is negative.
//
// Second, calculate the "weighted midpoint" between these points (see
// estimate_surface_edge_intersection).
//
// Third, take the average of all these points for all edges (for edges that have crossings).
fn estimate_surface_point<V, T, M>(voxels: &V, point: &lat::Point) -> Option<([f32; 3], [f32; 3])>
where
    V: GetWorldRef<T>,
    T: SurfaceNetsVoxel<M>,
    M: Copy + Eq + Hash,
{
    // Get the signed distance values at each corner of this cube.
    let mut dists = [0.0; 8];
    for (i, dist) in dists.iter_mut().enumerate() {
        let cube_corner =
            *point + lat::Point::new((i & 1) as i32, ((i >> 1) & 1) as i32, ((i >> 2) & 1) as i32);
        *dist = voxels.get_world_ref(&cube_corner).distance();
    }

    let edge_crossings = CUBE_EDGES.iter().filter_map(|&(offset1, offset2)| {
        estimate_surface_edge_intersection(offset1, offset2, dists[offset1], dists[offset2])
    });

    let mut count = 0;
    let mut sum = [0.0, 0.0, 0.0];
    for position in edge_crossings {
        count += 1;
        sum[0] += position[0];
        sum[1] += position[1];
        sum[2] += position[2];
    }

    if count == 0 {
        None
    } else {
        // Calculate the normal as the gradient of the distance field.
        let normal_x = (dists[0b001] + dists[0b011] + dists[0b101] + dists[0b111])
            - (dists[0b000] + dists[0b010] + dists[0b100] + dists[0b110]);
        let normal_y = (dists[0b010] + dists[0b011] + dists[0b110] + dists[0b111])
            - (dists[0b000] + dists[0b001] + dists[0b100] + dists[0b101]);
        let normal_z = (dists[0b100] + dists[0b101] + dists[0b110] + dists[0b111])
            - (dists[0b000] + dists[0b001] + dists[0b010] + dists[0b011]);
        let normal_len = (normal_x * normal_x + normal_y * normal_y + normal_z * normal_z).sqrt();

        Some((
            [
                sum[0] / count as f32 + point.x as f32 + 0.5,
                sum[1] / count as f32 + point.y as f32 + 0.5,
                sum[2] / count as f32 + point.z as f32 + 0.5,
            ],
            [
                normal_x / normal_len,
                normal_y / normal_len,
                normal_z / normal_len,
            ],
        ))
    }
}

// Given two points, A and B, find the point between them where the SDF is zero.
// (This might not exist).
// A and B are specified via A=coord+offset1 and B=coord+offset2, because code
// is weird.
fn estimate_surface_edge_intersection(
    offset1: usize,
    offset2: usize,
    value1: f32,
    value2: f32,
) -> Option<[f32; 3]> {
    if (value1 < 0.0) == (value2 < 0.0) {
        return None;
    }

    let interp = value1 / (value1 - value2);
    let position = [
        (offset1 & 1) as f32 * (1.0 - interp) + (offset2 & 1) as f32 * interp,
        ((offset1 >> 1) & 1) as f32 * (1.0 - interp) + ((offset2 >> 1) & 1) as f32 * interp,
        ((offset1 >> 2) & 1) as f32 * (1.0 - interp) + ((offset2 >> 2) & 1) as f32 * interp,
    ];

    Some(position)
}

// For every edge that crosses the boundary, make a quad between the "centers" of the four cubes
// touching that boundary. The "centers" are actually the vertex positions, found earlier. Also,
// make sure the triangles are facing the right way. There's some hellish off-by-one conditions and
// whatnot that make this code really gross.
fn make_all_quads<V, T, M>(
    voxels: &V,
    extent: &Extent,
    voxel_to_index: &HashMap<lat::Point, usize>,
    positions: &[[f32; 3]],
) -> (Vec<usize>, Vec<M>)
where
    V: GetWorldRef<T>,
    T: SurfaceNetsVoxel<M>,
    M: Copy + Eq + Hash,
{
    let mut indices = Vec::new();
    let mut quad_materials = Vec::new();
    let min = extent.get_minimum();
    for p in extent.add_to_supremum(&[-1, -1, -1].into()) {
        // Do edges parallel with the X axis
        if p.y != min.y && p.z != min.z {
            maybe_make_quad(
                voxels,
                voxel_to_index,
                positions,
                &p,
                &[1, 0, 0].into(),
                &[0, 1, 0].into(),
                &[0, 0, 1].into(),
                &mut indices,
                &mut quad_materials,
            );
        }
        // Do edges parallel with the Y axis
        if p.x != min.x && p.z != min.z {
            maybe_make_quad(
                voxels,
                voxel_to_index,
                positions,
                &p,
                &[0, 1, 0].into(),
                &[0, 0, 1].into(),
                &[1, 0, 0].into(),
                &mut indices,
                &mut quad_materials,
            );
        }
        // Do edges parallel with the Z axis
        if p.x != min.x && p.y != min.y {
            maybe_make_quad(
                voxels,
                voxel_to_index,
                positions,
                &p,
                &[0, 0, 1].into(),
                &[1, 0, 0].into(),
                &[0, 1, 0].into(),
                &mut indices,
                &mut quad_materials,
            );
        }
    }

    (indices, quad_materials)
}

fn maybe_make_quad<V, T, M>(
    voxels: &V,
    voxel_to_index: &HashMap<lat::Point, usize>,
    positions: &[[f32; 3]],
    p1: &lat::Point,
    offset: &lat::Point,
    axis1: &lat::Point,
    axis2: &lat::Point,
    indices: &mut Vec<usize>,
    materials: &mut Vec<M>,
) where
    V: GetWorldRef<T>,
    T: SurfaceNetsVoxel<M>,
    M: Copy + Eq + Hash,
{
    let p2 = *p1 + *offset;
    let face_result = is_face(voxels, p1, &p2);

    if let FaceResult::NoFace = face_result {
        return;
    }

    // The triangle points, viewed face-front, look like this:
    // v1 v3
    // v2 v4
    let v1 = *voxel_to_index.get(p1).unwrap();
    let v2 = *voxel_to_index.get(&(*p1 - *axis1)).unwrap();
    let v3 = *voxel_to_index.get(&(*p1 - *axis2)).unwrap();
    let v4 = *voxel_to_index.get(&(*p1 - *axis1 - *axis2)).unwrap();
    let (pos1, pos2, pos3, pos4) = (positions[v1], positions[v2], positions[v3], positions[v4]);
    // Split the quad along the shorter axis, rather than the longer one.
    let (quad, material) = if sq_dist(pos1, pos4) < sq_dist(pos2, pos3) {
        match face_result {
            FaceResult::NoFace => unreachable!(),
            FaceResult::FacePositive => (
                [v1, v2, v4, v1, v4, v3],
                voxels.get_world_ref(&p1).material(),
            ),
            FaceResult::FaceNegative => (
                [v1, v4, v2, v1, v3, v4],
                voxels.get_world_ref(&p2).material(),
            ),
        }
    } else {
        match face_result {
            FaceResult::NoFace => unreachable!(),
            FaceResult::FacePositive => (
                [v2, v4, v3, v2, v3, v1],
                voxels.get_world_ref(&p1).material(),
            ),
            FaceResult::FaceNegative => (
                [v2, v3, v4, v2, v1, v3],
                voxels.get_world_ref(&p2).material(),
            ),
        }
    };
    indices.extend(quad.iter());
    materials.push(material);
}

fn sq_dist(a: [f32; 3], b: [f32; 3]) -> f32 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];

    d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
}

enum FaceResult {
    NoFace,
    FacePositive,
    FaceNegative,
}

// Determine if the sign of the SDF flips between p1 and p2
fn is_face<V, T, M>(voxels: &V, p1: &lat::Point, p2: &lat::Point) -> FaceResult
where
    V: GetWorldRef<T>,
    T: SurfaceNetsVoxel<M>,
    M: Copy + Eq + Hash,
{
    match (
        voxels.get_world_ref(p1).distance() < 0.0,
        voxels.get_world_ref(p2).distance() < 0.0,
    ) {
        (true, false) => FaceResult::FacePositive,
        (false, true) => FaceResult::FaceNegative,
        _ => FaceResult::NoFace,
    }
}
