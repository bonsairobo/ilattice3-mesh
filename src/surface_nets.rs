// Adapted from the surface-nets crate by khyperia.

use ilattice3 as lat;
use ilattice3::prelude::*;
use std::collections::HashMap;

pub trait SurfaceNetsVoxel {
    fn distance(&self) -> f32;
    fn material(&self) -> u8;
}

/// Returns the (positions, normals, indices) for the isosurface.
pub fn surface_nets<V, T>(voxels: &V) -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<usize>)
where
    V: GetExtent + GetWorld<T>,
    T: SurfaceNetsVoxel,
{
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut grid_to_index = HashMap::new();

    // Find all vertex positions. Addtionally, create a hashmap from grid position to vertex index.
    for point in voxels.get_extent().add_to_supremum(&[-1, -1, -1].into()) {
        if let Some((center, normal)) = estimate_surface_point(voxels, &point) {
            grid_to_index.insert(point, positions.len());
            positions.push(center);
            normals.push(normal);
        }
    }

    // Find all triangles, in the form of [index, index, index] triples.
    let mut indices = Vec::new();
    make_all_triangles(voxels, &grid_to_index, &positions, &mut indices);

    (positions, normals, indices)
}

// List of all edges in a cube.
const OFFSETS: [(usize, usize); 12] = [
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
// [0,1]. Returns the (position, normal).
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
fn estimate_surface_point<V, T>(voxels: &V, point: &lat::Point) -> Option<([f32; 3], [f32; 3])>
where
    V: GetExtent + GetWorld<T>,
    T: SurfaceNetsVoxel,
{
    // Get the signed distance values at each corner of this cube.
    let mut dists = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    for (i, dist) in dists.iter_mut().enumerate() {
        let cube_corner =
            *point + lat::Point::new((i & 1) as i32, ((i >> 1) & 1) as i32, ((i >> 2) & 1) as i32);
        *dist = voxels.get_world(&cube_corner).distance();
    }

    let edge_crossings = OFFSETS.iter().filter_map(|&(offset1, offset2)| {
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
// touching that boundary. (Well, really, two triangles). The "centers" are actually the vertex
// positions, found earlier. Also, make sure the triangles are facing the right way. There's some
// hellish off-by-one conditions and whatnot that make this code really gross.
fn make_all_triangles<V, T>(
    voxels: &V,
    grid_to_index: &HashMap<lat::Point, usize>,
    positions: &[[f32; 3]],
    indices: &mut Vec<usize>,
) where
    V: GetExtent + GetWorld<T>,
    T: SurfaceNetsVoxel,
{
    let extent = voxels.get_extent();
    let min = extent.get_minimum();
    for p in extent.add_to_supremum(&[-1, -1, -1].into()) {
        // Do edges parallel with the X axis
        if p.y != min.y && p.z != min.z {
            make_triangle(
                voxels,
                grid_to_index,
                positions,
                indices,
                &p,
                &[1, 0, 0].into(),
                &[0, 1, 0].into(),
                &[0, 0, 1].into(),
            );
        }
        // Do edges parallel with the Y axis
        if p.x != min.x && p.z != min.z {
            make_triangle(
                voxels,
                grid_to_index,
                positions,
                indices,
                &p,
                &[0, 1, 0].into(),
                &[0, 0, 1].into(),
                &[1, 0, 0].into(),
            );
        }
        // Do edges parallel with the Z axis
        if p.x != min.x && p.y != min.y {
            make_triangle(
                voxels,
                grid_to_index,
                positions,
                indices,
                &p,
                &[0, 0, 1].into(),
                &[1, 0, 0].into(),
                &[0, 1, 0].into(),
            );
        }
    }
}

fn make_triangle<V, T>(
    voxels: &V,
    grid_to_index: &HashMap<lat::Point, usize>,
    positions: &[[f32; 3]],
    indices: &mut Vec<usize>,
    p: &lat::Point,
    offset: &lat::Point,
    axis1: &lat::Point,
    axis2: &lat::Point,
) where
    V: GetWorld<T>,
    T: SurfaceNetsVoxel,
{
    let face_result = is_face(voxels, p, offset);

    if let FaceResult::NoFace = face_result {
        return;
    }

    // The triangle points, viewed face-front, look like this:
    // v1 v3
    // v2 v4
    let v1 = *grid_to_index.get(p).unwrap();
    let v2 = *grid_to_index.get(&(*p - *axis1)).unwrap();
    let v3 = *grid_to_index.get(&(*p - *axis2)).unwrap();
    let v4 = *grid_to_index.get(&(*p - *axis1 - *axis2)).unwrap();
    // optional addition to algorithm: split quad to triangles in a certain way
    let (p1, p2, p3, p4) = (positions[v1], positions[v2], positions[v3], positions[v4]);
    fn sq_dist(a: [f32; 3], b: [f32; 3]) -> f32 {
        let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];

        d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
    }
    let d14 = sq_dist(p1, p4);
    let d23 = sq_dist(p2, p3);
    // Split the quad along the shorter axis, rather than the longer one.
    if d14 < d23 {
        match face_result {
            FaceResult::NoFace => (),
            FaceResult::FacePositive => {
                indices.extend([v1, v2, v4, v1, v4, v3].iter());
            }
            FaceResult::FaceNegative => {
                indices.extend([v1, v4, v2, v1, v3, v4].iter());
            }
        }
    } else {
        match face_result {
            FaceResult::NoFace => (),
            FaceResult::FacePositive => {
                indices.extend([v2, v4, v3, v2, v3, v1].iter());
            }
            FaceResult::FaceNegative => {
                indices.extend([v2, v3, v4, v2, v1, v3].iter());
            }
        }
    }
}

enum FaceResult {
    NoFace,
    FacePositive,
    FaceNegative,
}

// Determine if the sign of the SDF flips between p and (p + offset)
fn is_face<V, T>(voxels: &V, p: &lat::Point, offset: &lat::Point) -> FaceResult
where
    V: GetWorld<T>,
    T: SurfaceNetsVoxel,
{
    let other = *p + *offset;
    match (
        voxels.get_world(p).distance() < 0.0,
        voxels.get_world(&other).distance() < 0.0,
    ) {
        (true, false) => FaceResult::FacePositive,
        (false, true) => FaceResult::FaceNegative,
        _ => FaceResult::NoFace,
    }
}
