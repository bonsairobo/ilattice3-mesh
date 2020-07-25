// Adapted from the surface-nets crate by khyperia.

use ilattice3 as lat;
use ilattice3::{prelude::*, Extent, HasIndexer, Indexer};
use std::{collections::HashMap, hash::Hash};

pub trait SurfaceNetsVoxel<M: Copy + Eq + Hash> {
    fn distance(&self) -> f32;
    fn material(&self) -> M;
}

pub struct SurfaceNetsOutput<M> {
    /// Coordinates of every voxel that intersects the isosurface.
    pub surface_points: Vec<Point>,
    /// The isosurface points. Parallel to `surface_points`.
    pub positions: Vec<[f32; 3]>,
    /// The isosurface normals. Parallel to `surface_points`.
    pub normals: Vec<[f32; 3]>,

    /// The triangles belonging to each material.
    pub indices_by_material: HashMap<M, Vec<usize>>,
}

/// Returns a mesh for the isosurface. Assumes `extent` is a padded voxel chunk, so the returned
/// mesh will be compatible with meshes of adjacent chunks. If a voxel within 2 units of a chunk
/// boundary changes, then all chunks adjacent to that boundary need to be re-meshed.
pub fn surface_nets<V, T, I, M>(voxels: &V, extent: &Extent) -> SurfaceNetsOutput<M>
where
    // It saves quite a bit of time to do linear indexing.
    V: GetLinear<Data = T> + HasIndexer<Indexer = I>,
    T: SurfaceNetsVoxel<M>,
    I: Indexer,
    M: Copy + Eq + Hash,
{
    let local_extent = extent.with_minimum([0, 0, 0].into());

    let (mut positions, normals, surface_points, voxel_to_index) =
        estimate_surface(voxels, &local_extent);

    // Positions were generated in local coordinates, so translate them back to world coordinates.
    let min: [f32; 3] = extent.get_minimum().into();
    for p in positions.iter_mut() {
        p[0] += min[0];
        p[1] += min[1];
        p[2] += min[2];
    }

    let indices_by_material = make_all_quads(
        voxels,
        &local_extent,
        &voxel_to_index,
        &positions,
        &surface_points,
    );

    SurfaceNetsOutput {
        positions,
        normals,
        indices_by_material,
        surface_points,
    }
}

// Find all vertex positions and normals. Also generate a map from grid position to vertex index
// to be used to look up vertices when generating quads.
fn estimate_surface<V, T, I, M>(
    voxels: &V,
    extent: &Extent,
) -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<Point>, Vec<usize>)
where
    V: GetLinear<Data = T> + HasIndexer<Indexer = I>,
    T: SurfaceNetsVoxel<M>,
    I: Indexer,
    M: Copy + Eq + Hash,
{
    // Precalculate these offsets to do faster linear indexing.
    let sup = extent.get_local_supremum();
    let mut corner_offsets = [0; 8];
    for (i, p) in CUBE_CORNERS.iter().enumerate() {
        corner_offsets[i] = I::index_from_local_point(sup, p);
    }

    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut surface_points = Vec::new();
    let mut voxel_to_index = vec![0; extent.volume()];
    for p in extent.add_to_supremum(&[-1, -1, -1].into()) {
        // Get the corners of the cube "at" point p.
        let p_linear = I::index_from_local_point(sup, &p);
        let mut corner_indices = [0; 8];
        for i in 0..8 {
            corner_indices[i] = p_linear + corner_offsets[i];
        }

        if let Some((surface_point, normal)) = estimate_surface_point(voxels, &corner_indices, &p) {
            voxel_to_index[p_linear] = positions.len();
            surface_points.push(p);
            positions.push(surface_point);
            normals.push(normal);
        }
    }

    (positions, normals, surface_points, voxel_to_index)
}

const CUBE_EDGES: [(usize, usize); 12] = [
    (0b000, 0b001),
    (0b000, 0b010),
    (0b000, 0b100),
    (0b001, 0b011),
    (0b001, 0b101),
    (0b010, 0b011),
    (0b010, 0b110),
    (0b011, 0b111),
    (0b100, 0b101),
    (0b100, 0b110),
    (0b101, 0b111),
    (0b110, 0b111),
];

const CUBE_CORNERS: [lat::Point; 8] = [
    lat::Point { x: 0, y: 0, z: 0 },
    lat::Point { x: 1, y: 0, z: 0 },
    lat::Point { x: 0, y: 1, z: 0 },
    lat::Point { x: 1, y: 1, z: 0 },
    lat::Point { x: 0, y: 0, z: 1 },
    lat::Point { x: 1, y: 0, z: 1 },
    lat::Point { x: 0, y: 1, z: 1 },
    lat::Point { x: 1, y: 1, z: 1 },
];

// Consider the grid-aligned cube where `point` is the minimal corner. Find a point inside this cube
// that is approximately on the isosurface.
//
// This is done by estimating, for each cube edge, where the isosurface crosses the edge (if it
// does at all). Then the estimated surface point is the average of these edge crossings.
fn estimate_surface_point<V, T, M>(
    voxels: &V,
    corner_indices: &[usize],
    point: &lat::Point,
) -> Option<([f32; 3], [f32; 3])>
where
    V: GetLinear<Data = T>,
    T: SurfaceNetsVoxel<M>,
    M: Copy + Eq + Hash,
{
    // Get the signed distance values at each corner of this cube.
    let mut dists = [0.0; 8];
    let mut num_negative = 0;
    for (i, dist) in dists.iter_mut().enumerate() {
        let d = voxels.get_linear(corner_indices[i]).distance();
        *dist = d;
        if d < 0.0 {
            num_negative += 1;
        }
    }

    if num_negative == 0 || num_negative == 8 {
        // No crossings.
        return None;
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

// Given two cube corners, find the point between them where the SDF is zero.
// (This might not exist).
fn estimate_surface_edge_intersection(
    offset1: usize,
    offset2: usize,
    value1: f32,
    value2: f32,
) -> Option<[f32; 3]> {
    if (value1 < 0.0) == (value2 < 0.0) {
        return None;
    }

    let interp1 = value1 / (value1 - value2);
    let interp2 = 1.0 - interp1;
    let position = [
        (offset1 & 1) as f32 * interp2 + (offset2 & 1) as f32 * interp1,
        ((offset1 >> 1) & 1) as f32 * interp2 + ((offset2 >> 1) & 1) as f32 * interp1,
        ((offset1 >> 2) & 1) as f32 * interp2 + ((offset2 >> 2) & 1) as f32 * interp1,
    ];

    Some(position)
}

// For every edge that crosses the isosurface, make a quad between the "centers" of the four cubes
// touching that surface. The "centers" are actually the vertex positions found earlier. Also,
// make sure the triangles are facing the right way. See the comments on `maybe_make_quad` to help
// with understanding the indexing.
fn make_all_quads<V, T, I, M>(
    voxels: &V,
    extent: &Extent,
    voxel_to_index: &[usize],
    positions: &[[f32; 3]],
    surface_points: &[Point],
) -> HashMap<M, Vec<usize>>
where
    V: GetLinear<Data = T> + HasIndexer<Indexer = I>,
    T: SurfaceNetsVoxel<M>,
    I: Indexer,
    M: Copy + Eq + Hash,
{
    let mut material_indices = HashMap::new();
    let min = extent.get_minimum();
    debug_assert_eq!(min, [0, 0, 0].into());
    let sup = extent.get_local_supremum();
    let x_stride = I::index_from_local_point(sup, &[1, 0, 0].into());
    let y_stride = I::index_from_local_point(sup, &[0, 1, 0].into());
    let z_stride = I::index_from_local_point(sup, &[0, 0, 1].into());

    // NOTE: The checks against iter_max prevent us from making quads on the 3 maximal planes of the
    // grid. This is necessary to avoid redundant quads when meshing adjacent chunks (assuming this
    // will be used on a chunked voxel grid).

    let iter_extent = extent.add_to_supremum(&[-1, -1, -1].into());
    let iter_max = iter_extent.get_local_max();
    for p in surface_points.iter() {
        let p_linear = I::index_from_local_point(sup, p);
        // Do edges parallel with the X axis
        if p.y != min.y && p.z != min.z && p.x != iter_max.x {
            maybe_make_quad(
                voxels,
                voxel_to_index,
                positions,
                p_linear,
                p_linear + x_stride,
                y_stride,
                z_stride,
                &mut material_indices,
            );
        }
        // Do edges parallel with the Y axis
        if p.x != min.x && p.z != min.z && p.y != iter_max.y {
            maybe_make_quad(
                voxels,
                voxel_to_index,
                positions,
                p_linear,
                p_linear + y_stride,
                z_stride,
                x_stride,
                &mut material_indices,
            );
        }
        // Do edges parallel with the Z axis
        if p.x != min.x && p.y != min.y && p.z != iter_max.z {
            maybe_make_quad(
                voxels,
                voxel_to_index,
                positions,
                p_linear,
                p_linear + z_stride,
                x_stride,
                y_stride,
                &mut material_indices,
            );
        }
    }

    material_indices
}

// This is where the "dual" nature of surface nets comes into play.
//
// The surface point s was found somewhere inside of the cube "at" index i1.
//
//       x ---- x
//      /      /|
//     x ---- x |
//     |   s  | x
//     |      |/
//    i1 --- i2
//
// And now we want to find the quad between i1 and i2 where s is a corner of the quad.
//
//          s
//         /|
//        / |
//       |  |
//   i1  |  |  i2
//       | /
//       |/
//
// If A is (of the three grid axes) the axis between i1 and i2,
//
//       A
//   i1 ---> i2
//
// then we must find the other 3 quad corners by moving along the other two axes (those orthogonal
// to A) in the negative directions; these are axis B and axis C.
fn maybe_make_quad<V, T, M>(
    voxels: &V,
    voxel_to_index: &[usize],
    positions: &[[f32; 3]],
    i1: usize,
    i2: usize,
    axis_b_stride: usize,
    axis_c_stride: usize,
    material_indices: &mut HashMap<M, Vec<usize>>,
) where
    V: GetLinear<Data = T>,
    T: SurfaceNetsVoxel<M>,
    M: Copy + Eq + Hash,
{
    let voxel1 = voxels.get_linear(i1);
    let voxel2 = voxels.get_linear(i2);

    let face_result = is_face(voxel1.distance(), voxel2.distance());

    if let FaceResult::NoFace = face_result {
        return;
    }

    // The triangle points, viewed face-front, look like this:
    // v1 v3
    // v2 v4
    let v1 = voxel_to_index[i1];
    let v2 = voxel_to_index[i1 - axis_b_stride];
    let v3 = voxel_to_index[i1 - axis_c_stride];
    let v4 = voxel_to_index[i1 - axis_b_stride - axis_c_stride];
    let (pos1, pos2, pos3, pos4) = (positions[v1], positions[v2], positions[v3], positions[v4]);
    // Split the quad along the shorter axis, rather than the longer one.
    let (quad, material) = if sq_dist(pos1, pos4) < sq_dist(pos2, pos3) {
        match face_result {
            FaceResult::NoFace => unreachable!(),
            FaceResult::FacePositive => ([v1, v2, v4, v1, v4, v3], voxel1.material()),
            FaceResult::FaceNegative => ([v1, v4, v2, v1, v3, v4], voxel2.material()),
        }
    } else {
        match face_result {
            FaceResult::NoFace => unreachable!(),
            FaceResult::FacePositive => ([v2, v4, v3, v2, v3, v1], voxel1.material()),
            FaceResult::FaceNegative => ([v2, v3, v4, v2, v1, v3], voxel2.material()),
        }
    };
    let indices = material_indices.entry(material).or_insert_with(Vec::new);
    indices.extend_from_slice(&quad);
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
fn is_face(d1: f32, d2: f32) -> FaceResult {
    match (d1 < 0.0, d2 < 0.0) {
        (true, false) => FaceResult::FacePositive,
        (false, true) => FaceResult::FaceNegative,
        _ => FaceResult::NoFace,
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use ilattice3::{FnLatticeMap, VecLatticeMap, YLevelsIndexer};
    use std::io::Write;

    #[derive(Clone)]
    struct Voxel(f32);

    impl SurfaceNetsVoxel<u8> for Voxel {
        fn material(&self) -> u8 {
            1
        }

        fn distance(&self) -> f32 {
            self.0
        }
    }

    fn waves_sdf(p: &Point) -> Voxel {
        let n = 10.0;
        let val = ((p.x as f32 / 32.0) * n * std::f32::consts::PI / 2.0).sin()
            + ((p.y as f32 / 32.0) * n * std::f32::consts::PI / 2.0).sin()
            + ((p.z as f32 / 32.0) * n * std::f32::consts::PI / 2.0).sin();

        Voxel(val)
    }

    #[test]
    fn benchmark() {
        let sample_extent = Extent::from_center_and_radius([0, 0, 0].into(), 32);
        let samples = VecLatticeMap::<_, YLevelsIndexer>::copy_from_map(
            &FnLatticeMap::new(waves_sdf),
            &sample_extent,
        );

        let start = std::time::Instant::now();
        let _output = surface_nets(&samples, samples.get_extent());
        let elapsed_micros = start.elapsed().as_micros();
        std::io::stdout()
            .write(format!("surface_nets took {} micros\n", elapsed_micros).as_bytes())
            .unwrap();
    }
}
