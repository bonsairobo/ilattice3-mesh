use crate::{
    face::Face,
    quad::{Quad, QuadCornerInfo},
};

use ilattice3::{
    fill_extent, prelude::*, Direction, Extent, IsEmpty, Normal, PlaneSpanInfo, Point,
    VecLatticeMap, YLevelsIndexer, ALL_DIRECTIONS,
};
use rayon::prelude::*;
use std::cmp::{Ord, Ordering};
use std::{collections::HashMap, hash::Hash};

pub trait GreedyQuadsVoxel<M>: Clone + IsEmpty + Send + Sync {
    fn material(&self) -> M;
}

// TODO: There is still a lot of room for optimization in this algorithm, since I wrote it a long
// time ago without benchmarking it.
//
/// A data-parallelized version of the "Greedy Meshing" algorithm described here:
/// https://0fps.net/2012/06/30/meshing-in-a-minecraft-game/
pub fn greedy_quads<V, T, F>(voxels: &V, extent: Extent) -> F::Mesh
where
    V: GetWorldRef<Data = T> + Send + Sync,
    T: GreedyQuadsVoxel<<F as QuadMeshFactory>::Material>,
    F: QuadMeshFactory,
{
    let quads = boundary_quads::<_, _, F>(voxels, extent);

    F::make_mesh_from_quads(&quads)
}

/// This is the "greedy" part of finding quads.
fn grow_quad_extent(
    min: &Point,
    u: &Point,
    v: &Point,
    point_can_join_quad_fn: &impl Fn(&Point) -> bool,
) -> Extent {
    // Grow quad:
    // (1) in u direction until reaching a point that can't join.
    let mut max = *min;
    loop {
        let next_max = max + *u;
        if !point_can_join_quad_fn(&next_max) {
            break;
        }
        max = next_max;
    }
    // (2) in v direction until reaching row that can't join (entirely).
    let mut row = Extent::from_min_and_world_max(*min, max);
    'outer: loop {
        let next_row = row + *v;
        for row_p in next_row {
            if !point_can_join_quad_fn(&row_p) {
                break 'outer;
            }
        }
        row = next_row;
    }

    Extent::from_min_and_world_max(*min, row.get_world_max())
}

/// Greedily find visible quads (of the same type) in the plane.
fn boundary_quads_in_plane<V, T, M, F>(voxels: &V, extent: &Extent, plane: Quad) -> Vec<(Quad, M)>
where
    V: GetWorldRef<Data = T>,
    T: GreedyQuadsVoxel<M>,
    M: Clone + Hash + Eq + Send + Sync,
    F: QuadMeshFactory<Material = M>,
{
    let Quad {
        extent: quad_extent,
        normal,
    } = plane;
    let PlaneSpanInfo { u, v } = normal.get_plane_span_info();
    let n = Point::from(normal);

    let mut visited = VecLatticeMap::<_, YLevelsIndexer>::fill(quad_extent, false);

    let mut quads = vec![];
    for p in &quad_extent {
        let p_val = voxels.get_world_ref(&p);
        if p_val.is_empty() || visited.get_world(&p) {
            continue;
        }

        let face = Face::new(p, Normal::Vector(n));

        if !face.is_visible(voxels, extent) {
            continue;
        }

        let point_can_join_quad = |p: &Point| {
            let q_face = Face::new(*p, Normal::Vector(n));

            extent.contains_world(p)
                && !visited.get_world_ref(p)
                && q_face.is_visible(voxels, extent)
                && F::compatible(&p_val.material(), &voxels.get_world_ref(p).material())
        };

        let quad_extent = grow_quad_extent(&p, &u, &v, &point_can_join_quad);
        fill_extent(&mut visited, &quad_extent, true);
        quads.push((Quad::new(quad_extent, normal), p_val.material()));
    }

    quads
}

fn boundary_quads_unidirectional<V, T, F>(
    voxels: &V,
    extent: Extent,
    normal: Normal,
) -> Vec<(Quad, <F as QuadMeshFactory>::Material)>
where
    V: GetWorldRef<Data = T> + Send + Sync,
    T: GreedyQuadsVoxel<<F as QuadMeshFactory>::Material>,
    F: QuadMeshFactory,
{
    // Iterate over slices in the direction of their normal vector.
    // Note that we skip the left-most plane because it will be visited in the opposite normal
    // direction.
    //
    //               1st plane
    //                  v
    //  normal -->  | s | s | ... | s | s |
    //                ^
    //             1st slice
    //
    // For each plane, find visible quads of the same voxel type.

    let (start_slice_min, n_slices, slice_local_sup) = {
        let min = extent.get_minimum();
        let lsup = extent.get_local_supremum();

        match normal.into() {
            Direction::PosX => (min, lsup.x, [1, lsup.y, lsup.z].into()),
            Direction::PosY => (min, lsup.y, [lsup.x, 1, lsup.z].into()),
            Direction::PosZ => (min, lsup.z, [lsup.x, lsup.y, 1].into()),
            Direction::NegX => (
                min + [lsup.x - 1, 0, 0].into(),
                lsup.x,
                [1, lsup.y, lsup.z].into(),
            ),
            Direction::NegY => (
                min + [0, lsup.y - 1, 0].into(),
                lsup.y,
                [lsup.x, 1, lsup.z].into(),
            ),
            Direction::NegZ => (
                min + [0, 0, lsup.z - 1].into(),
                lsup.z,
                [lsup.x, lsup.y, 1].into(),
            ),
        }
    };

    (0..n_slices)
        .into_par_iter()
        .map(|i| {
            let quad = Quad::new(
                Extent::from_min_and_local_supremum(
                    start_slice_min + Point::from(normal) * i,
                    slice_local_sup,
                ),
                normal,
            );

            boundary_quads_in_plane::<_, _, _, F>(voxels, &extent, quad)
        })
        .flatten()
        .collect()
}

/// Returns all same-type quads of visible faces (only intersecting one voxel). The set of quads is
/// not unique and is not guaranteed to be optimal.
fn boundary_quads<V, T, F>(
    voxels: &V,
    extent: Extent,
) -> Vec<(Quad, <F as QuadMeshFactory>::Material)>
where
    V: GetWorldRef<Data = T> + Send + Sync,
    T: GreedyQuadsVoxel<<F as QuadMeshFactory>::Material>,
    F: QuadMeshFactory,
{
    ALL_DIRECTIONS
        .par_iter()
        .cloned()
        .map(|d| boundary_quads_unidirectional::<_, _, F>(voxels, extent, Normal::Axis(d)))
        .flatten()
        .collect()
}

const QUAD_VERTEX_PERM: [usize; 6] = [0, 1, 2, 2, 1, 3];

/// A trait to make `greedy_quads` more generic in the kinds of meshes it can produce.
pub trait QuadMeshFactory {
    type Material: Clone + Eq + Hash + Send + Sync;
    type Mesh;

    /// Determines whether two voxels can be part of the same quad.
    fn compatible(m1: &Self::Material, m2: &Self::Material) -> bool;

    /// Transforms `quads`, each labeled with some `Material`, into a mesh.
    fn make_mesh_from_quads(quads: &[(Quad, Self::Material)]) -> Self::Mesh;
}

/// A `QuadMeshFactory` that produces one `PosNormTangTex` mesh per material.
pub struct PosNormTangTexQuadMeshFactory<M> {
    marker: std::marker::PhantomData<M>,
}

impl<M> QuadMeshFactory for PosNormTangTexQuadMeshFactory<M>
where
    M: Clone + Eq + Hash + Send + Sync,
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
            let mesh = material_meshes.entry(material.clone()).or_default();
            Self::add_quad_vertices_to_mesh(quad, mesh);
        }

        material_meshes
    }
}

impl<M> PosNormTangTexQuadMeshFactory<M>
where
    M: Clone + Eq + Hash,
{
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

#[cfg(test)]
mod test {
    use super::*;

    use ilattice3::{FnLatticeMap, VecLatticeMap, YLevelsIndexer};
    use std::io::Write;

    #[derive(Clone, PartialEq)]
    struct Voxel(u16);

    impl IsEmpty for Voxel {
        fn is_empty(&self) -> bool {
            self.0 == 0
        }
    }

    impl GreedyQuadsVoxel<u16> for Voxel {
        fn material(&self) -> u16 {
            self.0
        }
    }

    const EXTENT_RADIUS: i32 = 32;

    fn cubes_in_sphere(p: &Point) -> Voxel {
        let sq_dist = p.dot(p);

        if sq_dist < EXTENT_RADIUS - 1 {
            Voxel(1)
        } else {
            Voxel(0)
        }
    }

    #[test]
    fn benchmark() {
        let sample_extent = Extent::from_center_and_radius([0, 0, 0].into(), EXTENT_RADIUS);
        let samples = VecLatticeMap::<_, YLevelsIndexer>::copy_from_map(
            &FnLatticeMap::new(cubes_in_sphere),
            &sample_extent,
        );

        let start = std::time::Instant::now();
        let _output = greedy_quads::<_, _, PosNormTangTexQuadMeshFactory<u16>>(
            &samples,
            *samples.get_extent(),
        );
        let elapsed_micros = start.elapsed().as_micros();
        std::io::stdout()
            .write(format!("greedy_quads took {} micros\n", elapsed_micros).as_bytes())
            .unwrap();
    }
}
