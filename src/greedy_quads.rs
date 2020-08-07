use crate::{face::Face, quad::Quad, QuadMeshFactory};

use ilattice3::{
    fill_extent, prelude::*, Direction, Extent, IsEmpty, Normal, PlaneSpanInfo, Point,
    VecLatticeMap, YLevelsIndexer, ALL_DIRECTIONS,
};
use rayon::prelude::*;

pub trait GreedyQuadsVoxel: Copy + IsEmpty + Send + Sync {
    type Material;

    fn material(&self) -> Self::Material;
}

// TODO: There is still a lot of room for optimization in this algorithm, since I wrote it a long
// time ago without benchmarking it.
//
/// A data-parallelized version of the "Greedy Meshing" algorithm described here:
/// https://0fps.net/2012/06/30/meshing-in-a-minecraft-game/
pub fn greedy_quads<V, T, M, F>(voxels: &V, extent: Extent) -> F::Mesh
where
    V: GetWorldRef<Data = T> + Send + Sync,
    T: GreedyQuadsVoxel<Material = M>,
    F: QuadMeshFactory<M>,
    M: Eq + Send + Sync,
{
    let quads = boundary_quads(voxels, extent);

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
fn boundary_quads_in_plane<V, T, M>(voxels: &V, extent: &Extent, plane: Quad) -> Vec<(Quad, M)>
where
    V: GetWorldRef<Data = T>,
    T: GreedyQuadsVoxel<Material = M>,
    M: Eq,
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

        let point_can_join_quad = |q: &Point| {
            if !extent.contains_world(q) {
                return false;
            }
            let q_val = voxels.get_world_ref(q);

            !q_val.is_empty()
                && !visited.get_world_ref(q)
                && Face::new(*q, Normal::Vector(n)).is_visible(voxels, extent)
                && p_val.material() == q_val.material()
        };

        let quad_extent = grow_quad_extent(&p, &u, &v, &point_can_join_quad);
        fill_extent(&mut visited, &quad_extent, true);
        quads.push((Quad::new(quad_extent, normal), p_val.material()));
    }

    quads
}

fn boundary_quads_unidirectional<V, T, M>(
    voxels: &V,
    extent: Extent,
    normal: Normal,
) -> Vec<(Quad, M)>
where
    V: GetWorldRef<Data = T> + Send + Sync,
    T: GreedyQuadsVoxel<Material = M>,
    M: Eq + Send + Sync,
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

            boundary_quads_in_plane(voxels, &extent, quad)
        })
        .flatten()
        .collect()
}

/// Returns all same-type quads of visible faces (only intersecting one voxel). The set of quads is
/// not unique and is not guaranteed to be optimal.
fn boundary_quads<V, T, M>(voxels: &V, extent: Extent) -> Vec<(Quad, M)>
where
    V: GetWorldRef<Data = T> + Send + Sync,
    T: GreedyQuadsVoxel<Material = M>,
    M: Eq + Send + Sync,
{
    ALL_DIRECTIONS
        .par_iter()
        .cloned()
        .map(|d| boundary_quads_unidirectional(voxels, extent, Normal::Axis(d)))
        .flatten()
        .collect()
}

#[cfg(test)]
mod test {
    use super::*;

    use crate::PosNormTangTexQuadMeshFactory;

    use ilattice3::{FnLatticeMap, VecLatticeMap, YLevelsIndexer};
    use std::io::Write;

    #[derive(Clone, Copy, PartialEq)]
    struct Voxel(u16);

    impl IsEmpty for Voxel {
        fn is_empty(&self) -> bool {
            self.0 == 0
        }
    }

    impl GreedyQuadsVoxel for Voxel {
        type Material = u16;

        fn material(&self) -> Self::Material {
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
        let _output = greedy_quads::<_, _, _, PosNormTangTexQuadMeshFactory<u16>>(
            &samples,
            *samples.get_extent(),
        );
        let elapsed_micros = start.elapsed().as_micros();
        std::io::stdout()
            .write(format!("greedy_quads took {} micros\n", elapsed_micros).as_bytes())
            .unwrap();
    }
}
