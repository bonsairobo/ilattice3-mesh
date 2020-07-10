use crate::{
    face::Face,
    quad::{Quad, QuadVertices},
};

use ilattice3::{
    fill_extent, prelude::*, Direction, Extent, IsEmpty, Normal, PlaneSpanInfo, Point,
    VecLatticeMap, YLevelsIndexer, ALL_DIRECTIONS,
};
use rayon::prelude::*;
use std::{collections::HashMap, hash::Hash};

#[derive(Default)]
pub struct PosNormTangTexMesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub tangents: Vec<[f32; 4]>,
    pub tex_coords: Vec<[f32; 2]>,
    pub indices: Vec<usize>,
}

pub trait GreedyQuadsVoxel<M>: Clone + IsEmpty + PartialEq + Send + Sync {
    fn material(&self) -> M;
}

/// A data-parallelized version of the "Greedy Meshing" algorithm described here:
/// https://0fps.net/2012/06/30/meshing-in-a-minecraft-game/
pub fn greedy_quads<V, T, M>(voxels: &V, extent: Extent) -> HashMap<M, PosNormTangTexMesh>
where
    V: GetWorldRef<T> + Send + Sync,
    T: GreedyQuadsVoxel<M>,
    M: Clone + Eq + Hash + Send + Sync,
{
    let quads = boundary_quads(voxels, extent);

    make_mesh_vertices_from_quads(&quads)
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
    V: GetWorldRef<T>,
    T: GreedyQuadsVoxel<M>,
    M: Eq + Hash,
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
                // TODO: users might want to have unequal voxels that can still join the same quad
                // (i.e. if they look the same)
                && p_val.material() == voxels.get_world_ref(p).material()
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
    V: GetWorldRef<T> + Send + Sync,
    T: GreedyQuadsVoxel<M>,
    M: Eq + Hash + Send + Sync,
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
    V: GetWorldRef<T> + Send + Sync,
    T: GreedyQuadsVoxel<M>,
    M: Eq + Hash + Send + Sync,
{
    ALL_DIRECTIONS
        .par_iter()
        .cloned()
        .map(|d| boundary_quads_unidirectional(voxels, extent, Normal::Axis(d)))
        .flatten()
        .collect()
}

const QUAD_VERTEX_PERM: [usize; 6] = [0, 1, 2, 2, 1, 3];

#[derive(Default)]
struct QuadIndexIterator {
    iter: usize,
}

impl QuadIndexIterator {
    pub fn get_for_n_vertices(self, n_vertices: usize) -> Vec<usize> {
        self.take(n_vertices * 6 / 4).collect()
    }
}

impl Iterator for QuadIndexIterator {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        let ret = Some(4 * (self.iter / 6) + QUAD_VERTEX_PERM[self.iter % 6]);
        self.iter += 1;

        ret
    }
}

fn make_mesh_vertices_from_quads<M>(quads: &[(Quad, M)]) -> HashMap<M, PosNormTangTexMesh>
where
    M: Clone + Eq + Hash,
{
    // Group the quad vertices, keyed by material.
    let mut mesh_vertices = HashMap::new();
    for (q, material) in quads.iter() {
        let vertices = mesh_vertices
            .entry(material.clone())
            .or_insert_with(PosNormTangTexMesh::default);
        let QuadVertices {
            positions,
            tex_coords,
            normal,
            tangent: t,
        } = q.mesh_vertices();
        vertices.positions.extend(positions.iter());
        vertices.tex_coords.extend(tex_coords.iter());
        // Repeat the normal and tangent for each vertex.
        vertices.normals.extend_from_slice(&[normal; 4]);
        vertices
            .tangents
            .extend_from_slice(&[[t[0], t[1], t[2], 1.0]; 4]);
        vertices.indices.extend(
            QuadIndexIterator::default()
                .get_for_n_vertices(vertices.positions.len())
                .iter(),
        );
    }

    mesh_vertices
}
