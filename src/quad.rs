use crate::face::Face;

use ilattice3 as lat;
use ilattice3::{ChunkedLattice, Direction, IsEmpty, Lattice, Normal, PlaneSpanInfo, ALL_DIRECTIONS};
use rayon::prelude::*;
use std::{
    cmp::{Ord, Ordering},
    collections::HashMap,
};

/// The face of a rectangular prism of voxels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quad {
    pub extent: lat::Extent,
    pub normal: Normal,
}

/// Useful structure for doing geometry with a quad.
pub struct QuadCornerInfo {
    span: PlaneSpanInfo,

    // Texture coordinatess.
    tex_max_u: f32,
    tex_max_v: f32,

    // Corner lattice points.
    min: lat::Point,
    u_corner: lat::Point,
    v_corner: lat::Point,
    max: lat::Point,
}

pub struct QuadVertices {
    pub positions: [[f32; 3]; 4],
    pub tex_coords: [[f32; 2]; 4],
    pub normal: [f32; 3],
    pub tangent: [f32; 3],
}

impl Quad {
    pub fn new(extent: lat::Extent, normal: Normal) -> Self {
        Quad { extent, normal }
    }

    pub fn get_corner_info(&self) -> QuadCornerInfo {
        let min = self.extent.get_minimum();
        let max = self.extent.get_local_max();
        let PlaneSpanInfo { u, v } = self.normal.get_plane_span_info();
        let u_max = max.dot(&u);
        let v_max = max.dot(&v);

        QuadCornerInfo {
            span: PlaneSpanInfo { u, v },

            min,
            u_corner: min + u * u_max,
            v_corner: min + v * v_max,
            max: min + max,

            // Span info guarantees positive coordinates.
            tex_max_u: (u_max + 1) as f32,
            tex_max_v: (v_max + 1) as f32,
        }
    }

    /// Extends the given `Vertices` object with the quad vertex attributes. Vertices are in world
    /// space, so we don't need a separate mesh and transform for each voxel (only for different
    /// voxel types).
    pub fn extend_mesh_vertices(&self) -> QuadVertices {
        let n: lat::Point = self.normal.into();

        let QuadCornerInfo {
            span: PlaneSpanInfo { u, v },
            min,
            u_corner,
            v_corner,
            max,
            tex_max_u,
            tex_max_v,
        } = self.get_corner_info();

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

        QuadVertices {
            positions,
            tex_coords,
            normal: n.into(),
            tangent: u.into(),
        }
    }

    pub fn iter_boundary_points(&self) -> Box<dyn Iterator<Item = lat::Point>> {
        if self.extent.volume() == 1 {
            Box::new(self.extent.into_iter())
        } else {
            let QuadCornerInfo {
                span: PlaneSpanInfo { u, v },
                min,
                u_corner,
                v_corner,
                max,
                ..
            } = self.get_corner_info();

            let walls = vec![
                lat::Extent::from_min_and_world_max(min, u_corner - u),
                lat::Extent::from_min_and_world_max(min + v, v_corner),
                lat::Extent::from_min_and_world_max(u_corner, max - v),
                lat::Extent::from_min_and_world_max(v_corner + u, max),
            ];

            Box::new(walls.into_iter().filter(|w| !w.is_empty()).flatten())
        }
    }

    fn build_adjacency_list(
        &self,
        self_index: QuadIndex,
        index_lattices: &HashMap<lat::Point, Lattice<Option<QuadIndex>>>,
    ) -> Vec<QuadIndex> {
        // Look up the indices for faces adjacent to faces on the perimeter of the quad. We are
        // guaranteed to only get faces in or adjacent to the quad, since there shouldn't be
        // non-null voxels on top of the quad (else it wouldn't be visible). This logic is similar
        // to `adjacent_visible_face_values`, but differs because
        //   (1) we have a map of lattices keyed by normal vector instead of just a single lattice
        //   (2) we can assume all adjacent faces are visible or null
        //   (3) we only need to look at boundary points of the quad
        let mut adjacent_indices = vec![self_index];
        for p in self.iter_boundary_points() {
            let quad_boundary_face = Face::new(p, self.normal);
            for adj_face in &quad_boundary_face.adjacent_faces_with_axes() {
                let lat = &index_lattices[&adj_face.normal.into()];
                if !lat.get_extent().contains_world(&adj_face.point) {
                    continue;
                }
                if let Some(index) = lat.get_world(&adj_face.point) {
                    adjacent_indices.push(*index);
                }
            }
        }
        adjacent_indices.sort();
        adjacent_indices.dedup();

        adjacent_indices
    }

    pub fn get_edges(&self) -> [[[f32; 3]; 2]; 4] {
        let n: lat::Point = self.normal.into();

        let QuadCornerInfo {
            span: PlaneSpanInfo { u, v },
            min,
            u_corner,
            v_corner,
            max,
            ..
        } = self.get_corner_info();

        let n_sign = (n.x + n.y + n.z).signum();
        let which_plane = if n_sign > 0 { n } else { [0, 0, 0].into() };

        let min: [f32; 3] = (min + which_plane).into();
        let u_corner: [f32; 3] = (u_corner + u + which_plane).into();
        let v_corner: [f32; 3] = (v_corner + v + which_plane).into();
        let max: [f32; 3] = (max + u + v + which_plane).into();

        [
            [min, u_corner],
            [min, v_corner],
            [u_corner, max],
            [v_corner, max],
        ]
    }
}

/// This is the "greedy" part of finding quads.
fn grow_quad_extent(
    min: &lat::Point,
    u: &lat::Point,
    v: &lat::Point,
    point_can_join_quad_fn: &impl Fn(&lat::Point) -> bool,
) -> lat::Extent {
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
    let mut row = lat::Extent::from_min_and_world_max(*min, max);
    'outer: loop {
        let next_row = row + *v;
        for row_p in next_row {
            if !point_can_join_quad_fn(&row_p) {
                break 'outer;
            }
        }
        row = next_row;
    }

    lat::Extent::from_min_and_world_max(*min, row.get_world_max())
}

/// Greedily find visible quads (of the same type) in the plane.
fn boundary_quads_in_plane<T>(voxels: &Lattice<T>, plane: Quad) -> Vec<(Quad, T)>
where
    T: Copy + Default + IsEmpty + Ord + PartialEq,
{
    let Quad { extent, normal } = plane;
    let PlaneSpanInfo { u, v } = normal.get_plane_span_info();
    let n = lat::Point::from(normal);

    let mut visited = Lattice::<_, lat::YLevelsIndexer>::fill(extent, false);

    let mut quads = vec![];
    for p in &extent {
        let p_val = voxels.get_world(&p);
        if p_val.is_empty() || *visited.get_world(&p) {
            continue;
        }

        let face = Face::new(p, Normal::Vector(n));

        if !face.is_visible(voxels) {
            continue;
        }

        let point_can_join_quad = |p: &lat::Point| {
            let q_face = Face::new(*p, Normal::Vector(n));

            extent.contains_world(p)
                && !visited.get_world(p)
                && q_face.is_visible(voxels)
                && *p_val == *voxels.get_world(p)
        };

        let quad_extent = grow_quad_extent(&p, &u, &v, &point_can_join_quad);
        visited.fill_extent(&quad_extent, true);
        quads.push((Quad::new(quad_extent, normal), *p_val));
    }

    quads
}

fn boundary_quads_unidirectional<T>(
    voxels: &Lattice<T>,
    extent: lat::Extent,
    normal: Normal,
) -> Vec<(Quad, T)>
where
    T: Copy + Default + IsEmpty + Ord + PartialEq + Send + Sync,
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
                lat::Extent::from_min_and_local_supremum(
                    start_slice_min + lat::Point::from(normal) * i,
                    slice_local_sup,
                ),
                normal,
            );

            boundary_quads_in_plane(voxels, quad)
        })
        .flatten()
        .collect()
}

/// Returns all same-type quads of visible faces (only intersecting one voxel). The set of quads is
/// not unique and is not guaranteed to be optimal.
pub fn boundary_quads<T>(voxels: &Lattice<T>, extent: lat::Extent) -> Vec<(Quad, T)>
where
    T: Copy + Default + IsEmpty + Ord + PartialEq + Send + Sync,
{
    ALL_DIRECTIONS
        .par_iter()
        .cloned()
        .map(|d| boundary_quads_unidirectional(voxels, extent, Normal::Axis(d)))
        .flatten()
        .collect()
}

/// Returns all same-type quads of visible faces (only intersecting one voxel). The set of quads is
/// not unique and is not guaranteed to be optimal.
pub fn boundary_quads_chunked<T>(voxels: &ChunkedLattice<T>) -> Vec<(Quad, T)>
where
    T: Copy + Default + IsEmpty + Ord + PartialEq + Send + Sync,
{
    voxels
        .chunk_keys()
        .cloned()
        .collect::<Vec<lat::Point>>()
        .into_par_iter()
        .map(|chunk_key| {
            // Need a padded extent to check adjacent points, but we only want to create quads
            // containing points in the chunk.
            let consider_extent = voxels.extent_for_chunk_key(&chunk_key);
            let padded_extent = consider_extent.padded(1);

            boundary_quads(
                &voxels.copy_extent_into_new_lattice(padded_extent),
                consider_extent,
            )
        })
        .flatten()
        .collect()
}

#[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
pub struct QuadIndex(usize);

fn build_quad_index_lattice(
    normal_dir: Direction,
    extent: lat::Extent,
    quads: &[Quad],
) -> Lattice<Option<QuadIndex>> {
    let mut lattice = Lattice::fill(extent, None);
    for (i, q) in quads.iter().enumerate() {
        if q.normal.as_axis() != Normal::Axis(normal_dir) {
            continue;
        }
        lattice.fill_extent(&q.extent, Some(QuadIndex(i)));
    }

    lattice
}

/// Returns indices of adjacent quads for each quad.
#[allow(dead_code)]
pub fn build_quad_adjacency_lists(extent: lat::Extent, quads: &[Quad]) -> Vec<Vec<QuadIndex>> {
    // O(quads) time algorithm using O(voxels) memory. Create a lattice for each normal direction,
    // then write the quad index of each face into its points. On a second pass, look at points
    // adjacent to each quad to determine the adjacent quad indices.

    let index_lattices: HashMap<lat::Point, Lattice<Option<QuadIndex>>> = ALL_DIRECTIONS
        .par_iter()
        .cloned()
        .map(|d| (d.into(), build_quad_index_lattice(d, extent, quads)))
        .collect();

    quads
        .par_iter()
        .enumerate()
        .map(|(i, q)| q.build_adjacency_list(QuadIndex(i), &index_lattices))
        .collect()
}

// ████████╗███████╗███████╗████████╗███████╗
// ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝██╔════╝
//    ██║   █████╗  ███████╗   ██║   ███████╗
//    ██║   ██╔══╝  ╚════██║   ██║   ╚════██║
//    ██║   ███████╗███████║   ██║   ███████║
//    ╚═╝   ╚══════╝╚══════╝   ╚═╝   ╚══════╝

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::assert_elements_eq;

    #[test]
    fn test_quad_iter_boundary_points_pos_x_normal_1x1() {
        let extent = lat::Extent::from_min_and_world_max((1, 1, 1).into(), (1, 1, 1).into());
        let quad = Quad::new(extent, Normal::Axis(Direction::PosX));

        let boundary_points = quad.iter_boundary_points().collect();
        assert_elements_eq(&boundary_points, &vec![(1, 1, 1).into()]);
    }

    #[test]
    fn test_quad_iter_boundary_points_pos_x_normal_2x1() {
        let extent = lat::Extent::from_min_and_world_max((1, 1, 1).into(), (1, 2, 1).into());
        let quad = Quad::new(extent, Normal::Axis(Direction::PosX));

        let boundary_points = quad.iter_boundary_points().collect();
        assert_elements_eq(&boundary_points, &vec![(1, 1, 1).into(), (1, 2, 1).into()]);
    }

    #[test]
    fn test_quad_iter_boundary_points_pos_x_normal_2x2() {
        let extent = lat::Extent::from_min_and_world_max((1, 1, 1).into(), (1, 2, 2).into());
        let quad = Quad::new(extent, Normal::Axis(Direction::PosX));

        let boundary_points = quad.iter_boundary_points().collect();
        assert_elements_eq(
            &boundary_points,
            &vec![
                (1, 1, 1).into(),
                (1, 2, 1).into(),
                (1, 1, 2).into(),
                (1, 2, 2).into(),
            ],
        );
    }

    #[test]
    fn test_quad_iter_boundary_points_pos_z_normal() {
        let extent = lat::Extent::from_min_and_world_max((1, 1, 1).into(), (3, 5, 1).into());
        let quad = Quad::new(extent, Normal::Axis(Direction::PosZ));

        let boundary_points = quad.iter_boundary_points().collect();
        assert_elements_eq(
            &boundary_points,
            &vec![
                (1, 1, 1).into(),
                (2, 1, 1).into(),
                (3, 1, 1).into(),
                (3, 2, 1).into(),
                (3, 3, 1).into(),
                (3, 4, 1).into(),
                (3, 5, 1).into(),
                (2, 5, 1).into(),
                (1, 5, 1).into(),
                (1, 4, 1).into(),
                (1, 3, 1).into(),
                (1, 2, 1).into(),
            ],
        );
    }

    #[test]
    fn test_quad_iter_boundary_points_neg_y_normal() {
        let extent = lat::Extent::from_min_and_world_max((2, -1, 0).into(), (3, -1, 3).into());
        let quad = Quad::new(extent, Normal::Axis(Direction::NegY));

        let boundary_points = quad.iter_boundary_points().collect();
        assert_elements_eq(
            &boundary_points,
            &vec![
                (2, -1, 0).into(),
                (3, -1, 0).into(),
                (3, -1, 1).into(),
                (3, -1, 2).into(),
                (3, -1, 3).into(),
                (3, -1, 3).into(),
                (2, -1, 3).into(),
                (2, -1, 2).into(),
                (2, -1, 1).into(),
            ],
        );
    }
}
