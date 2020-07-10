use ilattice3::{Extent, Normal, PlaneSpanInfo, Point};
use std::cmp::{Ord, Ordering};

/// The face of a rectangular prism of voxels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quad {
    pub extent: Extent,
    pub normal: Normal,
}

/// Useful structure for doing geometry with a quad.
pub struct QuadCornerInfo {
    span: PlaneSpanInfo,

    // Texture coordinatess.
    tex_max_u: f32,
    tex_max_v: f32,

    // Corner lattice points.
    min: Point,
    u_corner: Point,
    v_corner: Point,
    max: Point,
}

pub struct QuadVertices {
    pub positions: [[f32; 3]; 4],
    pub tex_coords: [[f32; 2]; 4],
    pub normal: [f32; 3],
    pub tangent: [f32; 3],
}

impl Quad {
    pub fn new(extent: Extent, normal: Normal) -> Self {
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
    pub fn mesh_vertices(&self) -> QuadVertices {
        let n: Point = self.normal.into();

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

    pub fn get_edges(&self) -> [[[f32; 3]; 2]; 4] {
        let n: Point = self.normal.into();

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
