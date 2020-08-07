use ilattice3::{Extent, Normal, PlaneSpanInfo, Point};

/// The face of a rectangular prism of voxels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quad {
    pub extent: Extent,
    pub normal: Normal,
}

/// Useful structure for doing geometry with a quad.
pub struct QuadCornerInfo {
    pub span: PlaneSpanInfo,

    // Texture coordinatess.
    pub tex_max_u: f32,
    pub tex_max_v: f32,

    // Corner lattice points.
    pub min: Point,
    pub u_corner: Point,
    pub v_corner: Point,
    pub max: Point,
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
