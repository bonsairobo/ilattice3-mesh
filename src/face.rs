use ilattice3::{prelude::*, Extent, IsEmpty, Normal};

/// One face of a voxel.
pub struct Face {
    pub point: Point,
    pub normal: Normal,
}

impl Face {
    pub fn new(point: Point, normal: Normal) -> Self {
        Face { point, normal }
    }

    pub fn adjacent_point(&self) -> Point {
        self.point + self.normal.into()
    }

    pub fn is_visible<V, T>(&self, voxels: &V, extent: &Extent) -> bool
    where
        V: GetWorldRef<T>,
        T: IsEmpty,
    {
        let adj_p = self.adjacent_point();

        !extent.contains_world(&adj_p) || voxels.get_world_ref(&adj_p).is_empty()
    }
}
