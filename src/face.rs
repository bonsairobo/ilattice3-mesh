use ilattice3 as lat;
use ilattice3::{prelude::*, IsEmpty, Normal, VecLatticeMap};

/// One face of a voxel.
pub struct Face {
    pub point: lat::Point,
    pub normal: Normal,
}

impl Face {
    pub fn new(point: lat::Point, normal: Normal) -> Self {
        Face { point, normal }
    }

    pub fn adjacent_point(&self) -> lat::Point {
        self.point + self.normal.into()
    }

    pub fn is_visible<T, I>(&self, voxels: &VecLatticeMap<T, I>) -> bool
    where
        T: IsEmpty,
        I: lat::Indexer,
    {
        let adj_p = self.adjacent_point();

        !voxels.get_extent().contains_world(&adj_p) || voxels.get_world_ref(&adj_p).is_empty()
    }
}
