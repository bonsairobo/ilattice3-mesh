pub trait SignedDistanceVoxel {
    fn distance(&self) -> f32;
}

pub trait MaterialVoxel {
    fn material_index(&self) -> usize;
}
