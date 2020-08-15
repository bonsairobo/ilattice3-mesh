use crate::{MaterialVoxel, SignedDistanceVoxel};

use ilattice3::{prelude::*, Indexer, Point, CUBE_CORNERS};

/// Uses a kernel to average the adjacent materials for each surface point.
pub fn calculate_material_weights<V, T, I>(voxels: &V, surface_points: &[Point]) -> Vec<[f32; 4]>
where
    V: GetExtent + GetLinear<Data = T> + HasIndexer<Indexer = I>,
    T: MaterialVoxel + SignedDistanceVoxel,
    I: Indexer,
{
    let sup = voxels.get_extent().get_local_supremum();

    // Precompute the offsets for cube corners, like we do in surface nets.
    let mut linear_offsets = [0; 8];
    for (i, offset) in CUBE_CORNERS.iter().enumerate() {
        linear_offsets[i] = I::index_from_local_point(sup, offset);
    }

    let mut material_weights = vec![[0.0; 4]; surface_points.len()];

    for (i, p) in surface_points.iter().enumerate() {
        let p_linear = I::index_from_local_point(sup, p);
        let w = &mut material_weights[i];
        for offset in linear_offsets.iter() {
            let q_linear = p_linear + offset;
            let voxel = voxels.get_linear(q_linear);
            if voxel.distance() < 0.0 {
                let material_w = WEIGHT_TABLE[voxel.material_index()];
                w[0] += material_w[0];
                w[1] += material_w[1];
                w[2] += material_w[2];
                w[3] += material_w[3];
            }
        }
    }

    material_weights
}

// The current vertex format is limited to 4 numbers for material weights.
const WEIGHT_TABLE: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];
