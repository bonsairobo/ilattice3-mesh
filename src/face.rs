use ilattice3 as lat;
use ilattice3::{prelude::*, IsEmpty, Normal, PlaneSpanInfo, VecLatticeMap};

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

    #[allow(dead_code)]
    pub fn adjacent_faces_with_vectors(&self) -> [Face; 8] {
        let n_vec = self.normal.as_vector();

        let n: lat::Point = n_vec.into();
        let PlaneSpanInfo { u, v } = self.normal.get_plane_span_info();

        [
            Face::new(self.point + u, n_vec),
            Face::new(self.point - u, n_vec),
            Face::new(self.point + v, n_vec),
            Face::new(self.point - v, n_vec),
            Face::new(self.point + u + n, Normal::Vector(-u)),
            Face::new(self.point - u + n, Normal::Vector(u)),
            Face::new(self.point + v + n, Normal::Vector(-v)),
            Face::new(self.point - v + n, Normal::Vector(v)),
        ]
    }

    pub fn adjacent_faces_with_axes(&self) -> [Face; 8] {
        let n_axis = self.normal.as_axis();

        let n: lat::Point = self.normal.into();
        let PlaneSpanInfo { u, v } = self.normal.get_plane_span_info();

        [
            Face::new(self.point + u, n_axis),
            Face::new(self.point - u, n_axis),
            Face::new(self.point + v, n_axis),
            Face::new(self.point - v, n_axis),
            Face::new(self.point + u + n, Normal::Axis((-u).into())),
            Face::new(self.point - u + n, Normal::Axis(u.into())),
            Face::new(self.point + v + n, Normal::Axis((-v).into())),
            Face::new(self.point - v + n, Normal::Axis(v.into())),
        ]
    }

    /// Returns all values from visible faces that share an edge with the given visible face.
    #[allow(dead_code)]
    pub fn adjacent_visible_face_values<T, I>(&self, voxels: &VecLatticeMap<T, I>) -> Vec<T>
    where
        T: Copy + IsEmpty + Ord,
        I: lat::Indexer,
    {
        assert!(self.is_visible(voxels));

        let mut vals = vec![voxels.get_world(&self.point)];
        for adj_face in &self.adjacent_faces_with_vectors() {
            if !voxels.get_extent().contains_world(&adj_face.point) {
                continue;
            }
            let val = voxels.get_world(&adj_face.point);
            if !val.is_empty() && adj_face.is_visible(voxels) {
                vals.push(val);
            }
        }
        vals.sort();
        vals.dedup();

        vals
    }
}

// ████████╗███████╗███████╗████████╗███████╗
// ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝██╔════╝
//    ██║   █████╗  ███████╗   ██║   ███████╗
//    ██║   ██╔══╝  ╚════██║   ██║   ╚════██║
//    ██║   ███████╗███████║   ██║   ███████║
//    ╚═╝   ╚══════╝╚══════╝   ╚═╝   ╚══════╝

#[cfg(test)]
mod tests {
    use ilattice3 as lat;

    use super::*;
    use crate::test_util::assert_elements_eq;

    #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
    struct MyInt(i32);

    impl lat::IsEmpty for MyInt {
        fn is_empty(&self) -> bool {
            *self == MyInt(0)
        }
    }

    const TEST_LATTICE_VALUES: [i32; 18] = [1, 0, 3, 2, 3, 4, 0, 9, 5, 1, 8, 9, 1, 0, 0, 3, 4, 6];

    #[derive(Clone)]
    struct TestIndexer;

    impl lat::Indexer for TestIndexer {
        fn index_from_local_point(s: &lat::Point, p: &lat::Point) -> usize {
            (p.x * s.y * s.z + p.y * s.z + p.z) as usize
        }

        fn local_point_from_index(_s: &lat::Point, _i: usize) -> lat::Point {
            unimplemented!()
        }
    }

    fn make_test_lattice() -> Lattice<MyInt, TestIndexer> {
        let extent = lat::Extent::from_min_and_local_supremum((0, 0, 0).into(), (2, 3, 3).into());

        Lattice::new_with_indexer(
            extent,
            TestIndexer,
            TEST_LATTICE_VALUES.iter().map(|i| MyInt(*i)).collect(),
        )
    }

    #[test]
    fn test_adjacent_face_values_pos_x() {
        let lattice = make_test_lattice();
        let axis = Normal::Axis(lat::Direction::PosX);

        let face = Face::new((0, 1, 1).into(), axis);
        assert_elements_eq(
            &face.adjacent_visible_face_values(&lattice),
            &vec![MyInt(1), MyInt(3), MyInt(4), MyInt(8)],
        );

        let face = Face::new((1, 0, 1).into(), axis);
        assert_elements_eq(
            &face.adjacent_visible_face_values(&lattice),
            &vec![MyInt(1), MyInt(8), MyInt(9)],
        );

        let face = Face::new((0, 1, 2).into(), axis);
        assert_elements_eq(
            &face.adjacent_visible_face_values(&lattice),
            &vec![MyInt(3), MyInt(4), MyInt(6), MyInt(9)],
        );

        let face = Face::new((1, 2, 2).into(), axis);
        assert_elements_eq(
            &face.adjacent_visible_face_values(&lattice),
            &vec![MyInt(4), MyInt(6)],
        );
    }

    #[test]
    fn test_adjacent_face_values_neg_y() {
        let lattice = make_test_lattice();
        let axis = Normal::Axis(lat::Direction::NegY);

        let face = Face::new((0, 1, 1).into(), axis);
        assert_elements_eq(
            &face.adjacent_visible_face_values(&lattice),
            &vec![MyInt(1), MyInt(3), MyInt(8)],
        );

        let face = Face::new((1, 0, 1).into(), axis);
        assert_elements_eq(
            &face.adjacent_visible_face_values(&lattice),
            &vec![MyInt(1), MyInt(8), MyInt(9)],
        );

        let face = Face::new((1, 2, 1).into(), axis);
        assert_elements_eq(
            &face.adjacent_visible_face_values(&lattice),
            &vec![MyInt(1), MyInt(3), MyInt(4), MyInt(6)],
        );

        let face = Face::new((1, 2, 2).into(), axis);
        assert_elements_eq(
            &face.adjacent_visible_face_values(&lattice),
            &vec![MyInt(4), MyInt(6)],
        );
    }
}
