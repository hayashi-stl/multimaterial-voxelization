use float_ord::FloatOrd;
use num_traits::Float;
use petgraph::prelude::*;
use std::hash::{Hash, Hasher};
use tri_mesh::prelude::*;

pub type Vec2 = Vector2<f64>;

/// A `Vec2` that can be hashed
#[derive(Copy, Clone, Debug)]
pub struct HashVec2(pub Vec2);

impl HashVec2 {
    fn float_ord(self) -> (FloatOrd<f64>, FloatOrd<f64>) {
        (FloatOrd(self.0[0]), FloatOrd(self.0[1]))
    }
}

impl PartialEq for HashVec2 {
    fn eq(&self, other: &Self) -> bool {
        self.float_ord() == other.float_ord()
    }
}

impl Eq for HashVec2 {}

impl Hash for HashVec2 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.float_ord().hash(state)
    }
}

/// Something with an indegree and outdegree measure
pub trait Degree {
    fn indegree(&self, node: NodeIndex) -> usize;

    fn outdegree(&self, node: NodeIndex) -> usize;

    /// Sum of indegree and outdegree
    fn degree(&self, node: NodeIndex) -> usize {
        self.indegree(node) + self.outdegree(node)
    }
}

impl<N, E> Degree for Graph<N, E> {
    fn indegree(&self, node: NodeIndex) -> usize {
        self.edges_directed(node, Direction::Incoming).count()
    }

    fn outdegree(&self, node: NodeIndex) -> usize {
        self.edges(node).count()
    }
}
