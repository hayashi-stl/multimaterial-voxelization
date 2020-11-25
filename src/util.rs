use float_ord::FloatOrd;
use num_traits::Float;
use petgraph::graph::Edges;
use petgraph::prelude::*;
use petgraph::EdgeType;
use std::hash::{Hash, Hasher};
use tri_mesh::prelude::*;

pub type Vec2 = Vector2<f64>;

/// A `Vec2` that can be hashed
#[derive(Copy, Clone, Debug)]
pub struct HashVec2(pub Vec2);

impl HashVec2 {
    fn float_ord(self) -> (FloatOrd<f64>, FloatOrd<f64>) {
        (FloatOrd(self.0.x), FloatOrd(self.0.y))
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

pub trait GraphEx {
    type Edge;
    type Type: EdgeType;

    fn indegree(&self, node: NodeIndex) -> usize;

    fn outdegree(&self, node: NodeIndex) -> usize;

    /// Sum of indegree and outdegree
    fn degree(&self, node: NodeIndex) -> usize {
        self.indegree(node) + self.outdegree(node)
    }

    /// Convenience method to get incoming edges.
    fn edges_in(&self, node: NodeIndex) -> Edges<Self::Edge, Self::Type>;
}

impl<N, E> GraphEx for Graph<N, E> {
    type Edge = E;
    type Type = Directed;

    fn indegree(&self, node: NodeIndex) -> usize {
        self.edges_in(node).count()
    }

    fn outdegree(&self, node: NodeIndex) -> usize {
        self.edges(node).count()
    }

    fn edges_in(&self, node: NodeIndex) -> Edges<Self::Edge, Self::Type> {
        self.edges_directed(node, Direction::Incoming)
    }
}
