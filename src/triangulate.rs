//! Implementation of polygon triangulation
//! that uses monotone decomposition and beautification
//! based on area/perimeter ratios

use float_ord::FloatOrd;
use fnv::{FnvHashMap, FnvHashSet};
use petgraph::prelude::*;
use std::collections::VecDeque;
use tri_mesh::prelude::*;

use crate::util::{GraphEx, HashVec2, Vec2};

/// A polygon structure to help with triangulation.
/// Polygons can have holes or multiple pieces.
#[derive(Clone, Debug)]
pub struct Polygon {
    boundary: Graph<Vec2, ()>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TriangulateError {
    UnbalancedDegreeVertex,
}

/// Used for the plane sweep.
/// Contains next vertex and
/// neighboring vertices of split lines in clockwise order.
#[derive(Debug)]
struct SweepVertex {
    target: NodeIndex,
    kind: SweepKind,
}

#[derive(Debug)]
enum SweepKind {
    Start(Option<NodeIndex>),
    End(Option<NodeIndex>),
    Continue([Option<NodeIndex>; 2], bool),
    Split([Option<NodeIndex>; 3]),
    Merge([Option<NodeIndex>; 3]),
}

impl SweepVertex {
    fn new(target: NodeIndex, kind: SweepKind) -> Self {
        Self { target, kind }
    }

    /// Adds the other node of a split line to the right of this node
    fn add_right(&mut self, node: NodeIndex, up: bool) {
        match &mut self.kind {
            SweepKind::Start(n) => *n = Some(node),
            SweepKind::End(_) => unreachable!(),
            SweepKind::Continue([a, b], flip) => *if *flip { a } else { b } = Some(node),
            SweepKind::Merge([_, n, _]) => *n = Some(node),
            // More complicated because we need to figure out which side (bottom or top)
            // the split line is on
            SweepKind::Split([b, _, t]) => *if up { t } else { b } = Some(node),
        }
    }

    /// Adds the other node of a split line to the left of this node
    fn add_left(&mut self, node: NodeIndex, up: bool) {
        match &mut self.kind {
            SweepKind::Start(_) => unreachable!(),
            SweepKind::End(n) => *n = Some(node),
            SweepKind::Continue([a, b], flip) => *if *flip { b } else { a } = Some(node),
            SweepKind::Split([_, n, _]) => *n = Some(node),

            // More complicated because we need to figure out which side (bottom or top)
            // the split line is on
            SweepKind::Merge([t, _, b]) => *if up { b } else { t } = Some(node),
        }
    }

    fn targets(self) -> Vec<NodeIndex> {
        match self.kind {
            SweepKind::Start(n) | SweepKind::End(n) => vec![n],

            SweepKind::Continue([a, b], _) => vec![a, b],

            SweepKind::Split([a, b, c]) | SweepKind::Merge([a, b, c]) => vec![a, b, c],
        }
        .into_iter()
        .flatten()
        .chain(std::iter::once(self.target))
        .collect()
    }
}

impl Polygon {
    /// Degree-4 or higher vertices get split into degree-2 vertices.
    /// Degree-0 vertices get removed.
    /// Errors if there's a vertex whose indegree does not equal its outdegree.
    pub fn fix_bad_degrees(boundary: &mut Graph<Vec2, ()>) -> Result<(), TriangulateError> {
        for node in boundary.node_indices() {
            if boundary.indegree(node) != boundary.outdegree(node) {
                return Err(TriangulateError::UnbalancedDegreeVertex);
            }
        }

        boundary.retain_nodes(|graph, n| graph.degree(n) > 0);

        let high_degree_nodes = boundary
            .node_indices()
            .filter(|n| boundary.degree(*n) > 2)
            .collect::<Vec<_>>();

        for node in &high_degree_nodes {
            let pos = boundary[*node];

            // Get edges around the node
            let mut targets = boundary
                .edges(*node)
                .map(|e| (e.target(), Direction::Outgoing))
                .chain(
                    boundary
                        .edges_directed(*node, Direction::Incoming)
                        .map(|e| (e.source(), Direction::Incoming)),
                )
                .collect::<Vec<_>>();

            targets.sort_by_key(|(n, dir)| {
                let diff = boundary[*n] - pos;
                FloatOrd(diff.y.atan2(diff.x))
            });

            // Pair outgoing edges with incoming edges and split vertex.
            // Optimize for not having self-intersections
            while targets.len() > 2 {
                let mut iter = targets.iter().cycle();
                let out_index = iter
                    .position(|(_, dir)| *dir == Direction::Outgoing)
                    .unwrap();
                let in_index = (iter
                    .position(|(_, dir)| *dir == Direction::Incoming)
                    .unwrap()
                    + out_index
                    + 1)
                    % targets.len();

                let new_node = boundary.add_node(pos);
                let out_node = targets[out_index].0;
                let in_node = targets[in_index].0;

                boundary.remove_edge(boundary.find_edge(in_node, *node).unwrap());
                boundary.remove_edge(boundary.find_edge(*node, out_node).unwrap());
                boundary.add_edge(in_node, new_node, ());
                boundary.add_edge(new_node, out_node, ());

                targets.remove(
                    targets
                        .iter()
                        .position(|x| *x == (in_node, Direction::Incoming))
                        .unwrap(),
                );
                targets.remove(
                    targets
                        .iter()
                        .position(|x| *x == (out_node, Direction::Outgoing))
                        .unwrap(),
                );
            }
        }

        Ok(())
    }

    pub fn from_boundary(mut boundary: Graph<Vec2, ()>) -> Result<Self, TriangulateError> {
        Self::fix_bad_degrees(&mut boundary)?;
        // Now every vertex is indegree-1 outdegree-1.
        //println!("graph: {:?}", boundary);

        Ok(Self { boundary })
    }

    fn monotone_cycles(sweep: FnvHashMap<NodeIndex, SweepVertex>) -> Vec<Vec<NodeIndex>> {
        let mut target_map = sweep
            .into_iter()
            .map(|(k, v)| (k, v.targets()))
            .collect::<FnvHashMap<_, _>>();

        target_map
            .iter()
            .for_each(|(k, v)| println!("node: {:?}, targets: {:?}", k, v));

        let mut cycles = vec![];

        // Extract cycles
        while let Some(start) = target_map.keys().next().copied() {
            let mut node = start;
            let mut prev = node;
            let mut cycle = vec![];

            while {
                cycle.push(node);
                let targets = target_map.get_mut(&node).unwrap();

                let new_node = if prev == node {
                    targets.pop().unwrap()
                } else {
                    todo!()
                };

                if targets.is_empty() {
                    target_map.remove(&node);
                }

                prev = node;
                node = new_node;
                node != start
            } {}

            cycles.push(cycle);
        }

        cycles
    }

    /// Decomposes the polygon into x-monotone polygons
    /// (actually monotone mountains, where one side is just a line segment)
    /// Returns a list of monotone mountains, where
    /// each item contains the vertices of the monotone mountain
    /// in circular order.
    fn monotone_decompose(&mut self) -> Vec<Vec<NodeIndex>> {
        let boundary = &mut self.boundary;

        // Plane sweep from left to right
        let mut nodes = boundary.node_indices().collect::<Vec<_>>();
        nodes.sort_by_key(|n| FloatOrd(boundary[*n].x));
        println!("Nodes: {:?}\n", nodes);

        // Because of the edge comparison that happens during a split/start vertex,
        // we can't let there be a vertical edge in the edge list at that time.
        // When a vertical edge is introduced, it needs to be continued immediately.
        let (nodes, index_map) = {
            let mut sorted = vec![];
            let mut map = FnvHashMap::default();

            nodes.reverse();
            while let Some(node) = nodes.pop() {
                if !map.contains_key(&node) {
                    map.insert(node, sorted.len());
                    sorted.push(node);

                    let node_a = boundary.edges_in(node).next().unwrap().source();
                    let node_b = boundary.edges(node).next().unwrap().target();

                    for other in vec![node_a, node_b] {
                        if boundary[node].x == boundary[other].x && !map.contains_key(&other) {
                            // Vertical edge. Add other vertex
                            nodes.push(other);
                        }
                    }
                }
            }

            (sorted, map)
        };

        // Edges sorted by y position.
        // Each edge also contains the vertex with an x position
        // >= its left coordinate and < the sweep line,
        // called the "helper" in https://people.csail.mit.edu/indyk/6.838-old/handouts/lec4.pdf
        let mut edges: Vec<(NodeIndex, NodeIndex, Option<NodeIndex>)> = vec![];
        let mut sweep_nodes = FnvHashMap::default();

        for (i, node) in nodes.into_iter().enumerate() {
            // Nodes exist because each node has an incoming edge and an outgoing edge.
            let mut node_a = boundary.edges_in(node).next().unwrap().source();
            let mut node_b = boundary.edges(node).next().unwrap().target();
            let out_node = node_b;
            let index_a = index_map[&node_a];
            let index_b = index_map[&node_b];

            let pos = boundary[node];

            // Edge to cut. Left node first.
            let mut cut = None;

            if index_a > i && index_b > i {
                // Split/start vertex
                if ((boundary[node_a] - pos).dot(boundary[node_b] - pos) < 0.0
                    && boundary[node_a].y > boundary[node_b].y)
                    || (boundary[node_a] - pos).perp_dot(boundary[node_b] - pos) < 0.0
                {
                    std::mem::swap(&mut node_a, &mut node_b);
                }

                // Add to edge list
                #[allow(irrefutable_let_patterns)]
                if let Ok(index) | Err(index) =
                    edges.binary_search_by_key(&FloatOrd(pos.y), |(l, r, _)| {
                        let l_pos = boundary[*l];
                        let r_pos = boundary[*r];
                        let t = (pos.x - l_pos.x) / (r_pos.x - l_pos.x);
                        FloatOrd(l_pos.y + (r_pos.y - l_pos.y) * t)
                    })
                {
                    // Add node as helper in the case of a start vertex
                    // to handle an edge case dealing with a split vertex
                    let helper = if index % 2 == 0 { Some(node) } else { None };
                    edges.insert(index, (node, node_a, helper));
                    edges.insert(index + 1, (node, node_b, helper));

                    if index % 2 != 0 {
                        // Split vertex
                        sweep_nodes.insert(
                            node,
                            SweepVertex::new(out_node, SweepKind::Split([None; 3])),
                        );

                        // Look at edges above and below and add a diagonal to the one
                        // with the rightmost left helper coordinate.
                        let (h_index, helper) = vec![edges[index - 1].2, edges[index + 2].2]
                            .into_iter()
                            .enumerate()
                            .flat_map(|(i, n)| n.map(|n| (i, n)))
                            .max_by_key(|(_, n)| FloatOrd(boundary[*n].x))
                            .unwrap();
                        cut = Some((helper, node, h_index == 0));

                        // Update helpers
                        edges[index - 1].2 = Some(node);
                        edges[index + 2].2 = Some(node);
                    } else {
                        // Start vertex
                        sweep_nodes
                            .insert(node, SweepVertex::new(out_node, SweepKind::Start(None)));
                    }
                }
            } else if index_a < i && index_b < i {
                // Merge/end vertex
                if ((boundary[node_a] - pos).dot(boundary[node_b] - pos) < 0.0
                    && boundary[node_a].y > boundary[node_b].y)
                    || (boundary[node_a] - pos).perp_dot(boundary[node_b] - pos) > 0.0
                {
                    std::mem::swap(&mut node_a, &mut node_b);
                }

                let index = edges
                    .iter()
                    .position(|(s, t, _)| (*s, *t) == (node_a, node))
                    .unwrap();

                if index % 2 != 0 {
                    // Merge vertex
                    sweep_nodes.insert(
                        node,
                        SweepVertex::new(out_node, SweepKind::Merge([None; 3])),
                    );

                    for (i, (_, _, helper)) in
                        vec![edges[index], edges[index + 1]].into_iter().enumerate()
                    {
                        if let Some(helper) = helper {
                            if helper != node_a && helper != node_b {
                                cut = Some((helper, node, i == 0));
                            }
                        }
                    }

                    // Update helpers
                    edges[index - 1].2 = Some(node);
                    edges[index + 2].2 = Some(node);
                } else {
                    // End vertex
                    sweep_nodes.insert(node, SweepVertex::new(out_node, SweepKind::End(None)));

                    if let Some(helper) = vec![edges[index], edges[index + 1]]
                        .into_iter()
                        .max_by_key(|(s, _, _)| index_map[s])
                        .unwrap()
                        .2
                    {
                        // Don't care about up/down
                        cut = Some((helper, node, false));
                    }
                }

                edges.remove(index + 1);
                edges.remove(index);
            } else {
                // Vertex crosses sweep line
                let mut flipped = false;
                if index_a > i {
                    std::mem::swap(&mut node_a, &mut node_b);
                    flipped = true
                }

                sweep_nodes.insert(
                    node,
                    SweepVertex::new(out_node, SweepKind::Continue([None; 2], flipped)),
                );

                let index = edges
                    .iter()
                    .position(|(s, t, _)| (*s, *t) == (node_a, node))
                    .unwrap();

                if let Some(helper) = edges[index].2 {
                    if helper != node_a {
                        cut = Some((helper, node, index % 2 == 1));
                    }
                }

                edges[index] = (node, node_b, None);
                edges[index ^ 1].2 = Some(node);
            }

            if let Some((cut_l, cut_r, up)) = cut {
                boundary.add_edge(cut_l, cut_r, ());
                boundary.add_edge(cut_r, cut_l, ());

                sweep_nodes.get_mut(&cut_l).unwrap().add_right(cut_r, up);
                sweep_nodes.get_mut(&cut_r).unwrap().add_left(cut_l, up);
            }
        }

        Self::monotone_cycles(sweep_nodes)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use fnv::FnvHasher;
    use petgraph::algo;
    use petgraph::data::{Element, FromElements};
    use std::hash::{Hash, Hasher};
    use std::num::Wrapping;

    fn create_graph<N>(vertices: Vec<N>, edges: Vec<(usize, usize)>) -> Graph<N, ()> {
        Graph::from_elements(
            vertices
                .into_iter()
                .map(|vertex| Element::Node { weight: vertex })
                .chain(edges.into_iter().map(|(s, t)| Element::Edge {
                    source: s,
                    target: t,
                    weight: (),
                })),
        )
    }

    fn create_decomposition(cycles: Vec<Vec<usize>>) -> FnvHashSet<Ring<NodeIndex>> {
        cycles
            .into_iter()
            .map(|v| Ring(v.into_iter().map(NodeIndex::new).collect()))
            .collect()
    }

    fn testable_decomposition(cycles: Vec<Vec<NodeIndex>>) -> FnvHashSet<Ring<NodeIndex>> {
        cycles
            .into_iter()
            .map(|v| Ring(v.into_iter().collect()))
            .collect()
    }

    #[derive(Clone, Debug, Eq)]
    struct Ring<T: Clone + Eq + Hash>(VecDeque<T>);

    impl<T: Clone + Eq + Hash> PartialEq for Ring<T> {
        fn eq(&self, other: &Self) -> bool {
            let mut rotatable = other.0.clone();
            (0..self.0.len()).into_iter().all(|_| {
                rotatable.rotate_right(1);
                rotatable == self.0
            })
        }
    }

    impl<T: Clone + Eq + Hash> Hash for Ring<T> {
        fn hash<H: Hasher>(&self, state: &mut H) {
            let int = self
                .0
                .iter()
                .map(|t| {
                    let mut hasher = FnvHasher::default();
                    t.hash(&mut hasher);
                    Wrapping(hasher.finish())
                })
                .product::<Wrapping<u64>>();

            int.hash(state)
        }
    }

    #[test]
    fn test_fix_bad_degrees_degree_0() {
        let mut graph = create_graph(
            vec![
                vec2(0.0, 0.0),
                vec2(1.0, 0.0),
                vec2(1.0, 1.0),
                vec2(0.0, 1.0),
            ],
            vec![(0, 1), (1, 2), (2, 0)],
        );
        let expected = create_graph(
            vec![vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(1.0, 1.0)],
            vec![(0, 1), (1, 2), (2, 0)],
        );

        let result = Polygon::fix_bad_degrees(&mut graph);
        assert_eq!(result, Ok(()));
        assert!(algo::is_isomorphic_matching(
            &graph,
            &expected,
            |x, y| x == y,
            |x, y| x == y
        ));
    }

    #[test]
    fn test_fix_bad_degrees_degree_1() {
        let mut graph = create_graph(
            vec![
                vec2(0.0, 0.0),
                vec2(1.0, 0.0),
                vec2(1.0, 1.0),
                vec2(0.0, 1.0),
            ],
            vec![(0, 1), (1, 2), (2, 3)],
        );

        let result = Polygon::fix_bad_degrees(&mut graph);
        assert_eq!(result, Err(TriangulateError::UnbalancedDegreeVertex));
    }

    #[test]
    fn test_fix_bad_degrees_degree_2() {
        let mut graph = create_graph(
            vec![
                vec2(0.0, 0.0),
                vec2(1.0, 0.0),
                vec2(1.0, 1.0),
                vec2(0.0, 1.0),
            ],
            vec![(0, 1), (1, 2), (2, 3), (3, 0)],
        );
        let expected = graph.clone();

        let result = Polygon::fix_bad_degrees(&mut graph);
        assert_eq!(result, Ok(()));
        assert!(algo::is_isomorphic_matching(
            &graph,
            &expected,
            |x, y| x == y,
            |x, y| x == y
        ));
    }

    #[test]
    fn test_fix_bad_degrees_degree_high() {
        let mut graph = create_graph(
            vec![
                vec2(0.0, 1.0),
                vec2(0.0, 0.0),
                vec2(1.0, 0.0),
                vec2(2.0, 0.0),
                vec2(3.0, 0.0),
                vec2(4.0, 0.0),
                vec2(5.0, 0.0),
            ],
            vec![
                (1, 2),
                (2, 0),
                (0, 1),
                (3, 4),
                (4, 0),
                (0, 3),
                (5, 6),
                (6, 0),
                (0, 5),
            ],
        );
        let expected = create_graph(
            vec![
                vec2(0.0, 1.0),
                vec2(0.0, 0.0),
                vec2(1.0, 0.0),
                vec2(2.0, 0.0),
                vec2(3.0, 0.0),
                vec2(4.0, 0.0),
                vec2(5.0, 0.0),
                vec2(0.0, 1.0),
                vec2(0.0, 1.0),
            ],
            vec![
                (1, 2),
                (2, 0),
                (0, 1),
                (3, 4),
                (4, 7),
                (7, 3),
                (5, 6),
                (6, 8),
                (8, 5),
            ],
        );

        let result = Polygon::fix_bad_degrees(&mut graph);
        assert_eq!(result, Ok(()));
        assert!(algo::is_isomorphic_matching(
            &graph,
            &expected,
            |x, y| x == y,
            |x, y| x == y
        ));
    }

    #[test]
    fn test_fix_bad_degrees_split_quad() {
        // multiple vertices need to be split
        let mut graph = create_graph(
            vec![
                vec2(0.0, 1.0),
                vec2(1.0, 0.0),
                vec2(2.0, 1.0),
                vec2(1.0, 2.0),
            ],
            vec![(0, 1), (1, 2), (2, 3), (3, 0), (1, 3), (3, 1)],
        );
        let expected = create_graph(
            vec![
                vec2(0.0, 1.0),
                vec2(1.0, 0.0),
                vec2(2.0, 1.0),
                vec2(1.0, 2.0),
                vec2(1.0, 0.0),
                vec2(1.0, 2.0),
            ],
            vec![(0, 1), (1, 3), (3, 0), (2, 5), (5, 4), (4, 2)],
        );

        let result = Polygon::fix_bad_degrees(&mut graph);
        assert_eq!(result, Ok(()));
        assert!(algo::is_isomorphic_matching(
            &graph,
            &expected,
            |x, y| x == y,
            |x, y| x == y
        ));
    }

    #[test]
    fn test_monotone_decomposition_triangle() {
        let mut polygon = Polygon::from_boundary(create_graph(
            vec![vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(1.0, 1.0)],
            vec![(0, 1), (1, 2), (2, 0)],
        ))
        .unwrap();
        let expected = polygon.boundary.clone();

        polygon.monotone_decompose();
        assert!(algo::is_isomorphic_matching(
            &polygon.boundary,
            &expected,
            |x, y| x == y,
            |x, y| x == y
        ));
    }

    #[test]
    fn test_monotone_decomposition_quad() {
        let mut polygon = Polygon::from_boundary(create_graph(
            vec![
                vec2(0.0, 1.0),
                vec2(1.0, 0.0),
                vec2(2.0, 1.0),
                vec2(1.0, 2.0),
            ],
            vec![(0, 1), (1, 2), (2, 3), (3, 0)],
        ))
        .unwrap();
        let expected = create_graph(
            vec![
                vec2(0.0, 1.0),
                vec2(1.0, 0.0),
                vec2(2.0, 1.0),
                vec2(1.0, 2.0),
            ],
            vec![(0, 1), (1, 2), (2, 3), (3, 0), (1, 3), (3, 1)],
        );
        let expected_cycles = create_decomposition(vec![vec![0, 1, 2], vec![3, 2, 1]]);

        let cycles = testable_decomposition(polygon.monotone_decompose());
        assert!(algo::is_isomorphic_matching(
            &polygon.boundary,
            &expected,
            |x, y| x == y,
            |x, y| x == y
        ));
        assert_eq!(cycles, expected_cycles);
    }

    #[test]
    fn test_monotone_decomposition_regression_quad() {
        // Each step (fix bad vertices, monotone decomposition) seems fine in isolation
        // but the combination glitches?
        // Turns out, order of vertical edges matters
        let mut polygon = Polygon::from_boundary(create_graph(
            vec![
                vec2(0.0, 1.0),
                vec2(1.0, 0.0),
                vec2(2.0, 1.0),
                vec2(1.0, 2.0),
            ],
            vec![(0, 1), (1, 2), (2, 3), (3, 0), (1, 3), (3, 1)],
        ))
        .unwrap();
        let expected = create_graph(
            vec![
                vec2(0.0, 1.0),
                vec2(1.0, 0.0),
                vec2(2.0, 1.0),
                vec2(1.0, 2.0),
                vec2(1.0, 0.0),
                vec2(1.0, 2.0),
            ],
            vec![(0, 1), (1, 3), (3, 0), (2, 5), (5, 4), (4, 2)],
        );

        polygon.monotone_decompose();
        assert!(algo::is_isomorphic_matching(
            &polygon.boundary,
            &expected,
            |x, y| x == y,
            |x, y| x == y
        ));
    }

    #[test]
    fn test_monotone_decomposition_split_quad() {
        // vertices in the same position (but belong to different triangles)
        let mut polygon = Polygon::from_boundary(create_graph(
            vec![
                vec2(0.0, 1.0),
                vec2(1.0, 0.0),
                vec2(2.0, 1.0),
                vec2(1.0, 2.0),
                vec2(1.0, 0.0),
                vec2(1.0, 2.0),
            ],
            vec![(0, 1), (1, 3), (3, 0), (2, 5), (5, 4), (4, 2)],
        ))
        .unwrap();
        let expected = polygon.boundary.clone();

        polygon.monotone_decompose();
        assert!(algo::is_isomorphic_matching(
            &polygon.boundary,
            &expected,
            |x, y| x == y,
            |x, y| x == y
        ));
    }

    #[test]
    fn test_monotone_decomposition_dart_left() {
        // contains a split vertex
        let mut polygon = Polygon::from_boundary(create_graph(
            vec![
                vec2(0.0, 1.0),
                vec2(2.0, 0.0),
                vec2(1.0, 1.0),
                vec2(2.0, 2.0),
            ],
            vec![(0, 1), (1, 2), (2, 3), (3, 0)],
        ))
        .unwrap();
        let expected = create_graph(
            vec![
                vec2(0.0, 1.0),
                vec2(2.0, 0.0),
                vec2(1.0, 1.0),
                vec2(2.0, 2.0),
            ],
            vec![(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (2, 0)],
        );

        polygon.monotone_decompose();
        assert!(algo::is_isomorphic_matching(
            &polygon.boundary,
            &expected,
            |x, y| x == y,
            |x, y| x == y
        ));
    }

    #[test]
    fn test_monotone_decomposition_dart_right() {
        // contains a merge vertex
        let mut polygon = Polygon::from_boundary(create_graph(
            vec![
                vec2(2.0, 1.0),
                vec2(0.0, 2.0),
                vec2(1.0, 1.0),
                vec2(0.0, 0.0),
            ],
            vec![(0, 1), (1, 2), (2, 3), (3, 0)],
        ))
        .unwrap();
        let expected = create_graph(
            vec![
                vec2(2.0, 1.0),
                vec2(0.0, 2.0),
                vec2(1.0, 1.0),
                vec2(0.0, 0.0),
            ],
            vec![(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (2, 0)],
        );

        polygon.monotone_decompose();
        assert!(algo::is_isomorphic_matching(
            &polygon.boundary,
            &expected,
            |x, y| x == y,
            |x, y| x == y
        ));
    }

    #[test]
    fn test_monotone_decomposition_parallelogram_ring() {
        // contains a hole
        let mut polygon = Polygon::from_boundary(create_graph(
            vec![
                vec2(0.0, 0.0),
                vec2(6.0, 0.0),
                vec2(7.0, 3.0),
                vec2(1.0, 3.0),
                vec2(2.0, 1.0),
                vec2(3.0, 2.0),
                vec2(5.0, 2.0),
                vec2(4.0, 1.0),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 4),
            ],
        ))
        .unwrap();
        let expected = create_graph(
            vec![
                vec2(0.0, 0.0),
                vec2(6.0, 0.0),
                vec2(7.0, 3.0),
                vec2(1.0, 3.0),
                vec2(2.0, 1.0),
                vec2(3.0, 2.0),
                vec2(5.0, 2.0),
                vec2(4.0, 1.0),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 4),
                (3, 4),
                (4, 3),
                (1, 6),
                (6, 1),
            ],
        );

        polygon.monotone_decompose();
        assert!(algo::is_isomorphic_matching(
            &polygon.boundary,
            &expected,
            |x, y| x == y,
            |x, y| x == y
        ));
    }

    #[test]
    fn test_monotone_decomposition_complex() {
        // contains a merge vertex that needs to be connected to a split vertex.
        // also a counterexample showing that comparing y coordinates of
        // edge endpoints is not enough to determine which edge comes first.
        let mut polygon = Polygon::from_boundary(create_graph(
            vec![
                vec2(3.0, 0.0),
                vec2(7.0, 0.0),
                vec2(9.0, 5.0),
                vec2(6.0, 2.0),
                vec2(7.0, 4.0),
                vec2(4.0, 4.0),
                vec2(2.0, 2.0),
                vec2(5.0, 3.0),
                vec2(4.0, 1.0),
                vec2(0.0, 2.0),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 0),
            ],
        ))
        .unwrap();
        let expected = create_graph(
            vec![
                vec2(3.0, 0.0),
                vec2(7.0, 0.0),
                vec2(9.0, 5.0),
                vec2(6.0, 2.0),
                vec2(7.0, 4.0),
                vec2(4.0, 4.0),
                vec2(2.0, 2.0),
                vec2(5.0, 3.0),
                vec2(4.0, 1.0),
                vec2(0.0, 2.0),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 0),
                (0, 8),
                (8, 0),
                (1, 3),
                (3, 1),
                (3, 7),
                (7, 3),
                (5, 7),
                (7, 5),
            ],
        );

        polygon.monotone_decompose();
        assert!(algo::is_isomorphic_matching(
            &polygon.boundary,
            &expected,
            |x, y| x == y,
            |x, y| x == y
        ));
    }
}
