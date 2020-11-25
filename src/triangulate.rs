//! Implementation of polygon triangulation
//! that uses monotone decomposition and beautification
//! based on area/perimeter ratios

use float_ord::FloatOrd;
use fnv::{FnvHashMap, FnvHashSet};
use petgraph::prelude::*;
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
        println!("graph: {:?}", boundary);

        Ok(Self { boundary })
    }

    /// Decomposes the polygon into x-monotone polygons
    /// (actually monotone mountains, where one side is just a line segment)
    fn monotone_decompose(&mut self) {
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

        for (i, node) in nodes.into_iter().enumerate() {
            // Nodes exist because each node has an incoming edge and an outgoing edge.
            let mut node_a = boundary.edges_in(node).next().unwrap().source();
            let mut node_b = boundary.edges(node).next().unwrap().target();
            let index_a = index_map[&node_a];
            let index_b = index_map[&node_b];

            let pos = boundary[node];

            if index_a > i && index_b > i {
                // Split/start vertex
                println!("Split/start!");
                println!("{}: {:?}", i, node);
                println!("edges: {:?}", edges);
                println!("before swap: a: {:?}, b: {:?}", node_a, node_b);
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
                    println!("after swap: a: {:?}, b: {:?}", node_a, node_b);
                    println!("index: {}", index);
                    println!();
                    // Add node as helper in the case of a start vertex
                    // to handle an edge case dealing with a split vertex
                    let helper = if index % 2 == 0 { Some(node) } else { None };
                    edges.insert(index, (node, node_a, helper));
                    edges.insert(index + 1, (node, node_b, helper));

                    if index % 2 != 0 {
                        // Split vertex
                        // Look at edges above and below and add a diagonal to the one
                        // with the rightmost left helper coordinate.
                        let helper = vec![edges[index - 1].2, edges[index + 2].2]
                            .into_iter()
                            .flatten()
                            .max_by_key(|n| FloatOrd(boundary[*n].x))
                            .unwrap();
                        boundary.add_edge(node, helper, ());
                        boundary.add_edge(helper, node, ());

                        // Update helpers
                        edges[index - 1].2 = Some(node);
                        edges[index + 2].2 = Some(node);
                    }
                }
            } else if index_a < i && index_b < i {
                // Merge/end vertex
                println!("Merge/end!");
                println!("{}: {:?}", i, node);
                println!("edges: {:?}", edges);
                println!("before swap: a: {:?}, b: {:?}", node_a, node_b);
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
                println!("after swap: a: {:?}, b: {:?}", node_a, node_b);
                println!("index: {}", index);
                println!();

                if index % 2 != 0 {
                    // Merge vertex
                    for (_, _, helper) in vec![edges[index], edges[index + 1]] {
                        if let Some(helper) = helper {
                            if helper != node_a && helper != node_b {
                                boundary.add_edge(node, helper, ());
                                boundary.add_edge(helper, node, ());
                            }
                        }
                    }

                    // Update helpers
                    edges[index - 1].2 = Some(node);
                    edges[index + 2].2 = Some(node);
                } else {
                    // End vertex
                    if let Some(helper) = vec![edges[index], edges[index + 1]]
                        .into_iter()
                        .max_by_key(|(s, _, _)| index_map[s])
                        .unwrap()
                        .2
                    {
                        boundary.add_edge(node, helper, ());
                        boundary.add_edge(helper, node, ());
                    }
                }

                edges.remove(index + 1);
                edges.remove(index);
            } else {
                // Vertex crosses sweep line
                println!("Continuation!");
                println!("{}: {:?}", i, node);
                println!("edges: {:?}", edges);
                println!("before swap: a: {:?}, b: {:?}", node_a, node_b);
                if index_a > i {
                    std::mem::swap(&mut node_a, &mut node_b);
                }

                let index = edges
                    .iter()
                    .position(|(s, t, _)| (*s, *t) == (node_a, node))
                    .unwrap();
                println!("after swap: a: {:?}, b: {:?}", node_a, node_b);
                println!("index: {}", index);
                println!();

                if let Some(helper) = edges[index].2 {
                    if helper != node_a {
                        boundary.add_edge(node, helper, ());
                        boundary.add_edge(helper, node, ());
                    }
                }

                edges[index] = (node, node_b, None);
                edges[index ^ 1].2 = Some(node);
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use petgraph::algo;
    use petgraph::data::{Element, FromElements};

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

        polygon.monotone_decompose();
        assert!(algo::is_isomorphic_matching(
            &polygon.boundary,
            &expected,
            |x, y| x == y,
            |x, y| x == y
        ));
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
                vec2(3.0, 0.0), vec2(7.0, 0.0), vec2(9.0, 5.0), vec2(6.0, 2.0), vec2(7.0, 4.0),
                vec2(4.0, 4.0), vec2(2.0, 2.0), vec2(5.0, 3.0), vec2(4.0, 1.0), vec2(0.0, 2.0),
            ],
            vec![
                (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 0),
            ],
        ))
        .unwrap();
        let expected = create_graph(
            vec![
                vec2(3.0, 0.0), vec2(7.0, 0.0), vec2(9.0, 5.0), vec2(6.0, 2.0), vec2(7.0, 4.0),
                vec2(4.0, 4.0), vec2(2.0, 2.0), vec2(5.0, 3.0), vec2(4.0, 1.0), vec2(0.0, 2.0),
            ],
            vec![
                (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 0),
                (0, 8), (8, 0), (1, 3), (3, 1), (3, 7), (7, 3), (5, 7), (7, 5)
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
