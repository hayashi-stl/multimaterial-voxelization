use float_ord::FloatOrd;
use fnv::{FnvHashMap, FnvHashSet};
use petgraph::prelude::*;
use std::path::Path;
use tri_mesh::prelude::*;

/// A tetrahedralization of a bunch of points.
/// Uses Bowyer's algorithm
#[derive(Clone, Debug)]
pub struct DelaunayTetrahedralization {
    /// Graph of voronoi diagram, where vertices are
    /// meeting points of voronoi regions, and edges are
    /// connections between meeting points, with the input point
    /// on the opposite side of the edge as the weight.
    /// Each vertex corresponds to a tetrahedron formed by
    /// the 4 forming input points of the vertex.
    /// Vertex 0 is reserved for the vertex at infinity.
    voronoi: StableGraph<(), NodeIndex>,
    /// Graph of edges in the tetrahedralization
    tet_edges: StableUnGraph<Vec3, ()>,
    /// Which vertices each point is adjacent to.
    /// Does not include the vertex at infinity.
    point_vertices: Vec<FnvHashSet<NodeIndex>>,
    points_to_add: Vec<Vec3>,
}

impl DelaunayTetrahedralization {
    fn init(mut points: Vec<Vec3>) -> Self {
        // Initialize big tetrahedron that contains all points inside it
        let min = points.iter().fold(
            Vec3::new(f64::MAX, f64::MAX, f64::MAX),
            |Vec3 {
                 x: x1,
                 y: y1,
                 z: z1,
             },
             Vec3 {
                 x: x2,
                 y: y2,
                 z: z2,
             }| vec3(x1.min(*x2), y1.min(*y2), z1.min(*z2)),
        ) + vec3(-1.0, -1.0, -1.0);
        let max = points.iter().fold(
            Vec3::new(f64::MIN, f64::MIN, f64::MIN),
            |Vec3 {
                 x: x1,
                 y: y1,
                 z: z1,
             },
             Vec3 {
                 x: x2,
                 y: y2,
                 z: z2,
             }| vec3(x1.max(*x2), y1.max(*y2), z1.max(*z2)),
        ) + vec3(1.0, 1.0, 1.0);

        let mut tet_edges = StableUnGraph::default();
        tet_edges.add_node(min);
        tet_edges.add_node(min + Vec3::unit_x() * 3.0 * (max.x - min.x));
        tet_edges.add_node(min + Vec3::unit_y() * 3.0 * (max.y - min.y));
        tet_edges.add_node(min + Vec3::unit_z() * 3.0 * (max.z - min.z));
        for (i, j) in vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)].into_iter() {
            tet_edges.add_edge(NodeIndex::new(i), NodeIndex::new(j), ());
        }

        let point_vertices = vec![NodeIndex::new(1); 4]
            .into_iter()
            .map(|n| std::iter::once(n).collect())
            .collect();

        let mut voronoi = StableGraph::new();
        let inf_vertex = voronoi.add_node(()); // vertex at infinity
        let tet_vertex = voronoi.add_node(()); // outer tetrahedron voronoi vertex
        for i in (0..4).map(NodeIndex::new) {
            voronoi.add_edge(tet_vertex, inf_vertex, i);
        }

        points.reverse();
        Self {
            voronoi,
            tet_edges,
            point_vertices,
            points_to_add: points,
        }
    }

    fn circumcenter(points: [Vec3; 4]) -> Vec3 {
        let [p0, p1, p2, p3] = points;

        let a_inv = Transform::<Point3<f64>>::inverse_transform(
            &Mat3::from_cols(p1 - p0, p2 - p0, p3 - p0).transpose(),
        )
        .unwrap();

        let b = 0.5
            * vec3(
                p1.magnitude2() - p0.magnitude2(),
                p2.magnitude2() - p0.magnitude2(),
                p3.magnitude2() - p0.magnitude2(),
            );

        a_inv * b
    }

    /// Finds the position of a voronoi vertex,
    /// which is the circumcenter of its 4 forming points.
    fn vertex_position(&self, vertex: NodeIndex) -> Vec3 {
        let mut points = self.voronoi.edges(vertex).map(|e| e.weight());
        let p0 = self.tet_edges[*points.next().unwrap()];
        let p1 = self.tet_edges[*points.next().unwrap()];
        let p2 = self.tet_edges[*points.next().unwrap()];
        let p3 = self.tet_edges[*points.next().unwrap()];

        Self::circumcenter([p0, p1, p2, p3])
    }

    /// Finds a voronoi vertex to delete when point is added.
    /// The point is closer to this vertex than any of its forming points.
    fn find_vertex_to_delete(&self, point: Vec3) -> NodeIndex {
        // Find the nearest point first
        let mut nearest = NodeIndex::new(self.tet_edges.node_count() - 1);
        while {
            let maybe_closer = self
                .tet_edges
                .edges_directed(nearest, Outgoing)
                .map(|e| e.target())
                .min_by_key(|p| FloatOrd((point - self.tet_edges[*p]).magnitude2()))
                .unwrap();

            if (self.tet_edges[maybe_closer] - point).magnitude2()
                < (self.tet_edges[nearest] - point).magnitude2()
            {
                nearest = maybe_closer;
                true
            } else {
                false
            }
        } {}

        // Find vertex that's closer to the new point than a forming point.
        // Such a vertex is guaranteed to exist
        *self.point_vertices[nearest.index()]
            .iter()
            .find(|v| {
                let pos = self.vertex_position(**v);
                (point - pos).magnitude2() < (self.tet_edges[nearest] - pos).magnitude2()
            })
            .unwrap()
    }

    fn find_vertices_to_delete_helper(
        &self,
        point: Vec3,
        vertex: NodeIndex,
        visited: &mut FnvHashSet<NodeIndex>,
    ) -> Vec<NodeIndex> {
        let mut vertices = vec![];

        for v in self.voronoi.edges(vertex).map(|e| e.target()) {
            if v.index() != 0 && visited.insert(v) {
                let pos = self.vertex_position(v);
                let form_point = self.voronoi.edges(v).next().unwrap().weight();
                if (point - pos).magnitude2() < (self.tet_edges[*form_point] - pos).magnitude2() {
                    vertices.push(v);
                    vertices.extend(self.find_vertices_to_delete_helper(point, v, visited));
                }
            }
        }

        vertices
    }

    fn find_vertices_to_delete(&self, point: Vec3) -> Vec<NodeIndex> {
        let vertex = self.find_vertex_to_delete(point);

        let mut to_delete = vec![vertex];
        to_delete.extend(
            self.find_vertices_to_delete_helper(
                point,
                vertex,
                &mut std::iter::once(vertex).collect(),
            )
            .into_iter(),
        );
        to_delete
    }

    /// Get the points that form some Voronoi edge s→t
    /// This is all the points opposite some edge from s that isn't s→t.
    fn points_forming_vertex<'a>(
        &'a self,
        vertex: NodeIndex,
    ) -> impl Iterator<Item = NodeIndex> + 'a {
        self.voronoi.edges(vertex).map(|e| *e.weight())
    }

    /// Get the points that form some Voronoi edge s→t
    /// This is all the points opposite some edge from s that isn't s→t.
    fn points_forming_edge<'a>(&'a self, edge: EdgeIndex) -> impl Iterator<Item = NodeIndex> + 'a {
        let (s, t) = self.voronoi.edge_endpoints(edge).unwrap();
        let mut iter = self.voronoi.neighbors(s).detach();
        std::iter::from_fn(move || iter.next_edge(&self.voronoi))
            .filter(move |e| *e != edge)
            .map(move |e| self.voronoi[e])
    }

    /// Adds the next point to the tetrahedralization
    fn add_point(&mut self) {
        let point = match self.points_to_add.pop() {
            Some(point) => point,
            None => return,
        };

        // Find all vertices to delete
        let v_delete = self.find_vertices_to_delete(point);

        // Points forming deleted vertices
        let near_points = v_delete
            .iter()
            .flat_map(|v| self.voronoi.edges(*v).map(|e| *e.weight()))
            .collect::<FnvHashSet<_>>();

        // edges from vertex-to-delete to vertex-to-not-delete
        let edges = v_delete
            .iter()
            .flat_map(|v| {
                let mut iter = self.voronoi.neighbors(*v).detach();
                let voronoi = &self.voronoi;
                let v_delete = &v_delete;
                std::iter::from_fn(move || iter.next_edge(voronoi))
                    .filter(move |e| !v_delete.contains(&voronoi.edge_endpoints(*e).unwrap().1))
            })
            .collect::<Vec<_>>();

        let p_index = self.tet_edges.add_node(point);

        // Add new vertices to voronoi diagram. Keep the edges they were added on top of.
        let v_new = edges
            .iter()
            .map(|e| {
                let (s, t) = self.voronoi.edge_endpoints(*e).unwrap();
                let vertex = self.voronoi.add_node(());
                self.voronoi.add_edge(vertex, t, p_index);
                if t.index() != 0 {
                    self.voronoi.add_edge(
                        t,
                        vertex,
                        self.voronoi[self.voronoi.find_edge(t, s).unwrap()],
                    );
                }

                (vertex, *e)
            })
            .collect::<FnvHashSet<_>>();

        // Add edges connecting new vertices in voronoi diagram
        for ((v1, e1), (v2, e2)) in v_new
            .iter()
            .flat_map(|v1| v_new.iter().map(move |v2| (v1, v2)))
        {
            if v1 < v2 {
                let (s1, t1) = self.voronoi.edge_endpoints(*e1).unwrap();
                let (s2, t2) = self.voronoi.edge_endpoints(*e2).unwrap();

                let p1 = self.points_forming_edge(*e1).collect::<FnvHashSet<_>>();
                let p2 = self.points_forming_edge(*e2).collect::<FnvHashSet<_>>();

                if p1.intersection(&p2).count() == 2 {
                    self.voronoi
                        .add_edge(*v1, *v2, *p1.difference(&p2).next().unwrap());
                    self.voronoi
                        .add_edge(*v2, *v1, *p2.difference(&p1).next().unwrap());
                }
            }
        }

        // Add new point adjacencies
        for point in &near_points {
            self.tet_edges.add_edge(*point, p_index, ());
        }

        // Add new adjacenices from points to vertices
        self.point_vertices.resize(
            self.point_vertices.len().max(p_index.index() + 1),
            FnvHashSet::default(),
        );
        for (vertex, _) in v_new {
            for edge in self.voronoi.edges(vertex) {
                self.point_vertices[edge.weight().index()].insert(vertex);
            }
        }

        // Delete old vertices from voronoi diagram
        for vertex in &v_delete {
            self.voronoi.remove_node(*vertex);
        }

        // Delete invalid point-vertex adjacencies
        for point in &near_points {
            self.point_vertices[point.index()].retain(|v| !v_delete.contains(v));
        }

        // Delete invalid adjacencies in points
        let edges = near_points
            .iter()
            .flat_map(|p| {
                let mut iter = self.tet_edges.neighbors(*p).detach();
                let near_points = &near_points;
                let tet_edges = &self.tet_edges;
                std::iter::from_fn(move || iter.next_edge(&tet_edges))
                    .filter(move |e| near_points.contains(&tet_edges.edge_endpoints(*e).unwrap().1))
            })
            .collect::<FnvHashSet<_>>();

        for edge in edges {
            let (s, t) = self.tet_edges.edge_endpoints(edge).unwrap();
            let vs = &self.point_vertices[s.index()];
            let vt = &self.point_vertices[t.index()];

            if vs.is_disjoint(&vt) {
                self.tet_edges.remove_edge(edge);
            }
        }
    }

    /// Constructs a tetrahedralization of the input points.
    pub fn new(points: Vec<Vec3>) -> Self {
        let mut tet = Self::init(points);

        while !tet.points_to_add.is_empty() {
            tet.add_point();
        }

        tet
    }

    /// Obtain the tetrahedrons of the tetrahedralization
    pub fn tetrahedrons<'a>(&'a self) -> impl Iterator<Item = [usize; 4]> + 'a {
        self.voronoi
            .node_indices()
            .filter(move |v| v.index() != 0)
            .map(move |v| {
                let mut iter = self.points_forming_vertex(v);
                let p0 = iter.next().unwrap();
                let p1 = iter.next().unwrap();
                let p2 = iter.next().unwrap();
                let p3 = iter.next().unwrap();
                [p0.index(), p1.index(), p2.index(), p3.index()]
            })
            .filter(|tet| tet.iter().all(|i| *i >= 4))
            // Subtract 4 to ignore outer tetrahedron points
            .map(|[p0, p1, p2, p3]| [p0 - 4, p1 - 4, p2 - 4, p3 - 4])
    }

    pub fn export_debug_obj<P: AsRef<Path>>(&self, path: P) {
        let mut output = String::from("o object\n");

        for i in self.tet_edges.node_indices() {
            let pos = self.tet_edges[i];
            output += &format!("v {} {} {}\n", pos.x, pos.y, pos.z);
        }

        for e in self.tet_edges.edge_indices() {
            let (s, t) = self.tet_edges.edge_endpoints(e).unwrap();
            output += &format!("l {} {}\n", s.index() + 1, t.index() + 1);
        }

        std::fs::write(path, output).expect("Could not debug obj");
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use fnv::FnvHashSet;
    use petgraph::algo;
    use petgraph::data::{Element, FromElements};
    use std::collections::BTreeSet;

    fn create_graph<N, I, E>(
        vertices: Vec<N>,
        edges: Vec<(usize, usize, I)>,
        mut edge_fn: impl FnMut(I) -> E,
    ) -> StableGraph<N, E> {
        StableGraph::from_elements(
            vertices
                .into_iter()
                .map(|vertex| Element::Node { weight: vertex })
                .chain(edges.into_iter().map(|(s, t, e)| Element::Edge {
                    source: s,
                    target: t,
                    weight: edge_fn(e),
                })),
        )
    }

    fn graph_edges<'a, N, E: Clone>(
        graph: &'a StableGraph<N, E>,
    ) -> impl Iterator<Item = (NodeIndex, NodeIndex, E)> + 'a {
        graph.edge_indices().map(move |i| {
            let (s, t) = graph.edge_endpoints(i).unwrap();
            (s, t, graph[i].clone())
        })
    }

    fn point_vertices(vec: Vec<Vec<usize>>) -> Vec<FnvHashSet<NodeIndex>> {
        vec.into_iter()
            .map(|v| v.into_iter().map(NodeIndex::new).collect())
            .collect()
    }

    fn is_point_in_tetrahedron(point: Vec3, tet: [Vec3; 4]) -> bool {
        let mtx = Mat4::from_cols(
            tet[0].extend(1.0),
            tet[1].extend(1.0),
            tet[2].extend(1.0),
            tet[3].extend(1.0),
        );
        let barycentric = mtx.inverse_transform().unwrap() * point.extend(1.0);

        Into::<[f64; 4]>::into(barycentric)
            .iter()
            .all(|x| *x > 0.0 && *x < 1.0)
    }

    fn tetrahedrons(tets: Vec<[usize; 4]>) -> FnvHashSet<BTreeSet<usize>> {
        tets.into_iter()
            .map(|[a, b, c, d]| vec![a, b, c, d].into_iter().collect())
            .collect()
    }

    #[test]
    fn test_init() {
        let points = (0..2)
            .flat_map(|z| {
                (0..2).flat_map(move |y| (0..2).map(move |x| vec3(x as f64, y as f64, z as f64)))
            })
            .collect::<Vec<_>>();
        let exp_voronoi = create_graph(
            vec![(), ()],
            vec![(1, 0, 0), (1, 0, 1), (1, 0, 2), (1, 0, 3)],
            NodeIndex::<u32>::new,
        );
        let exp_point_vertices = point_vertices(vec![vec![1], vec![1], vec![1], vec![1]]);

        let tet = DelaunayTetrahedralization::init(points.clone());
        assert_eq!(tet.voronoi.node_count(), exp_voronoi.node_count());
        assert_eq!(
            graph_edges(&tet.voronoi).collect::<FnvHashSet<_>>(),
            graph_edges(&exp_voronoi).collect::<FnvHashSet<_>>()
        );
        assert_eq!(tet.point_vertices, exp_point_vertices);

        let one_tet = (0..4)
            .map(|i| tet.tet_edges[NodeIndex::new(i)])
            .collect::<Vec<_>>();
        let one_tet = [one_tet[0], one_tet[1], one_tet[2], one_tet[3]];
        for point in points {
            assert!(is_point_in_tetrahedron(point, one_tet));
        }
    }

    #[test]
    fn test_circumcenter() {
        let points = [
            vec3(1.0, 0.0, 0.0),
            vec3(3.0, 1.0, 0.0),
            vec3(1.0, 1.0, 0.0),
            vec3(1.0, 1.0, 1.0),
        ];

        assert_eq!(
            DelaunayTetrahedralization::circumcenter(points),
            vec3(2.0, 0.5, 0.5)
        );
    }

    #[test]
    fn test_tetrahedralize_one() {
        // One measly tetrahedron
        let points = vec![
            vec3(1.0, 0.0, 0.0),
            vec3(3.0, 1.0, 0.0),
            vec3(1.0, 1.0, 0.0),
            vec3(1.0, 1.0, 1.0),
        ];
        let exp_tets = tetrahedrons(vec![[0, 1, 2, 3]]);
        let tets = tetrahedrons(
            DelaunayTetrahedralization::new(points)
                .tetrahedrons()
                .collect(),
        );

        assert_eq!(tets, exp_tets);
    }

    #[test]
    fn test_tetrahedralize_two() {
        // Two joined tetrahedrons
        let points = vec![
            vec3(0.0, 0.0, 0.0),
            vec3(2.0, 0.0, 0.0),
            vec3(1.0, 2.0, 0.0),
            vec3(1.0, 1.0, 2.0),
            vec3(1.0, 1.0, -2.0),
        ];
        let exp_tets = tetrahedrons(vec![[0, 1, 2, 3], [0, 1, 2, 4]]);
        let tets = tetrahedrons(
            DelaunayTetrahedralization::new(points)
                .tetrahedrons()
                .collect(),
        );

        assert_eq!(tets, exp_tets);
    }

    #[test]
    fn test_tetrahedralize_three() {
        // Three cycling tetrahedrons
        let points = vec![
            vec3(0.0, 0.0, 0.0),
            vec3(2.0, 0.0, 0.0),
            vec3(1.0, 2.0, 0.0),
            vec3(1.0, 1.0, 0.5),
            vec3(1.0, 1.0, -0.5),
        ];
        let exp_tets = tetrahedrons(vec![[0, 1, 3, 4], [1, 2, 3, 4], [2, 0, 3, 4]]);
        let tets = tetrahedrons(
            DelaunayTetrahedralization::new(points)
                .tetrahedrons()
                .collect(),
        );

        assert_eq!(tets, exp_tets);
    }

    //#[test]
    //fn export_debug_obj() {
    //    let points = vec![
    //        vec3(0.0, 0.0, 0.0),
    //        vec3(2.0, 0.0, 0.0),
    //        vec3(1.0, 2.0, 0.0),
    //        vec3(1.0, 1.0, 0.5),
    //        vec3(1.0, 1.0, -0.5)
    //    ];
    //    let dt = DelaunayTetrahedralization::new(points);

    //    dt.export_debug_obj("assets/debug/dt_test.obj");
    //}
}
