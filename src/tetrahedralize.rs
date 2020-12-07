use float_ord::FloatOrd;
use fnv::{FnvHashMap, FnvHashSet};
use petgraph::{unionfind::UnionFind, prelude::*};
use stable_vec::StableVec;
use std::path::Path;
use tri_mesh::prelude::*;
use std::collections::BinaryHeap;

use crate::util::ArrayEx;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TetError {
    /// Voronoi diagram is not 4-regular as it should be
    VoronoiNot4Regular,
    /// Could not find voronoi vertex closer to new point than the point that point is closest to,
    /// even though this should be guaranteed
    NoCloserVoronoiVertex,
    /// Some Voronoi edge was formed by less than 3 points, causing a difference to be empty
    VoronoiEdgeFormedByLessThan3Points,
    /// Attempted to flip a boundary tet that needs flipping to recover a face.
    FlipBoundaryTetNecessary,
    /// Could not recover face because no tet could be found in any edge
    NoTetOnFaceEdge,
    /// The flip algorithm secretly failed and some face was not actually recovered.
    FaceNotRecovered,
}

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
    const EPSILON: f64 = Tetrahedralization::EPSILON;

    fn init(mut points: Vec<Vec3>) -> Self {
        // Initialize big tetrahedron that contains all points inside it
        // Make the tetrahedron REALLY big to hopefully avoid concave tetrahedralization
        // after removing the tetrahedron
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
        ) + vec3(-1.0, -1.0, -1.0) * 1000.0;
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
        ) + vec3(1.0, 1.0, 1.0) * 1000.0;

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

    /// Returns None if the points are coplanar
    fn circumcenter(points: [Vec3; 4]) -> Option<Vec3> {
        let [p0, p1, p2, p3] = points;

        Transform::<Point3<f64>>::inverse_transform(
            &Mat3::from_cols(p1 - p0, p2 - p0, p3 - p0).transpose(),
        ).map(|a_inv| {
            let b = 0.5
                * vec3(
                    p1.magnitude2() - p0.magnitude2(),
                    p2.magnitude2() - p0.magnitude2(),
                    p3.magnitude2() - p0.magnitude2(),
                );

            a_inv * b
        })
    }

    fn line_endpoints(&self, vertex: NodeIndex, visited: &mut FnvHashSet<NodeIndex>) -> Result<Vec<Vec3>, TetError> {
        visited.insert(vertex);

        let mut endpoints = vec![];

        // Tree search for vertices that have a position
        for vertex in self.voronoi.neighbors(vertex) {
            if let Some(pos) = self.vertex_position_helper(vertex)? {
                endpoints.push(pos);
            } else if !visited.contains(&vertex) {
                endpoints.extend(self.line_endpoints(vertex, visited)?);
            }
        }

        Ok(endpoints)
    }

    /// Finds the position of a voronoi vertex,
    /// which is the circumcenter of its 4 forming points,
    /// if it can find the position in the first place.
    fn vertex_position_helper(&self, vertex: NodeIndex) -> Result<Option<Vec3>, TetError> {
        let mut points = self.voronoi.edges(vertex).map(|e| e.weight());
        let p0 = self.tet_edges[*points.next().ok_or(TetError::VoronoiNot4Regular)?];
        let p1 = self.tet_edges[*points.next().ok_or(TetError::VoronoiNot4Regular)?];
        let p2 = self.tet_edges[*points.next().ok_or(TetError::VoronoiNot4Regular)?];
        let p3 = self.tet_edges[*points.next().ok_or(TetError::VoronoiNot4Regular)?];

        Ok(Self::circumcenter([p0, p1, p2, p3]))
    }

    /// Finds the position of a voronoi vertex,
    /// which is the circumcenter of its 4 forming points.
    fn vertex_position(&self, vertex: NodeIndex) -> Result<Vec3, TetError> {
        if let Some(pos) = self.vertex_position_helper(vertex)? {
            Ok(pos)
        } else {
            let points = self.line_endpoints(vertex, &mut FnvHashSet::default())?;
            let len = points.len();
            Ok(points.into_iter().sum::<Vec3>() / len as f64)
        }
    }

    /// Finds a voronoi vertex to delete when point is added.
    /// The point is closer to this vertex than any of its forming points.
    fn find_vertex_to_delete(&self, point: Vec3) -> Result<NodeIndex, TetError> {
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

        let mut error = false;
        // Find vertex that's closer to the new point than a forming point.
        // Such a vertex is guaranteed to exist
        let res = self.point_vertices[nearest.index()]
            .iter()
            .find(|v| {
                if let Ok(pos) = self.vertex_position(**v) {
                    (point - pos).magnitude() <= (self.tet_edges[nearest] - pos).magnitude() + Self::EPSILON
                } else {
                    error = true;
                    false
                }
            })
            .copied();

        if error {
            Err(TetError::VoronoiNot4Regular)?;
        }
        res.ok_or(TetError::NoCloserVoronoiVertex)
    }

    fn find_vertices_to_delete_helper(
        &self,
        point: Vec3,
        vertex: NodeIndex,
        visited: &mut FnvHashSet<NodeIndex>,
    ) -> Result<Vec<NodeIndex>, TetError> {
        let mut vertices = vec![];

        for v in self.voronoi.edges(vertex).map(|e| e.target()) {
            if v.index() != 0 && visited.insert(v) {
                let pos = self.vertex_position(v)?;
                let form_point = self.voronoi.edges(v).next().unwrap().weight();
                if (point - pos).magnitude() <= (self.tet_edges[*form_point] - pos).magnitude() + Self::EPSILON {
                    vertices.push(v);
                    vertices.extend(self.find_vertices_to_delete_helper(point, v, visited)?);
                }
            }
        }

        Ok(vertices)
    }

    fn find_vertices_to_delete(&self, point: Vec3) -> Result<Vec<NodeIndex>, TetError> {
        let vertex = self.find_vertex_to_delete(point)?;

        let mut to_delete = vec![vertex];
        to_delete.extend(
            self.find_vertices_to_delete_helper(
                point,
                vertex,
                &mut std::iter::once(vertex).collect(),
            )?
            .into_iter(),
        );
        Ok(to_delete)
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
    /// Returns (tet edges removed, tet edges added)
    pub fn add_point(&mut self, point: Vec3) -> Result<(Vec<(usize, usize)>, Vec<(usize, usize)>), TetError> {
        // Find all vertices to delete
        //for v in self.voronoi.node_indices().skip(1) {
        //    println!("Vertex {}: {:?}", v.index(), self.voronoi.edges(v)
        //        .map(|e| (e.source().index(), e.target().index(), e.weight()))
        //        .collect::<Vec<_>>());
        //}
        //println!();
        let v_delete = self.find_vertices_to_delete(point)?;

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
                        .add_edge(*v1, *v2, *p1.difference(&p2).next().ok_or(TetError::VoronoiEdgeFormedByLessThan3Points)?);
                    self.voronoi
                        .add_edge(*v2, *v1, *p2.difference(&p1).next().ok_or(TetError::VoronoiEdgeFormedByLessThan3Points)?);
                }
            }
        }

        // Add new point adjacencies
        let mut edges_added = vec![];
        for point in &near_points {
            self.tet_edges.add_edge(*point, p_index, ());

            if point.index() >= 4 && p_index.index() >= 4 {
                edges_added.push((
                    if *point < p_index {
                        point.index()
                    } else {
                        p_index.index()
                    } - 4,
                    if *point < p_index {
                        p_index.index()
                    } else {
                        point.index()
                    } - 4,
                ));
            }
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

        let mut edges_removed = vec![];
        for edge in edges {
            let (s, t) = self.tet_edges.edge_endpoints(edge).unwrap();
            let vs = &self.point_vertices[s.index()];
            let vt = &self.point_vertices[t.index()];

            if vs.is_disjoint(&vt) {
                self.tet_edges.remove_edge(edge);

                if s.index() >= 4 && t.index() >= 4 {
                    edges_removed.push((
                        if s < t { s.index() } else { t.index() } - 4,
                        if s < t { t.index() } else { s.index() } - 4,
                    ))
                }
            }
        }

        Ok((edges_removed, edges_added))
    }

    /// Constructs a tetrahedralization of the input points.
    pub fn new(points: Vec<Vec3>) -> Result<Self, TetError> {
        let mut tet = Self::init(points);

        while let Some(point) = tet.points_to_add.pop() {
            tet.add_point(point)?;
        }

        Ok(tet)
    }

    /// Obtain the tetrahedrons of the tetrahedralization
    pub fn tetrahedrons(&self) -> Result<(Vec<Vec3>, Vec<[usize; 4]>), TetError> {
        let positions = self
            .tet_edges
            .node_indices()
            .skip(4)
            .map(|p| self.tet_edges[p])
            .collect();

        let mut error = false;
        let tets = self
            .voronoi
            .node_indices()
            .filter(move |v| v.index() != 0)
            .map(move |v| {
                let mut iter = self.points_forming_vertex(v);
                let p0 = iter.next()?;
                let p1 = iter.next()?;
                let p2 = iter.next()?;
                let p3 = iter.next()?;
                Some([p0.index(), p1.index(), p2.index(), p3.index()])
            })
            .inspect(|opt| if opt.is_none() { error = true; })
            .flatten()
            .filter(|tet| tet.iter().all(|i| *i >= 4))
            // Subtract 4 to ignore outer tetrahedron points
            .map(|[p0, p1, p2, p3]| [p0 - 4, p1 - 4, p2 - 4, p3 - 4])
            .collect();

        if error {
            Err(TetError::VoronoiNot4Regular)?;
        }

        Ok((positions, tets))
    }

    /// Obtain the edges of the tetrahedrons
    pub fn tetrahedron_edges<'a>(&'a self) -> impl Iterator<Item = (usize, usize)> + 'a {
        self.tet_edges
            .edge_indices()
            .map(move |e| {
                let (s, t) = self.tet_edges.edge_endpoints(e).unwrap();
                (s.index(), t.index())
            })
            .filter(|(s, t)| *s >= 4 && *t >= 4)
            .map(|(s, t)| (s - 4, t - 4))
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

    pub fn export_voronoi_debug_obj<P: AsRef<Path>>(&self, path: P) {
        let mut output = String::from("o object\n");

        let graph: Graph<_, _> = self.voronoi.clone().into();

        for i in self.voronoi.node_indices().skip(1) {
            let pos = self.vertex_position(i).unwrap();
            output += &format!("v {} {} {}\n", pos.x, pos.y, pos.z);
        }
        for i in self.tet_edges.node_indices() {
            let pos = self.tet_edges[i];
            output += &format!("v {} {} {}\n", pos.x, pos.y, pos.z);
        }

        for e in graph.edge_indices() {
            let (s, t) = graph.edge_endpoints(e).unwrap();
            if t.index() != 0 {
                output += &format!("l {} {}\n", s.index(), t.index());
            }
        }

        std::fs::write(path, output).expect("Could not debug obj");
    }
}

/// A tetrahedralization that isn't necessarily Delaunay.
#[derive(Clone, Debug, PartialEq)]
pub struct Tetrahedralization {
    /// Vertex positions and adjacent tetrahedrons. Tet vertices are sorted.
    vertices: StableVec<(Vec3, FnvHashSet<usize>)>,
    /// Faces and adjacent tetrahedrons, for convenience. Face and tet vertices are sorted.
    faces: FnvHashMap<[usize; 3], FnvHashSet<usize>>,
    tets: StableVec<[usize; 4]>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum EdgeFlex {
    Convex,
    Flat,
    Concave,
}

impl Tetrahedralization {
    const EPSILON: f64 = 1e-5;

    pub fn new(positions: Vec<Vec3>, tetrahedrons: Vec<[usize; 4]>) -> Self {
        let mut vertices = positions
            .into_iter()
            .map(|pos| (pos, FnvHashSet::default()))
            .collect::<StableVec<_>>();

        let mut faces = FnvHashMap::<[usize; 3], FnvHashSet<usize>>::default();

        let mut tets = StableVec::new();

        for mut tet in tetrahedrons.into_iter() {
            // No sliver tets (hack)
            let pos = tet.iter().map(|v| vertices[*v].0).collect::<Vec<_>>();
            if (pos[1] - pos[0]).cross(pos[2] - pos[0]).dot(pos[3] - pos[0]).abs() < Self::EPSILON {
                continue;
            }

            tet.sort();
            let i = tets.push(tet);

            for v in tet.iter() {
                vertices[*v].1.insert(i);
            }

            for j in 0..4 {
                // Keep sorted order
                let face = [
                    tet[0 + (j <= 0) as usize],
                    tet[1 + (j <= 1) as usize],
                    tet[2 + (j <= 2) as usize],
                ];
                faces.entry(face).or_insert(FnvHashSet::default()).insert(i);
            }
        }

        Self {
            vertices,
            faces,
            tets,
        }
    }

    /// Assumes tet is sorted, faces returned are sorted
    fn tet_faces_and_opposite(tet: [usize; 4]) -> impl Iterator<Item = ([usize; 3], usize)> {
        (0..4).map(move |i| {
            ([
                tet[0 + (i <= 0) as usize],
                tet[1 + (i <= 1) as usize],
                tet[2 + (i <= 2) as usize],
            ], tet[i]
            )
        })
    }

    /// Assumes tet is sorted, faces returned are sorted
    fn tet_faces(tet: [usize; 4]) -> impl Iterator<Item = [usize; 3]> {
        (0..4).map(move |i| {
            [
                tet[0 + (i <= 0) as usize],
                tet[1 + (i <= 1) as usize],
                tet[2 + (i <= 2) as usize],
            ]
        })
    }

    /// Assumes face is sorted, edges returned are sorted
    fn face_edges(face: [usize; 3]) -> impl Iterator<Item = [usize; 2]> {
        (0..3).map(move |i| [face[0 + (i <= 0) as usize], face[1 + (i <= 1) as usize]])
    }

    /// Assumes face is sorted
    fn face_tet_indexes_and_tets<'a>(
        &'a self,
        face: [usize; 3],
    ) -> impl Iterator<Item = (usize, [usize; 4])> + 'a {
        self.faces
            .get(&face)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .map(move |i| (i, self.tets[i]))
    }

    /// Assumes edge is sorted
    fn edge_tet_indexes_and_tets<'a>(
        &'a self,
        edge: [usize; 2],
    ) -> impl Iterator<Item = (usize, [usize; 4])> + 'a {
        let v1 = &self.vertices[edge[0]].1;
        let v2 = &self.vertices[edge[1]].1;
        v1.intersection(v2).map(move |i| (*i, self.tets[*i]))
    }

    fn vertex_tet_indexes_and_tets<'a>(
        &'a self,
        vertex: usize,
    ) -> impl Iterator<Item = (usize, [usize; 4])> + 'a {
        self.vertices[vertex]
            .1
            .iter()
            .map(move |i| (*i, self.tets[*i]))
    }

    /// Assumes edge is sorted.
    fn opposite_vertices_of_edge<'a>(&'a self, edge: [usize; 2]) -> impl Iterator<Item = usize> + 'a {
        self.opposite_edges_of_edge(edge)
            .flat_map(|edge| vec![edge[0], edge[1]].into_iter())
            .collect::<FnvHashSet<_>>()
            .into_iter()
    }

    /// Gets index of tet, if the tet exists.
    /// Assumes tet is sorted.
    fn tet_index(&self, tet: [usize; 4]) -> Option<usize> {
        self.face_tet_indexes_and_tets([tet[0], tet[1], tet[2]])
            .find(|(i, test_tet)| *test_tet == tet)
            .map(|(i, _)| i)
    }

    /// Assumes face is sorted.
    fn contains_face(&self, face: [usize; 3]) -> bool {
        self.faces.contains_key(&face)
    }

    /// Vertex opposite a face in a tet.
    /// Assumes face and tet are sorted
    fn opposite_vertex_of_face(tet: [usize; 4], face: [usize; 3]) -> usize {
        for i in 0..3 {
            if tet[i] != face[i] {
                return tet[i];
            }
        }
        tet[3]
    }

    /// Vertices opposite a face.
    /// Assumes face is sorted.
    fn opposite_vertices_of_face<'a>(
        &'a self,
        face: [usize; 3],
    ) -> impl Iterator<Item = usize> + 'a {
        self.face_tet_indexes_and_tets(face)
            .map(move |(_, tet)| Self::opposite_vertex_of_face(tet, face))
    }

    /// Vertex opposite an edge in a face.
    /// Assumes edge and face are sorted
    fn opposite_vertex_of_edge(face: [usize; 3], edge: [usize; 2]) -> usize {
        for i in 0..2 {
            if face[i] != edge[i] {
                return face[i];
            }
        }
        face[2]
    }

    /// Edges opposite an edge.
    /// Assumes edge is sorted.
    fn opposite_edges_of_edge<'a>(
        &'a self,
        edge: [usize; 2],
    ) -> impl Iterator<Item = [usize; 2]> + 'a {
        self.edge_tet_indexes_and_tets(edge)
            .map(move |(_, tet)| Self::opposite_edge_of_edge(tet, edge))
    }

    /// Edge opposite an edge in a tet.
    /// Assumes edge and tet are sorted.
    fn opposite_edge_of_edge(tet: [usize; 4], edge: [usize; 2]) -> [usize; 2] {
        if tet[0] == edge[0] {
            if tet[1] == edge[1] {
                [tet[2], tet[3]]
            } else if tet[2] == edge[1] {
                [tet[1], tet[3]]
            } else {
                [tet[1], tet[2]]
            }
        } else if tet[1] == edge[0] {
            if tet[2] == edge[1] {
                [tet[0], tet[3]]
            } else {
                [tet[0], tet[2]]
            }
        } else {
            [tet[0], tet[1]]
        }
    }

    /// Adds a tet. Assumes the tet is sorted.
    /// Returns the index of the tet.
    fn add_tet(&mut self, tet: [usize; 4]) -> usize {
        let index = self.tets.push(tet);

        for v in tet.iter() {
            self.vertices[*v].1.insert(index);
        }

        for face in Self::tet_faces(tet) {
            self.faces
                .entry(face)
                .or_insert(FnvHashSet::default())
                .insert(index);
        }

        index
    }

    /// Removes a tet by index, assuming it exists.
    fn remove_tet(&mut self, index: usize) {
        let tet = self.tets.remove(index).unwrap();

        for v in tet.iter() {
            self.vertices[*v].1.remove(&index);
        }

        for face in Self::tet_faces(tet) {
            self.faces.get_mut(&face).unwrap().remove(&index);
            if self.faces.get(&face).unwrap().is_empty() {
                self.faces.remove(&face);
            }
        }
    }

    /// Calculate the flex of an edge [e1, e2] in
    /// tetrahedrons [e1, e2, v_e, v_f1] and [e1, e2, v_e, v_f2]
    /// Assumes edge is sorted
    fn edge_flex(&self, edge: [usize; 2], v_e: usize, v_f1: usize, v_f2: usize) -> EdgeFlex {
        let pos_e1 = self.vertices[edge[0]].0;
        let pos_e = self.vertices[edge[1]].0 - pos_e1;
        let pos_v = self.vertices[v_e].0 - pos_e1;
        let pos_f1 = self.vertices[v_f1].0 - pos_e1;
        let pos_f2 = self.vertices[v_f2].0 - pos_e1;

        // Obtain normals of faces adjacent to edge
        let mut normal1 = pos_e.cross(pos_f1).normalize();
        let mut normal2 = pos_f2.cross(pos_e).normalize();

        // Point them in the right direction, away from v_e
        if normal1.dot(pos_v) > 0.0 {
            normal1 *= -1.0;
            normal2 *= -1.0;
        }

        // Normalize normal in case exterior dihedral angle is close to 0° or 360°
        let normal = (normal1 + normal2).normalize();
        let dot = normal.dot(pos_f1);

        if dot.abs() < Self::EPSILON {
            EdgeFlex::Flat
        } else if dot < 0.0 {
            EdgeFlex::Convex
        } else {
            EdgeFlex::Concave
        }
    }

    /// Gets the boundary of a bunch of tets.
    /// Assumes the tet_indexes don't have duplicates.
    fn boundary(&self, tet_indexes: &[usize]) -> impl Iterator<Item = [usize; 3]> {
        let mut boundary = FnvHashSet::default();
        for index in tet_indexes {
            for face in Self::tet_faces(self.tets[*index]) {
                if !boundary.insert(face) {
                    boundary.remove(&face);
                }
            }
        }

        boundary.into_iter()
    }

    /// Gets the boundary of a bunch of tets.
    /// Assumes the tet_indexes don't have duplicates.
    /// Cares about orientation, so the faces are not sorted.
    /// Index 0 does have the smallest vertex index, though
    fn oriented_boundary(&self, tet_indexes: &[usize]) -> impl Iterator<Item = [usize; 3]> {
        let mut boundary = FnvHashSet::default();
        for index in tet_indexes {
            for (mut face, vertex) in Self::tet_faces_and_opposite(self.tets[*index]) {
                let base = self.vertices[face[0]].0;
                let pos1 = self.vertices[face[1]].0 - base;
                let pos2 = self.vertices[face[2]].0 - base;
                let pos3 = self.vertices[vertex].0 - base;

                if pos1.cross(pos2).dot(pos3) > 0.0 {
                    let tmp = face[1];
                    face[1] = face[2];
                    face[2] = tmp;
                }

                if !boundary.remove(&[face[0], face[2], face[1]]) {
                    boundary.insert(face);
                }
            }
        }

        boundary.into_iter()
    }


    pub fn flip(&mut self, mut face: [usize; 3]) -> Vec<usize> {
        let mut region = self.tets.indices().collect();
        self.flip_in_region(face, &mut region)
    }

    /// Flips away a face as part of Shewchuk's algorithm
    /// for inserting constraining faces into a tetrahedralization
    /// via flips.
    /// Returns the tet indexes that got added, or an empty vec if the flip didn't happen
    pub fn flip_in_region(&mut self, mut face: [usize; 3], region: &mut FnvHashSet<usize>) -> Vec<usize> {
        face.sort();

        let mut iter = self.face_tet_indexes_and_tets(face);
        let s = iter.next();
        // Obtain adjacent tetrahedrons
        let ((s, s_tet), (t, t_tet)) = match (s, iter.next()) {
            (Some((s, s_tet)), Some((t, t_tet))) => ((s, s_tet), (t, t_tet)),
            _ => return vec![],
        };
        std::mem::drop(iter);

        // vertices of tets to remove/add
        let mut v_remove = vec![];
        let s_own = Self::opposite_vertex_of_face(s_tet, face);
        let t_own = Self::opposite_vertex_of_face(t_tet, face);
        let mut v_add = vec![s_own, t_own];

        for edge in Self::face_edges(face) {
            let vertex = Self::opposite_vertex_of_edge(face, edge);

            match self.edge_flex(edge, vertex, s_own, t_own) {
                EdgeFlex::Convex => v_remove.push(vertex),
                EdgeFlex::Concave => v_add.push(vertex),
                EdgeFlex::Flat => {} // One less vertex
            }
        }

        let mut new_tets = vec![];

        let v_change = v_remove
            .iter()
            .chain(v_add.iter())
            .copied()
            .collect::<FnvHashSet<_>>();

        // Simplexes to remove/add
        let x_remove = v_add
            .into_iter()
            .map(|v| {
                let mut vs = v_change
                    .difference(&std::iter::once(v).collect())
                    .copied()
                    .collect::<Vec<_>>();
                vs.sort();
                vs
            })
            .collect::<Vec<_>>();
        let x_add = v_remove
            .into_iter()
            .map(|v| {
                let mut vs = v_change
                    .difference(&std::iter::once(v).collect())
                    .copied()
                    .collect::<Vec<_>>();
                vs.sort();
                vs
            })
            .collect::<Vec<_>>();

        match v_change.len() {
            5 => {
                // Tetrahedron replacement (general)
                let x_remove = x_remove
                    .into_iter()
                    .map(|tet| self.tet_index([tet[0], tet[1], tet[2], tet[3]]))
                    .collect::<Vec<_>>();

                if x_remove.iter().any(|i| i.map(|i| !region.contains(&i)).unwrap_or(true)) {
                    // Concave angle. Do not flip.
                    return vec![];
                }
                for i in x_remove.into_iter().flatten() {
                    self.remove_tet(i);
                    region.remove(&i);
                }
                for tet in x_add {
                    new_tets.push(self.add_tet([tet[0], tet[1], tet[2], tet[3]]));
                    region.insert(*new_tets.last().unwrap());
                }
            }

            4 => {
                // Triangle replacement (degenerate)
                let vertices = x_remove
                    .iter()
                    .flat_map(|face| self.opposite_vertices_of_face([face[0], face[1], face[2]]))
                    .collect::<FnvHashSet<_>>();

                let x_remove = x_remove
                    .into_iter()
                    .flat_map(|face| {
                        let tets = &*self;
                        vertices.iter().map(move |v| {
                            let mut tet = [face[0], face[1], face[2], *v];
                            tet.sort();
                            tets.tet_index(tet)
                        })
                    })
                    .collect::<Vec<_>>();

                if x_remove.iter().any(|i| i.map(|i| !region.contains(&i)).unwrap_or(true)) {
                    // Concave angle or 4th-party tet. Do not flip.
                    return vec![];
                }

                for i in x_remove.into_iter().flatten() {
                    self.remove_tet(i);
                    region.remove(&i);
                }

                let x_add = x_add.into_iter().flat_map(|face| {
                    vertices.iter().map(move |v| {
                        let mut tet = [face[0], face[1], face[2], *v];
                        tet.sort();
                        tet
                    })
                });

                for tet in x_add {
                    new_tets.push(self.add_tet(tet));
                    region.insert(*new_tets.last().unwrap());
                }
            }

            3 => {
                // Line replacement (degenerate)
                let edges = x_remove
                    .iter()
                    .flat_map(|edge| self.opposite_edges_of_edge([edge[0], edge[1]]))
                    .collect::<FnvHashSet<_>>();

                let x_remove = x_remove
                    .into_iter()
                    .flat_map(|edge| {
                        let tets = &*self;
                        edges.iter().map(move |e| {
                            let mut tet = [edge[0], edge[1], e[0], e[1]];
                            tet.sort();
                            tets.tet_index(tet)
                        })
                    })
                    .collect::<Vec<_>>();

                if x_remove.iter().any(|i| i.map(|i| !region.contains(&i)).unwrap_or(true)) {
                    // 3rd-party tet. Do not flip.
                    return vec![];
                }

                for i in x_remove.into_iter().flatten() {
                    self.remove_tet(i);
                    region.remove(&i);
                }

                let x_add = x_add.into_iter().flat_map(|edge| {
                    edges.iter().map(move |e| {
                        let mut tet = [edge[0], edge[1], e[0], e[1]];
                        tet.sort();
                        tet
                    })
                });

                for tet in x_add {
                    new_tets.push(self.add_tet(tet));
                    region.insert(*new_tets.last().unwrap());
                }
            }

            _ => panic!("Unexpected number of vertices involved: {}", v_change.len()),
        }

        new_tets
    }

    fn find_first_intersecting_edge(
        &self,
        edges: &FnvHashSet<[usize; 2]>,
        normal: Vec3,
        vertex_set: &FnvHashSet<usize>,
    ) -> (
        Vec<[usize; 2]>,
        FnvHashSet<[usize; 3]>,
        FnvHashSet<usize>,
        Vec3,
        FnvHashSet<[usize; 2]>,
    ) {
        let mut edges_to_search = vec![];
        let mut inner_faces = FnvHashSet::default();
        let mut inner_tets = FnvHashSet::default();
        let mut point = vec3(0.0, 0.0, 0.0);
        let mut face_edges = FnvHashSet::default();

        'edge_loop: for edge in edges {
            let c_edge = edge.sorted();

            for (index, tet) in self.edge_tet_indexes_and_tets(c_edge) {
                let oppose = Self::opposite_edge_of_edge(tet, c_edge);

                point = self.vertices[edge[0]].0;
                let e_pos = self.vertices[edge[1]].0 - point;
                let o0_pos = self.vertices[oppose[0]].0 - point;
                let o1_pos = self.vertices[oppose[1]].0 - point;

                // Check if tet is "inside" face

                // Obtain tet edge normal
                let mut tet_normal =
                    e_pos.cross(o0_pos).normalize() + o1_pos.cross(e_pos).normalize();
                if tet_normal.dot(o1_pos) > 0.0 {
                    tet_normal *= -1.0;
                }

                // Obtain plc face edge normal
                let edge_normal = e_pos.cross(normal);
                if tet_normal.dot(edge_normal) > 0.0 {
                    // Tet is "inside"
                    // Check if tetrahedron at least partially triangulates face
                    for v in &oppose {
                        if vertex_set.contains(v) {
                            face_edges = vec![*edge, [edge[1], *v], [*v, edge[0]]]
                                .into_iter()
                                .collect();
                            break 'edge_loop;
                        }
                    }

                    // Check if tetrahedron overlaps face
                    if (normal.dot(o0_pos) >= 0.0) != (normal.dot(o1_pos) >= 0.0) {
                        edges_to_search.push([oppose[0], oppose[1]].sorted());
                        inner_tets.insert(index);
                        inner_faces.insert([oppose[0], oppose[1], edge[0]].sorted());
                        inner_faces.insert([oppose[0], oppose[1], edge[1]].sorted());
                        face_edges.insert(*edge);
                        break 'edge_loop;
                    }
                }
            }
        }

        (edges_to_search, inner_faces, inner_tets, point, face_edges)
    }

    fn find_all_intersecting_faces(
        &self,
        normal: Vec3,
        edges_to_search: &mut Vec<[usize; 2]>,
        inner_faces: &mut FnvHashSet<[usize; 3]>,
        inner_tets: &mut FnvHashSet<usize>,
        point: Vec3,
        face_edges: &mut FnvHashSet<[usize; 2]>,
        vertex_set: &FnvHashSet<usize>,
    ) {
        // Find the rest of the inner tets.
        // This edge is sorted.
        while let Some(edge) = edges_to_search.pop() {
            // This edge intersects the face. Add every tetrahedron around the edge.
            for (index, tet) in self.edge_tet_indexes_and_tets(edge) {
                if !inner_tets.insert(index) {
                    continue; // already visited
                }

                let oppose = Self::opposite_edge_of_edge(tet, edge);
                inner_faces.insert([edge[0], edge[1], oppose[0]].sorted());
                inner_faces.insert([edge[0], edge[1], oppose[1]].sorted());

                let e0_pos = self.vertices[edge[0]].0 - point;
                let e1_pos = self.vertices[edge[1]].0 - point;
                let o0_pos = self.vertices[oppose[0]].0 - point;
                let o1_pos = self.vertices[oppose[1]].0 - point;

                if vertex_set.contains(&oppose[0]) && vertex_set.contains(&oppose[1]) {
                    // Contains edge of face (or triangulation edge)
                    let mut tet_normal = (o1_pos - o0_pos).cross(e0_pos - o0_pos).normalize()
                        + (e1_pos - o0_pos).cross(o1_pos - o0_pos).normalize();
                    if tet_normal.dot(e1_pos - o0_pos) > 0.0 {
                        tet_normal *= -1.0;
                    }
                    let edge_dir = normal.cross(tet_normal);
                    face_edges.insert(if edge_dir.dot(o1_pos - o0_pos) >= 0.0 {
                        [oppose[0], oppose[1]]
                    } else {
                        [oppose[1], oppose[0]]
                    });
                } else if vertex_set.contains(&oppose[0]) || vertex_set.contains(&oppose[1]) {
                    // Contains vertex of face
                    let i = if vertex_set.contains(&oppose[0]) {
                        1
                    } else {
                        0
                    };
                    let other = if (normal.dot(if i == 0 { o0_pos } else { o1_pos }) >= 0.0)
                        == (normal.dot(e0_pos) >= 0.0)
                    {
                        edge[1]
                    } else {
                        edge[0]
                    };
                    edges_to_search.push([oppose[i], other].sorted());
                } else if (normal.dot(o0_pos) >= 0.0) != (normal.dot(o1_pos) >= 0.0) {
                    // Opposite edge intersects face
                    let other_index = if (normal.dot(o0_pos) >= 0.0) == (normal.dot(e0_pos) >= 0.0)
                    {
                        1
                    } else {
                        0
                    };
                    edges_to_search.push(oppose);
                    edges_to_search.push([oppose[0], edge[other_index]].sorted());
                    edges_to_search.push([oppose[1], edge[1 - other_index]].sorted());
                } else {
                    // Opposite edge is on one side of face
                    let other = if (normal.dot(o0_pos) >= 0.0) == (normal.dot(e0_pos) >= 0.0) {
                        edge[1]
                    } else {
                        edge[0]
                    };
                    edges_to_search.push([oppose[0], other].sorted());
                    edges_to_search.push([oppose[1], other].sorted());
                }
            }
        }
    }

    fn recover_plc_face_piece(
        &mut self,
        edges: &FnvHashSet<[usize; 2]>,
        normal: Vec3,
        iteration: u32,
    ) -> Result<FnvHashSet<[usize; 2]>, TetError> {
        //self.export_debug_obj("assets/debug/test_cube_recover.obj");
        let vertex_set = edges
            .iter()
            .flat_map(|[e1, e2]| vec![*e1, *e2].into_iter())
            .collect::<FnvHashSet<_>>();

        let (mut edges_to_search, mut inner_faces, mut inner_tets, point, mut face_edges) =
            self.find_first_intersecting_edge(edges, normal, &vertex_set);
        if face_edges.len() > 1 {
            // Found a triangle on the face
            return Ok(face_edges);
        }

        self.find_all_intersecting_faces(
            normal,
            &mut edges_to_search,
            &mut inner_faces,
            &mut inner_tets,
            point,
            &mut face_edges,
            &vertex_set,
        );

        //if iteration > 0 {
        //    self.export_tets_debug_obj(format!("assets/debug/tet_test_inner_tets_{}.obj", iteration), &inner_tets);
        //    self.export_faces_debug_obj(format!("assets/debug/tet_test_inner_faces_{}.obj", iteration), &inner_faces);
        //}

        //for pos in vec![vec3(-23.05, 1.35, 0.8), vec3(-23.05, 1.35, 0.3), vec3(-23.2, 0.95, 0.25)] {
        //    let v = self.vertices.iter()
        //        .find(|(_, (p, _))| (pos - p).magnitude() < 0.01)
        //        .unwrap().0;
        //    println!("v: {}, actual: {:?}", v, self.vertices[v].0);
        //}
        //println!();

        let insert_face_fn = |tets: &Tetrahedralization, heap: &mut BinaryHeap<(FloatOrd<f64>, [usize; 3])>, face: [usize; 3],
            curr_time: f64| -> Result<(), TetError> {
            let opposite = tets.opposite_vertices_of_face(face).collect::<Vec<_>>();
            if opposite.len() < 2 {
                Err(TetError::FlipBoundaryTetNecessary)?;
            }
            //println!();
            //println!("Face vertex {}: {:?}", face[0], tets.vertices[face[0]].0);
            //println!("Face vertex {}: {:?}", face[1], tets.vertices[face[1]].0);
            //println!("Face vertex {}: {:?}", face[2], tets.vertices[face[2]].0);
            //println!("Opps vertex {}: {:?}", opposite[0], tets.vertices[opposite[0]].0);
            //println!("Opps vertex {}: {:?}", opposite[1], tets.vertices[opposite[1]].0);
            let mut pos = vec![face[0], face[1], face[2], opposite[0], opposite[1]].into_iter()
                .map(|v| tets.vertices[v].0)
                .collect::<Vec<_>>();
            //// Reference point should be above face so we don't
            //// have to subtract a time-varying height from every point.
            //pos.sort_by_key(|pos| normal.dot(*pos - point) < 0.0);
            //println!("Pos stuff: {:?}, {:?}", pos, normal.dot(pos[0] - point));

            let mtx1 = Mat4::from_cols(
                (pos[0] - pos[4]).extend(pos[0].magnitude2() - pos[4].magnitude2()),
                (pos[1] - pos[4]).extend(pos[1].magnitude2() - pos[4].magnitude2()),
                (pos[2] - pos[4]).extend(pos[2].magnitude2() - pos[4].magnitude2()),
                (pos[3] - pos[4]).extend(pos[3].magnitude2() - pos[4].magnitude2()),
            );

            let mut mtx2 = mtx1;
            for i in 0..4 {
                mtx2[i][3] = (normal.dot(pos[i] - point)).max(0.0) - (normal.dot(pos[4] - point)).max(0.0);
            }
            //println!("Mtx 1");
            //for c in 0..4 {
            //    println!("{:?}", mtx1[c]);
            //}
            //println!("Mtx 2");
            //for c in 0..4 {
            //    println!("{:?}", mtx2[c]);
            //}

            let det2 = mtx2.determinant();
            let time = -mtx1.determinant() / det2;
            //if face == [248, 260, 634] {
            //    println!("Time: {}", time);
            //}
            //let mut dummy = String::new();
            //std::io::stdin().read_line(&mut dummy).unwrap();
            // Should act like a min heap
            //if time > curr_time {
            if time.is_finite() {
                heap.push((FloatOrd(-time), face));
            }

            Ok(())
        };

        let mut curr_time = 0.0;
        let mut face_queue = BinaryHeap::new();
        for face in inner_faces.into_iter() {
            insert_face_fn(self, &mut face_queue, face, curr_time)?;
        }

        while let Some((time, face)) = face_queue.pop() {
            //if iteration == 2 {
            //    println!("Time: {}, Face: {:?}, Pos: {:?}", -time.0, face, [self.vertices[face[0]].0, self.vertices[face[1]].0, self.vertices[face[2]].0]);
            //}
            //if face == [248, 260, 634] {
            //    println!("Checking, point: {:?}, normal: {:?}", point, normal);
            //    self.export_debug_obj("assets/debug/tet_test_before_check_248_260_634.obj");
            //    self.export_tets_debug_obj("assets/debug/tet_test_inner_tets_before_check_248_260_634.obj", &inner_tets);
            //}
            if self.contains_face(face) {
                //if iteration == 2 {
                //    println!("Face still component");
                //}
                //if face == [248, 260, 634] && normal.z > 0.0 {
                //    let opposite = self.opposite_vertices_of_face(face).collect::<Vec<_>>();
                //    println!("face pos: {:?}", [self.vertices[face[0]].0, self.vertices[face[1]].0, self.vertices[face[2]].0]);
                //    println!("opps: {:?}, pos: {:?}", opposite, [self.vertices[opposite[0]].0, self.vertices[opposite[1]].0]);
                //    self.export_debug_obj("assets/debug/tet_test_before_flip_248_260_634.obj");
                //}
                let added_tets = self.flip_in_region(face, &mut inner_tets);

                for face in self.boundary(&added_tets) {
                    //if iteration == 2 {
                    //    println!("Boundary face: {:?}", face);
                    //}
                    //if face == [248, 260, 634] {
                    //    println!("Reached, point: {:?}, normal: {:?}", point, normal);
                    //}
                    // Check if boundary face intersects plc face
                    let dots = face.iter()
                        .filter(|v| !vertex_set.contains(*v)) // these don't count
                        .map(|v| normal.dot(self.vertices[*v].0 - point));
                    let dots2 = dots.clone();

                    if dots.zip(dots2.skip(1)).any(|(d1, d2)| (d1 >= 0.0) != (d2 >= 0.0)) {
                        insert_face_fn(self, &mut face_queue, face, curr_time)?;
                    }
                }
            }
        }

        Ok(face_edges)
    }

    /// Recovers a PLC face enclosed by certain edges with a certain normal.
    /// The edges here are NOT sorted. They wind counterclockwise around the face if the normal points up.
    /// Output is the face that actually got recovered, which may be smaller because
    /// the face may already be split by the tetrahedralization.
    /// Uses Shewchuk's algorithm.
    pub fn recover_plc_face(
        &mut self,
        edges: &[[usize; 2]],
        normal: Vec3,
    ) -> Result<(), TetError> {
        let mut edge_set = edges.iter().copied().collect::<FnvHashSet<_>>();

        //if (normal - vec3(0.0, 0.0, -1.0)).magnitude() < Self::EPSILON &&
        //    (self.vertices[edges[0][0]].0.z + 0.3).abs() < Self::EPSILON {
        //    println!("{}", self.vertices[edges[0][0]].0.y);
        //}
        let mut i = 0;
        //if (normal - vec3(0.0, 0.0, 1.0)).magnitude() < Self::EPSILON &&
        //    self.vertices[edges[0][0]].0.y >= 0.84 && self.vertices[edges[0][0]].0.y <= 1.36 &&
        //    (self.vertices[edges[0][0]].0.z - 0.3).abs() < Self::EPSILON {
        //    println!("Preserving...");
        //    self.export_debug_obj("assets/debug/tet_test_before_flip_x_0.85_0.3.obj");
        //    i = 1;
        //}

        while !edge_set.is_empty() {
            //println!("Edge set: {:?}", edge_set);
            //for edge in &edge_set {
            //    println!("{:?}: {:?}", edge, [self.vertices[edge[0]].0, self.vertices[edge[1]].0]);
            //}
            let edges = self.recover_plc_face_piece(&edge_set, normal, i)?;
            if edges.is_empty() {
                Err(TetError::NoTetOnFaceEdge)?;
            }
            for edge in edges {
                //println!("Edge: {:?}", edge);
                if !edge_set.remove(&edge) {
                    edge_set.insert([edge[1], edge[0]]);
                }
            }
            if i > 0 {
                i += 1;
            }
        }

        Ok(())
    }

    /// Returns the triangles that make up some PLC face enclosed by edges
    pub fn plc_face_triangles(&self, edges: &[[usize; 2]], normal: Vec3) -> Result<Vec<[usize; 3]>, TetError> {
        let mut edge_set = edges.iter().copied().collect::<FnvHashSet<_>>();
        let vertex_set = edges
            .iter()
            .flat_map(|[e1, e2]| vec![*e1, *e2].into_iter())
            .collect::<FnvHashSet<_>>();

        let mut triangles = vec![];

        //println!("Normal: {:?}", normal);
        //for edge in edges {
        //    println!("e {:?} at {:?}", edge, [self.vertices[edge[0]].0, self.vertices[edge[1]].0]);
        //}
        while !edge_set.is_empty() {
            let edge = edge_set.iter().next().copied().unwrap();
            //println!("Edge: {:?}, Position: {:?}", edge, [self.vertices[edge[0]].0, self.vertices[edge[1]].0]);

            // Get triangle inside PLC face
            let vertex = self.opposite_vertices_of_edge(edge.sorted()).find(|vertex| {
                let base = self.vertices[edge[0]].0;
                let p1 = self.vertices[edge[1]].0 - base;
                let p2 = self.vertices[*vertex].0 - base;

                // Triangle must be inside, winding counterclockwise
                vertex_set.contains(vertex) && p1.cross(p2).dot(normal) > 0.0
            }).ok_or(TetError::FaceNotRecovered)?;

            triangles.push([edge[0], edge[1], vertex]);
            for edge in &[edge, [edge[1], vertex], [vertex, edge[0]]] {
                if !edge_set.remove(edge) {
                    edge_set.insert([edge[1], edge[0]]);
                }
            }
        }

        Ok(triangles)
    }

    /// Removes tets that are outside a boundary delimited by faces.
    /// Assumes the boundary is a manifold and covered by tets.
    pub fn remove_tets_outside_boundary(&mut self, boundary: &[[usize; 3]]) {
        let mut tets_to_search = vec![];
        let mut is_inside = StableVec::with_capacity(self.tets.next_push_index());
        
        // Find assignment for a tet to start things off
        let face = boundary[0];
        let (index, tet) = self.face_tet_indexes_and_tets(face.sorted()).next().unwrap();
        let vertex = Self::opposite_vertex_of_face(tet, face.sorted());

        let base = self.vertices[face[0]].0;
        let p1 = self.vertices[face[1]].0 - base;
        let p2 = self.vertices[face[2]].0 - base;
        let p3 = self.vertices[vertex].0 - base;

        is_inside.insert(index, p1.cross(p2).dot(p3) < 0.0);
        tets_to_search.push(index);

        // each face is sorted
        let c_boundary = boundary.iter().copied().map(ArrayEx::sorted).collect::<FnvHashSet<_>>();

        // Expand assignment to all tets
        while let Some(index) = tets_to_search.pop() {
            let tet = self.tets[index];
            let inside = is_inside[index];

            for face in Self::tet_faces(tet) {
                if let Some((adj_index, adj)) = self.face_tet_indexes_and_tets(face)
                    .filter(|(_, other)| tet != *other)
                    .next()
                {
                    if is_inside.insert(adj_index, inside != c_boundary.contains(&face)).is_none() {
                        tets_to_search.push(adj_index);
                    }
                }
            }
        }

        for (index, inside) in is_inside {
            if !inside {
                self.remove_tet(index);
            }
        }
    }

    fn is_boundary_convex(&self, boundary: &[[usize; 3]]) -> bool {
        let mut edges = FnvHashMap::default();

        for face in boundary {
            edges.insert([face[0], face[1]], face[2]);
            edges.insert([face[1], face[2]], face[0]);
            edges.insert([face[2], face[0]], face[1]);
        }

        // Check bend of edge
        edges.iter().all(|(edge, opps)| {
            let base = self.vertices[edge[0]].0;
            let pos1 = self.vertices[edge[1]].0 - base;
            let pos2 = self.vertices[*opps].0 - base;
            let pos3 = self.vertices[edges[&[edge[1], edge[0]]]].0 - base;

            pos1.cross(pos2).dot(pos3) < Self::EPSILON
        })
    }

    pub fn convex_hulls(&self) -> Vec<Vec<Vec3>> {
        let mut hull_find = UnionFind::new(self.tets.next_push_index());

        let expand_fn = |tets: &Tetrahedralization, hull: &mut FnvHashSet<usize>, hull_find: &UnionFind<usize>| {
            let mut visited = hull.iter().copied().collect::<FnvHashSet<_>>();
            let mut to_search = visited.iter().copied().collect::<Vec<_>>();

            while let Some(index) = to_search.pop() {
                for face in Self::tet_faces(tets.tets[index]) {
                    for (other, _) in tets.face_tet_indexes_and_tets(face) {
                        if visited.insert(other) && hull_find.equiv(index, other) {
                            hull.insert(other);
                            to_search.push(other);
                        }
                    }
                }
            }
        };

        // Delete vertices as necessary
        for (vertex, (_, tets)) in &self.vertices {
            let mut hull = tets.iter().copied().collect::<FnvHashSet<_>>();
            expand_fn(self, &mut hull, &hull_find);
            let boundary = self.oriented_boundary(&hull.into_iter().collect::<Vec<_>>())
                .collect::<Vec<_>>();

            // Do NOT delete boundary vertices
            if !boundary.iter().any(|face| face.contains(&vertex)) &&
                self.is_boundary_convex(&boundary)
            {
                for (t1, t2) in tets.iter().zip(tets.iter().skip(1)) {
                    hull_find.union(*t1, *t2);
                }
            }
        }

        // Delete edges as necessary
        for edge in self.faces.keys().flat_map(|face| Self::face_edges(*face)).collect::<FnvHashSet<_>>() {
            let mut hull = self.edge_tet_indexes_and_tets(edge).map(|(index, _)| index).collect::<FnvHashSet<_>>();
            let tets = hull.clone();
            expand_fn(self, &mut hull, &hull_find);
            let boundary = self.oriented_boundary(&hull.into_iter().collect::<Vec<_>>())
                .collect::<Vec<_>>();

            // To contain an edge, the boundary has to contain a face which contains the edge
            if !boundary.iter().any(|face| Self::face_edges(face.sorted()).any(|e| e == edge)) &&
                self.is_boundary_convex(&boundary)
            {
                for (t1, t2) in tets.iter().zip(tets.iter().skip(1)) {
                    hull_find.union(*t1, *t2);
                }
            }
        }

        // Finally, faces.
        for (face, _) in &self.faces {
            let mut hull = self.face_tet_indexes_and_tets(*face).map(|(index, _)| index).collect::<FnvHashSet<_>>();
            let tets = hull.clone();
            expand_fn(self, &mut hull, &hull_find);
            let boundary = self.oriented_boundary(&hull.into_iter().collect::<Vec<_>>())
                .collect::<Vec<_>>();

            // No boundary faces. Each face must have 2 tets adjacent.
            if tets.len() == 2 && self.is_boundary_convex(&boundary) {
                for (t1, t2) in tets.iter().zip(tets.iter().skip(1)) {
                    hull_find.union(*t1, *t2);
                }
            }
        }
        
        let rep_map = hull_find.into_labeling();
        let mut hulls = FnvHashMap::default();

        for index in self.tets.indices() {
            hulls.entry(rep_map[index]).or_insert(vec![]).push(index);
        }

        hulls.into_iter().map(|(_, hull)| {
            self.boundary(&hull)
                .flat_map(|face| vec![face[0], face[1], face[2]].into_iter())
                .map(|v| self.vertices[v].0)
                .collect()
        }).collect()
    }

    /// Used only for testing.
    /// Sorts *array* of tets lexicographically. (Not the same as sorting a tet)
    fn canonicalize(&mut self) {
        let mut sorted = self
            .tets
            .iter()
            .map(|(i, tet)| (i, *tet))
            .collect::<Vec<_>>();

        sorted.sort_by_key(|(_, tet)| *tet);
        let inv = sorted
            .iter()
            .enumerate()
            .map(|(i, (j, _))| (*j, i))
            .collect::<FnvHashMap<_, _>>();

        for (_, (_, tets)) in self.vertices.iter_mut() {
            *tets = tets.iter().map(|i| inv[i]).collect();
        }

        for (_, tets) in self.faces.iter_mut() {
            *tets = tets.iter().map(|i| inv[i]).collect();
        }

        self.tets = sorted.into_iter().map(|(_, tet)| tet).collect();
    }

    pub fn export_debug_obj<P: AsRef<Path>>(&self, path: P) {
        let mut output = String::from("o object\n");

        //for (_, (pos, _)) in &self.vertices {
        //    output += &format!("v {} {} {}\n", pos.x, pos.y, pos.z);
        //}
        for (_, tet) in self.tets.iter() {
            for vertex in tet {
                let pos = self.vertices[*vertex].0;
                output += &format!("v {} {} {}\n", pos.x, pos.y, pos.z);
            }
        }

        for (i, (_, tet)) in self.tets.iter().enumerate() {
            // Winding
            let base = self.vertices[tet[0]].0;
            let p1 = self.vertices[tet[1]].0 - base;
            let p2 = self.vertices[tet[2]].0 - base;
            let p3 = self.vertices[tet[3]].0 - base;

            let mut x = [0, 1, 2, 3];
            if p1.cross(p2).dot(p3) > 0.0 {
                x = [0, 1, 3, 2];
            }

            output += &format!("f {} {} {}\n", 4 * i + x[0] + 1, 4 * i + x[1] + 1, 4 * i + x[2] + 1);
            output += &format!("f {} {} {}\n", 4 * i + x[1] + 1, 4 * i + x[0] + 1, 4 * i + x[3] + 1);
            output += &format!("f {} {} {}\n", 4 * i + x[3] + 1, 4 * i + x[0] + 1, 4 * i + x[2] + 1);
            output += &format!("f {} {} {}\n", 4 * i + x[2] + 1, 4 * i + x[1] + 1, 4 * i + x[3] + 1);
        }

        std::fs::write(path, output).expect("Could not debug obj");
    }

    pub fn export_faces_debug_obj<P: AsRef<Path>>(&self, path: P, faces: &FnvHashSet<[usize; 3]>) {
        let mut output = String::from("o object\n");

        for (_, (pos, _)) in &self.vertices {
            output += &format!("v {} {} {}\n", pos.x, pos.y, pos.z);
        }

        for face in faces {
            output += &format!("f {} {} {}\n", face[0] + 1, face[1] + 1, face[2] + 1);
        }

        std::fs::write(path, output).expect("Could not debug obj");
    }

    pub fn export_tets_debug_obj<P: AsRef<Path>>(&self, path: P, tets: &FnvHashSet<usize>) {
        let mut output = String::from("o object\n");

        //for (_, (pos, _)) in &self.vertices {
        //    output += &format!("v {} {} {}\n", pos.x, pos.y, pos.z);
        //}
        for (_, tet) in self.tets.iter() {
            for vertex in tet {
                let pos = self.vertices[*vertex].0;
                output += &format!("v {} {} {}\n", pos.x, pos.y, pos.z);
            }
        }

        for (i, (index, tet)) in self.tets.iter().enumerate() {
            if tets.contains(&index) {
                // Winding
                let base = self.vertices[tet[0]].0;
                let p1 = self.vertices[tet[1]].0 - base;
                let p2 = self.vertices[tet[2]].0 - base;
                let p3 = self.vertices[tet[3]].0 - base;

                let mut x = [0, 1, 2, 3];
                if p1.cross(p2).dot(p3) > 0.0 {
                    x = [0, 1, 3, 2];
                }

                output += &format!("f {} {} {}\n", 4 * i + x[0] + 1, 4 * i + x[1] + 1, 4 * i + x[2] + 1);
                output += &format!("f {} {} {}\n", 4 * i + x[1] + 1, 4 * i + x[0] + 1, 4 * i + x[3] + 1);
                output += &format!("f {} {} {}\n", 4 * i + x[3] + 1, 4 * i + x[0] + 1, 4 * i + x[2] + 1);
                output += &format!("f {} {} {}\n", 4 * i + x[2] + 1, 4 * i + x[1] + 1, 4 * i + x[3] + 1);
            }
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
    use std::hash::Hash;

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
            Some(vec3(2.0, 0.5, 0.5))
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
        let tets = tetrahedrons(DelaunayTetrahedralization::new(points).unwrap().tetrahedrons().unwrap().1);

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
        let tets = tetrahedrons(DelaunayTetrahedralization::new(points).unwrap().tetrahedrons().unwrap().1);

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
        let tets = tetrahedrons(DelaunayTetrahedralization::new(points).unwrap().tetrahedrons().unwrap().1);

        assert_eq!(tets, exp_tets);
    }

    //#[test]
    //fn export_debug_obj() {
    //    let points = vec![
    //        vec3(0.0, 0.0, 0.0),
    //        vec3(2.0, 0.0, 0.0),
    //        vec3(2.0, 0.0, 1.0),
    //        vec3(3.0, 0.0, 1.0),
    //        vec3(3.0, 0.0, 3.0),
    //        vec3(0.0, 0.0, 3.0),
    //        vec3(0.0, 3.0, 0.0),
    //        vec3(2.0, 3.0, 0.0),
    //        vec3(2.0, 3.0, 1.0),
    //        vec3(3.0, 3.0, 1.0),
    //        vec3(3.0, 3.0, 3.0),
    //        vec3(0.0, 3.0, 3.0),
    //    ];
    //    let dt = DelaunayTetrahedralization::new(points);

    //    dt.export_debug_obj("assets/debug/dt_test.obj");
    //}

    //#[test]
    //fn export_debug_obj() {
    //    let points = vec![
    //        vec3(1.0, 0.0, 0.0),
    //        vec3(-0.5, 0.0, -0.75f64.sqrt()),
    //        vec3(-0.5, 0.0, 0.75f64.sqrt()),
    //        vec3(0.75f64.sqrt(), 1.5, -0.5),
    //        vec3(-0.75f64.sqrt(), 1.5, -0.5),
    //        vec3(0.0, 1.5, 1.0),
    //    ];
    //    let dt = DelaunayTetrahedralization::new(points);

    //    dt.export_debug_obj("assets/debug/dt_test2.obj");
    //}

    fn vertex_list<N: Copy, E>(graph: &Graph<N, E>) -> Vec<N> {
        graph.node_indices().map(|n| graph[n]).collect()
    }

    fn edge_set<N, E: Copy + Eq + Hash>(
        graph: &Graph<N, E>,
    ) -> FnvHashSet<(NodeIndex, NodeIndex, E)> {
        graph
            .edge_references()
            .map(|e| (e.source(), e.target(), *e.weight()))
            .collect()
    }

    fn create_tets(
        vertices: Vec<(Vec3, Vec<usize>)>,
        faces: Vec<([usize; 3], Vec<usize>)>,
        tets: Vec<[usize; 4]>,
    ) -> Tetrahedralization {
        let vertices = vertices
            .into_iter()
            .map(|(pos, tets)| (pos, tets.into_iter().collect()))
            .collect();

        let faces = faces
            .into_iter()
            .map(|(face, tets)| (face, tets.into_iter().collect()))
            .collect();

        let tets = tets.into_iter().collect();

        Tetrahedralization {
            vertices,
            faces,
            tets,
        }
    }

    #[test]
    fn test_tetrahedralization_init() {
        let positions = vec![
            vec3(0.0, 0.0, 0.0),
            vec3(1.0, 0.0, 0.0),
            vec3(0.0, 1.0, 0.0),
            vec3(0.0, 0.0, 1.0),
            vec3(1.0, 1.0, 1.0),
        ];
        let tets = vec![[0, 1, 2, 3], [4, 3, 2, 1]];
        let tet = Tetrahedralization::new(positions, tets);

        let exp = create_tets(
            vec![
                (vec3(0.0, 0.0, 0.0), vec![0]),
                (vec3(1.0, 0.0, 0.0), vec![0, 1]),
                (vec3(0.0, 1.0, 0.0), vec![0, 1]),
                (vec3(0.0, 0.0, 1.0), vec![0, 1]),
                (vec3(1.0, 1.0, 1.0), vec![1]),
            ],
            vec![
                ([0, 1, 2], vec![0]),
                ([0, 1, 3], vec![0]),
                ([0, 2, 3], vec![0]),
                ([1, 2, 3], vec![0, 1]),
                ([1, 2, 4], vec![1]),
                ([1, 3, 4], vec![1]),
                ([2, 3, 4], vec![1]),
            ],
            vec![[0, 1, 2, 3], [1, 2, 3, 4]],
        );

        assert_eq!(tet, exp);
    }

    #[test]
    fn test_tetrahedralization_tet_faces() {
        assert_eq!(
            Tetrahedralization::tet_faces([4, 5, 6, 7]).collect::<FnvHashSet<_>>(),
            vec![[4, 5, 6], [4, 5, 7], [4, 6, 7], [5, 6, 7]]
                .into_iter()
                .collect::<FnvHashSet<_>>()
        );
    }

    #[test]
    fn test_tetrahedralization_face_edges() {
        assert_eq!(
            Tetrahedralization::face_edges([4, 5, 6]).collect::<FnvHashSet<_>>(),
            vec![[4, 5], [4, 6], [5, 6]]
                .into_iter()
                .collect::<FnvHashSet<_>>()
        );
    }

    #[test]
    #[rustfmt::skip]
    fn test_tetrahedralization_opposite_vertex_of_face() {
        assert_eq!(Tetrahedralization::opposite_vertex_of_face([4, 5, 6, 7], [4, 5, 6]), 7);
        assert_eq!(Tetrahedralization::opposite_vertex_of_face([4, 5, 6, 7], [4, 5, 7]), 6);
        assert_eq!(Tetrahedralization::opposite_vertex_of_face([4, 5, 6, 7], [4, 6, 7]), 5);
        assert_eq!(Tetrahedralization::opposite_vertex_of_face([4, 5, 6, 7], [5, 6, 7]), 4);
    }

    #[test]
    #[rustfmt::skip]
    fn test_tetrahedralization_opposite_vertex_of_edge() {
        assert_eq!(Tetrahedralization::opposite_vertex_of_edge([4, 5, 6], [4, 5]), 6);
        assert_eq!(Tetrahedralization::opposite_vertex_of_edge([4, 5, 6], [4, 6]), 5);
        assert_eq!(Tetrahedralization::opposite_vertex_of_edge([4, 5, 6], [5, 6]), 4);
    }

    #[test]
    #[rustfmt::skip]
    fn test_tetrahedralization_opposite_edge_of_edge() {
        assert_eq!(Tetrahedralization::opposite_edge_of_edge([4, 5, 6, 7], [4, 5]), [6, 7]);
        assert_eq!(Tetrahedralization::opposite_edge_of_edge([4, 5, 6, 7], [4, 6]), [5, 7]);
        assert_eq!(Tetrahedralization::opposite_edge_of_edge([4, 5, 6, 7], [4, 7]), [5, 6]);
        assert_eq!(Tetrahedralization::opposite_edge_of_edge([4, 5, 6, 7], [5, 6]), [4, 7]);
        assert_eq!(Tetrahedralization::opposite_edge_of_edge([4, 5, 6, 7], [5, 7]), [4, 6]);
        assert_eq!(Tetrahedralization::opposite_edge_of_edge([4, 5, 6, 7], [6, 7]), [4, 5]);
    }

    #[test]
    fn test_tetrahedralization_edge_flex_convex() {
        let positions = vec![
            vec3(0.0, 1.0, 0.0),
            vec3(-1.0, -1.0, 0.0),
            vec3(1.0, -1.0, 0.0),
            vec3(0.0, 0.0, -1.0),
            vec3(0.0, 0.0, 1.0),
        ];
        let tets = vec![[0, 1, 2, 3], [0, 1, 2, 4]];
        let tet = Tetrahedralization::new(positions, tets);

        assert_eq!(tet.edge_flex([0, 1], 2, 3, 4), EdgeFlex::Convex);
    }

    #[test]
    fn test_tetrahedralization_edge_flex_flat() {
        let positions = vec![
            vec3(0.0, 1.0, 0.0),
            vec3(-1.0, -1.0, 0.0),
            vec3(1.0, -1.0, 0.0),
            vec3(0.0, 0.0, -1.0),
            vec3(-1.0, 0.0, 1.0),
        ];
        let tets = vec![[0, 1, 2, 3], [0, 1, 2, 4]];
        let tet = Tetrahedralization::new(positions, tets);

        assert_eq!(tet.edge_flex([0, 1], 2, 3, 4), EdgeFlex::Flat);
    }

    #[test]
    fn test_tetrahedralization_edge_flex_concave() {
        let positions = vec![
            vec3(0.0, 1.0, 0.0),
            vec3(-1.0, -1.0, 0.0),
            vec3(1.0, -1.0, 0.0),
            vec3(0.0, 0.0, -1.0),
            vec3(-2.0, 0.0, 1.0),
        ];
        let tets = vec![[0, 1, 2, 3], [0, 1, 2, 4]];
        let tet = Tetrahedralization::new(positions, tets);

        assert_eq!(tet.edge_flex([0, 1], 2, 3, 4), EdgeFlex::Concave);
    }

    #[test]
    fn test_tetrahedralization_remove_face() {
        let positions = vec![
            vec3(0.0, 0.0, 0.0),
            vec3(1.0, 0.0, 0.0),
            vec3(0.0, 1.0, 0.0),
            vec3(0.0, 0.0, 1.0),
            vec3(1.0, 1.0, 1.0),
        ];
        let tets = vec![[0, 1, 2, 3], [4, 3, 2, 1]];
        let mut tet = Tetrahedralization::new(positions, tets);
        tet.remove_tet(0);
        tet.canonicalize();

        let exp = create_tets(
            vec![
                (vec3(0.0, 0.0, 0.0), vec![]),
                (vec3(1.0, 0.0, 0.0), vec![0]),
                (vec3(0.0, 1.0, 0.0), vec![0]),
                (vec3(0.0, 0.0, 1.0), vec![0]),
                (vec3(1.0, 1.0, 1.0), vec![0]),
            ],
            vec![
                ([1, 2, 3], vec![0]),
                ([1, 2, 4], vec![0]),
                ([1, 3, 4], vec![0]),
                ([2, 3, 4], vec![0]),
            ],
            vec![[1, 2, 3, 4]],
        );

        assert_eq!(tet, exp);
    }

    #[test]
    fn test_tetrahedralization_add_face() {
        let positions = vec![
            vec3(0.0, 0.0, 0.0),
            vec3(1.0, 0.0, 0.0),
            vec3(0.0, 1.0, 0.0),
            vec3(0.0, 0.0, 1.0),
            vec3(1.0, 1.0, 1.0),
        ];
        let tets = vec![[0, 1, 2, 4], [0, 2, 3, 4]];
        let mut tet = Tetrahedralization::new(positions, tets);
        tet.add_tet([0, 1, 3, 4]);
        tet.canonicalize();

        let exp = create_tets(
            vec![
                (vec3(0.0, 0.0, 0.0), vec![0, 1, 2]),
                (vec3(1.0, 0.0, 0.0), vec![0, 1]),
                (vec3(0.0, 1.0, 0.0), vec![0, 2]),
                (vec3(0.0, 0.0, 1.0), vec![1, 2]),
                (vec3(1.0, 1.0, 1.0), vec![0, 1, 2]),
            ],
            vec![
                ([0, 1, 2], vec![0]),
                ([0, 1, 3], vec![1]),
                ([0, 1, 4], vec![0, 1]),
                ([0, 2, 3], vec![2]),
                ([0, 2, 4], vec![0, 2]),
                ([0, 3, 4], vec![1, 2]),
                ([1, 2, 4], vec![0]),
                ([1, 3, 4], vec![1]),
                ([2, 3, 4], vec![2]),
            ],
            vec![[0, 1, 2, 4], [0, 1, 3, 4], [0, 2, 3, 4]],
        );

        assert_eq!(tet, exp);
    }

    #[test]
    fn test_tetrahedralization_flip_2_3() {
        // 2 tets to 3 tets
        let positions = vec![
            vec3(0.0, 1.0, 0.0),
            vec3(-1.0, -1.0, 0.0),
            vec3(1.0, -1.0, 0.0),
            vec3(0.0, 0.0, -1.0),
            vec3(0.0, 0.0, 1.0),
        ];
        let tets = vec![[0, 1, 2, 3], [0, 1, 2, 4]];
        let mut tet = Tetrahedralization::new(positions.clone(), tets);
        tet.flip([0, 1, 2]);
        tet.canonicalize();

        let exp_tets = vec![[0, 1, 3, 4], [1, 2, 3, 4], [0, 2, 3, 4]];
        let mut exp_tet = Tetrahedralization::new(positions, exp_tets);
        exp_tet.canonicalize();

        assert_eq!(tet, exp_tet);
    }

    #[test]
    fn test_tetrahedralization_flip_3_2() {
        // 3 tets to 2 tets
        let positions = vec![
            vec3(0.0, 1.0, 0.0),
            vec3(-1.0, -1.0, 0.0),
            vec3(1.0, -1.0, 0.0),
            vec3(0.0, 0.0, -1.0),
            vec3(0.0, 0.0, 1.0),
        ];
        let tets = vec![[0, 1, 3, 4], [1, 2, 3, 4], [0, 2, 3, 4]];
        let mut tet = Tetrahedralization::new(positions.clone(), tets);
        tet.flip([0, 3, 4]);
        tet.canonicalize();

        let exp_tets = vec![[0, 1, 2, 3], [0, 1, 2, 4]];
        let mut exp_tet = Tetrahedralization::new(positions, exp_tets);
        exp_tet.canonicalize();

        assert_eq!(tet, exp_tet);
    }

    #[test]
    fn test_tetrahedralization_flip_3_2_fail() {
        // 2 tets to concave angle fail
        let positions = vec![
            vec3(0.0, 1.0, 0.0),
            vec3(-1.0, -1.0, 0.0),
            vec3(1.0, -1.0, 0.0),
            vec3(0.0, 0.0, -1.0),
            vec3(0.0, 0.0, 1.0),
        ];
        let tets = vec![[0, 1, 3, 4], [0, 2, 3, 4]];
        let mut tet = Tetrahedralization::new(positions.clone(), tets);
        let clone = tet.clone();
        assert_eq!(tet.flip([0, 3, 4]), vec![]);
        assert_eq!(tet, clone);
    }

    #[test]
    fn test_tetrahedralization_flip_4_1() {
        // 4 tets to 1 tet
        let positions = vec![
            vec3(0.0, 1.0, 0.0),
            vec3(-1.0, -1.0, 0.0),
            vec3(1.0, -1.0, 0.0),
            vec3(0.0, 0.0, 2.0),
            vec3(0.0, 0.0, 1.0),
        ];
        let tets = vec![[0, 1, 2, 4], [0, 1, 3, 4], [0, 2, 3, 4], [1, 2, 3, 4]];
        let mut tet = Tetrahedralization::new(positions.clone(), tets);
        tet.flip([0, 1, 4]);
        tet.canonicalize();

        let exp_tets = vec![[0, 1, 2, 3]];
        let mut exp_tet = Tetrahedralization::new(positions, exp_tets);
        exp_tet.canonicalize();

        assert_eq!(tet, exp_tet);
    }

    #[test]
    fn test_tetrahedralization_flip_4_1_fail() {
        // 3 tets to concave angle fail
        let positions = vec![
            vec3(0.0, 1.0, 0.0),
            vec3(-1.0, -1.0, 0.0),
            vec3(1.0, -1.0, 0.0),
            vec3(0.0, 0.0, 2.0),
            vec3(0.0, 0.0, 1.0),
        ];
        let tets = vec![[0, 1, 2, 4], [0, 1, 3, 4], [1, 2, 3, 4]];
        let mut tet = Tetrahedralization::new(positions.clone(), tets);
        let clone = tet.clone();
        assert_eq!(tet.flip([0, 1, 4]), vec![]);
        assert_eq!(tet, clone);
    }

    #[test]
    fn test_tetrahedralization_flip_2_2() {
        // degenerate 2 faces to 2 faces
        let positions = vec![
            vec3(0.0, 1.0, 0.0),
            vec3(-1.0, -1.0, 0.0),
            vec3(1.0, -1.0, 0.0),
            vec3(0.0, -1.0, -1.0),
            vec3(0.0, -1.0, 1.0),
        ];
        let tets = vec![[0, 1, 2, 3], [0, 1, 2, 4]];
        let mut tet = Tetrahedralization::new(positions.clone(), tets);
        tet.flip([0, 1, 2]);
        tet.canonicalize();

        let exp_tets = vec![[0, 1, 3, 4], [0, 2, 3, 4]];
        let mut exp_tet = Tetrahedralization::new(positions, exp_tets);
        exp_tet.canonicalize();

        assert_eq!(tet, exp_tet);
    }

    #[test]
    fn test_tetrahedralization_flip_2_2_fail_other() {
        // degenerate 2 faces to 2 faces, but 3rd tetrahedron blocks flip
        let positions = vec![
            vec3(0.0, 1.0, 0.0),
            vec3(-1.0, -1.0, 0.0),
            vec3(1.0, -1.0, 0.0),
            vec3(0.0, -1.0, -1.0),
            vec3(0.0, -1.0, 1.0),
            vec3(0.0, -2.0, 0.0),
        ];
        let tets = vec![[0, 1, 2, 3], [0, 1, 2, 4], [1, 2, 4, 5]];
        let mut tet = Tetrahedralization::new(positions.clone(), tets);
        let clone = tet.clone();
        assert_eq!(tet.flip([0, 1, 2]), vec![]);
        assert_eq!(tet, clone);
    }

    #[test]
    fn test_tetrahedralization_flip_3_1() {
        // degenerate 3 faces to 1 face
        let positions = vec![
            vec3(0.0, 1.0, 0.0),
            vec3(-1.0, -1.0, 0.0),
            vec3(1.0, -1.0, 0.0),
            vec3(0.0, -1.0, -1.0),
            vec3(3.0, -1.0, 1.0),
        ];
        let tets = vec![[0, 1, 2, 3], [0, 1, 2, 4], [0, 2, 3, 4]];
        let mut tet = Tetrahedralization::new(positions.clone(), tets);
        tet.flip([0, 1, 2]);
        tet.canonicalize();

        let exp_tets = vec![[0, 1, 3, 4]];
        let mut exp_tet = Tetrahedralization::new(positions, exp_tets);
        exp_tet.canonicalize();

        assert_eq!(tet, exp_tet);
    }

    #[test]
    fn test_tetrahedralization_flip_3_1_fail() {
        // degenerate 2 faces to concave angle fail
        let positions = vec![
            vec3(0.0, 1.0, 0.0),
            vec3(-1.0, -1.0, 0.0),
            vec3(1.0, -1.0, 0.0),
            vec3(0.0, -1.0, -1.0),
            vec3(3.0, -1.0, 1.0),
        ];
        let tets = vec![[0, 1, 2, 3], [0, 1, 2, 4]];
        let mut tet = Tetrahedralization::new(positions.clone(), tets);
        let clone = tet.clone();
        assert_eq!(tet.flip([0, 1, 2]), vec![]);
        assert_eq!(tet, clone);
    }

    #[test]
    fn test_tetrahedralization_flip_3_1_fail_other() {
        // degenerate 3 faces to 1 face, but 4th tetrahedron blocks flip
        let positions = vec![
            vec3(0.0, 1.0, 0.0),
            vec3(-1.0, -1.0, 0.0),
            vec3(1.0, -1.0, 0.0),
            vec3(0.0, -1.0, -1.0),
            vec3(3.0, -1.0, 1.0),
            vec3(1.0, -2.0, 0.0),
        ];
        let tets = vec![[0, 1, 2, 3], [0, 1, 2, 4], [0, 2, 3, 4], [1, 2, 4, 5]];
        let mut tet = Tetrahedralization::new(positions.clone(), tets);
        let clone = tet.clone();
        assert_eq!(tet.flip([0, 1, 2]), vec![]);
        assert_eq!(tet, clone);
    }

    #[test]
    fn test_tetrahedralization_flip_2_1() {
        // degenerate 2 edges to 1 edge
        let positions = vec![
            vec3(0.0, 1.0, 0.0),
            vec3(-1.0, -1.0, 0.0),
            vec3(1.0, -1.0, 0.0),
            vec3(0.0, -1.0, -1.0),
            vec3(2.0, -1.0, 1.0),
        ];
        let tets = vec![[0, 1, 2, 3], [0, 1, 2, 4]];
        let mut tet = Tetrahedralization::new(positions.clone(), tets);
        tet.flip([0, 1, 2]);
        tet.canonicalize();

        let exp_tets = vec![[0, 1, 3, 4]];
        let mut exp_tet = Tetrahedralization::new(positions, exp_tets);
        exp_tet.canonicalize();

        assert_eq!(tet, exp_tet);
    }

    #[test]
    fn test_tetrahedralization_flip_2_1_fail_other() {
        // degenerate 2 edges to 1 edge, but 3rd tetrahedron blocks flip
        let positions = vec![
            vec3(0.0, 1.0, 0.0),
            vec3(-1.0, -1.0, 0.0),
            vec3(1.0, -1.0, 0.0),
            vec3(0.0, -1.0, -1.0),
            vec3(2.0, -1.0, 1.0),
            vec3(1.0, -2.0, 0.0),
        ];
        let tets = vec![[0, 1, 2, 3], [0, 1, 2, 4], [1, 2, 4, 5]];
        let mut tet = Tetrahedralization::new(positions.clone(), tets);
        let clone = tet.clone();
        assert_eq!(tet.flip([0, 1, 2]), vec![]);
        assert_eq!(tet, clone);
    }

    #[test]
    fn test_tetrahedralization_find_first_intersecting_edge() {
        let positions = vec![
            vec3(0.0, 1.0, 0.0),
            vec3(-1.0, -1.0, 0.0),
            vec3(1.0, -1.0, 0.0),
            vec3(0.0, 0.0, -1.0),
            vec3(0.0, 0.0, 1.0),
        ];
        let tets = vec![[0, 1, 3, 4], [1, 2, 3, 4], [0, 2, 3, 4]];
        let tet = Tetrahedralization::new(positions.clone(), tets);
        let (edges_to_search, inner_faces, inner_tets, point, face_edges) = tet
            .find_first_intersecting_edge(
                &[[0, 1], [1, 2], [2, 0]].iter().copied().collect(),
                Vec3::unit_z(),
                &(0..3).collect(),
            );

        assert_eq!(edges_to_search, vec![[3, 4]]);
        assert_eq!(
            inner_faces,
            vec![[0, 3, 4], [1, 3, 4]].into_iter().collect()
        );
        assert_eq!(inner_tets, vec![0].into_iter().collect());
        assert_eq!(point, vec3(0.0, 1.0, 0.0));
        assert_eq!(face_edges, vec![[0, 1]].into_iter().collect());
    }

    #[test]
    fn test_tetrahedralization_find_first_intersecting_edge_2() {
        let positions = vec![
            vec3(0.0, 1.0, 0.0),
            vec3(-1.0, -1.0, 0.0),
            vec3(1.0, -1.0, 0.0),
            vec3(0.0, 0.0, -1.0),
            vec3(0.0, 0.0, 1.0),
        ];
        let tets = vec![[0, 1, 2, 3], [0, 1, 2, 4]];
        let tet = Tetrahedralization::new(positions.clone(), tets);
        let (edges_to_search, inner_faces, inner_tets, point, face_edges) = tet
            .find_first_intersecting_edge(
                &[[0, 1], [1, 2], [2, 0]].iter().copied().collect(),
                Vec3::unit_z(),
                &(0..3).collect(),
            );

        assert_eq!(
            face_edges,
            vec![[0, 1], [1, 2], [2, 0]].into_iter().collect()
        );
    }

    #[test]
    fn test_tetrahedralization_find_first_intersecting_edge_bigger() {
        let positions = vec![
            vec3(0.0, 0.0, 0.0),
            vec3(3.0, 0.0, 0.0),
            vec3(3.0, 3.0, 0.0),
            vec3(0.0, 3.0, 0.0),
            vec3(1.0, 1.0, -1.0),
            vec3(2.0, 2.0, -1.0),
            vec3(1.0, 1.0, 1.0),
            vec3(2.0, 2.0, 1.0),
        ];
        let tets = vec![
            [0, 1, 4, 6],
            [0, 3, 4, 6],
            [1, 3, 4, 6],
            [1, 3, 4, 5],
            [1, 3, 6, 7],
            [1, 2, 5, 7],
            [1, 3, 5, 7],
            [2, 3, 5, 7],
        ];
        let tet = Tetrahedralization::new(positions.clone(), tets);
        let (edges_to_search, inner_faces, inner_tets, point, face_edges) = tet
            .find_first_intersecting_edge(
                &[[0, 1], [1, 2], [2, 3], [3, 0]].iter().copied().collect(),
                Vec3::unit_z(),
                &(0..4).collect(),
            );

        assert_eq!(edges_to_search, vec![[4, 6]]);
        assert_eq!(
            inner_faces,
            vec![[0, 4, 6], [1, 4, 6]].into_iter().collect()
        );
        assert_eq!(inner_tets, vec![0].into_iter().collect());
        assert_eq!(point, vec3(0.0, 0.0, 0.0));
        assert_eq!(face_edges, vec![[0, 1]].into_iter().collect());
    }

    #[test]
    fn test_tetrahedralization_find_all_intersecting_faces() {
        let positions = vec![
            vec3(0.0, 1.0, 0.0),
            vec3(-1.0, -1.0, 0.0),
            vec3(1.0, -1.0, 0.0),
            vec3(0.0, 0.0, -1.0),
            vec3(0.0, 0.0, 1.0),
        ];
        let tets = vec![[0, 1, 3, 4], [1, 2, 3, 4], [0, 2, 3, 4]];
        let tet = Tetrahedralization::new(positions.clone(), tets);
        let (mut edges_to_search, mut inner_faces, mut inner_tets, point, mut face_edges) = tet
            .find_first_intersecting_edge(
                &[[0, 1], [1, 2], [2, 0]].iter().copied().collect(),
                Vec3::unit_z(),
                &(0..3).collect(),
            );
        tet.find_all_intersecting_faces(
            Vec3::unit_z(),
            &mut edges_to_search,
            &mut inner_faces,
            &mut inner_tets,
            point,
            &mut face_edges,
            &(0..3).collect(),
        );

        assert_eq!(edges_to_search, vec![] as Vec<[usize; 2]>);
        assert_eq!(
            inner_faces,
            vec![[0, 3, 4], [1, 3, 4], [2, 3, 4]].into_iter().collect()
        );
        assert_eq!(inner_tets, vec![0, 1, 2].into_iter().collect());
        assert_eq!(point, vec3(0.0, 1.0, 0.0));
        assert_eq!(
            face_edges,
            vec![[0, 1], [1, 2], [2, 0]].into_iter().collect()
        );
    }

    #[test]
    fn test_tetrahedralization_find_all_intersecting_faces_bigger() {
        let positions = vec![
            vec3(0.0, 0.0, 0.0),
            vec3(3.0, 0.0, 0.0),
            vec3(3.0, 3.0, 0.0),
            vec3(0.0, 3.0, 0.0),
            vec3(1.0, 1.0, -1.0),
            vec3(2.0, 2.0, -1.0),
            vec3(1.0, 1.0, 1.0),
            vec3(2.0, 2.0, 1.0),
        ];
        let tets = vec![
            [0, 1, 4, 6],
            [0, 3, 4, 6],
            [1, 3, 4, 6],
            [1, 3, 4, 5],
            [1, 3, 6, 7],
            [1, 2, 5, 7],
            [1, 3, 5, 7],
            [2, 3, 5, 7],
        ];
        let tet = Tetrahedralization::new(positions.clone(), tets);
        let (mut edges_to_search, mut inner_faces, mut inner_tets, point, mut face_edges) = tet
            .find_first_intersecting_edge(
                &[[0, 1], [1, 2], [2, 3], [3, 0]].iter().copied().collect(),
                Vec3::unit_z(),
                &(0..4).collect(),
            );
        tet.find_all_intersecting_faces(
            Vec3::unit_z(),
            &mut edges_to_search,
            &mut inner_faces,
            &mut inner_tets,
            point,
            &mut face_edges,
            &(0..4).collect(),
        );

        assert_eq!(edges_to_search, vec![] as Vec<[usize; 2]>);
        assert_eq!(
            inner_faces,
            vec![[0, 4, 6], [1, 4, 6], [3, 4, 6]].into_iter().collect()
        );
        assert_eq!(inner_tets, vec![0, 1, 2].into_iter().collect());
        assert_eq!(point, vec3(0.0, 0.0, 0.0));
        assert_eq!(
            face_edges,
            vec![[0, 1], [1, 3], [3, 0]].into_iter().collect()
        );
    }

    #[test]
    fn test_tetrahedralization_find_all_intersecting_faces_one_vertex_case() {
        let positions = vec![
            vec3(0.0, 0.0, 0.0),
            vec3(3.0, 0.0, 0.0),
            vec3(3.0, 3.0, 0.0),
            vec3(0.0, 3.0, 0.0),
            vec3(1.0, 1.0, -1.0),
            vec3(2.0, 2.0, -1.0),
            vec3(1.0, 1.0, 1.0),
            vec3(2.0, 2.0, 1.0),
            vec3(1.5, 1.5, 2.0),
        ];
        let tets = vec![
            [0, 1, 4, 6],
            [0, 3, 4, 6],
            [3, 4, 5, 6],
            [3, 5, 6, 7],
            [1, 4, 5, 6],
            [1, 5, 6, 7],
            [1, 2, 5, 7],
            [2, 3, 5, 7],
            [3, 6, 7, 8],
        ];
        let tet = Tetrahedralization::new(positions.clone(), tets);
        let (mut edges_to_search, mut inner_faces, mut inner_tets, point, mut face_edges) = tet
            .find_first_intersecting_edge(
                &[[0, 1], [1, 2], [2, 3], [3, 0]].iter().copied().collect(),
                Vec3::unit_z(),
                &(0..4).collect(),
            );
        tet.find_all_intersecting_faces(
            Vec3::unit_z(),
            &mut edges_to_search,
            &mut inner_faces,
            &mut inner_tets,
            point,
            &mut face_edges,
            &(0..4).collect(),
        );

        assert_eq!(edges_to_search, vec![] as Vec<[usize; 2]>);
        assert_eq!(
            inner_faces,
            vec![
                [0, 4, 6],
                [1, 4, 6],
                [3, 4, 6],
                [4, 5, 6],
                [5, 6, 7],
                [1, 5, 7],
                [2, 5, 7],
                [3, 5, 7],
                [1, 5, 6],
                [3, 5, 6]
            ]
            .into_iter()
            .collect()
        );
        assert_eq!(
            inner_tets,
            vec![0, 1, 2, 3, 4, 5, 6, 7].into_iter().collect()
        );
        assert_eq!(point, vec3(0.0, 0.0, 0.0));
        assert_eq!(
            face_edges,
            vec![[0, 1], [1, 2], [2, 3], [3, 0]].into_iter().collect()
        );
    }
}
