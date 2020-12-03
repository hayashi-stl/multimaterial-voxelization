use fnv::{FnvHashMap, FnvHashSet};
use petgraph::prelude::*;
use petgraph::unionfind::UnionFind;
use tri_mesh::prelude::*;

use crate::material_mesh::{MaterialID, MaterialMesh};
use crate::tetrahedralize::DelaunayTetrahedralization;
use crate::util::GraphEx;

/// Vertices are nodes weighted by positions.
/// Faces are collections directed edges weighted by the same face ID.
#[derive(Clone, Debug)]
pub struct PiecewiseLinearComplex {
    mesh: Graph<Vec3, usize>,
    normals: Vec<Vec3>,
    materials: Vec<MaterialID>,
}

impl PiecewiseLinearComplex {
    const EPSILON: f64 = 1e-5;

    /// Constructs a new piecewise linear complex
    /// from a hopefully manifold tri-mesh.
    pub fn new(mesh: MaterialMesh) -> Self {
        let mesh = mesh.mesh();
        let vertices = mesh.vertex_iter().map(|v| mesh.vertex_position(v));

        let faces = mesh
            .face_iter()
            .map(|f| (mesh.face_vertices(f), mesh.face_normal(f), mesh.face_tag(f)));

        let mut graph = Graph::new();
        let mut normals = vec![];
        let mut materials = vec![];
        for pos in vertices {
            graph.add_node(pos);
        }
        for (i, ((a, b, c), normal, mat)) in faces.enumerate() {
            let a = NodeIndex::new(a.deref() as usize);
            let b = NodeIndex::new(b.deref() as usize);
            let c = NodeIndex::new(c.deref() as usize);
            graph.add_edge(a, b, i);
            graph.add_edge(b, c, i);
            graph.add_edge(c, a, i);

            normals.push(normal);
            materials.push(mat);
        }

        Self {
            mesh: graph,
            normals,
            materials,
        }
    }

    /// Gets rid of triangulation edges,
    /// i.e. edges that were only there to triangulate the mesh
    fn dissolve(&mut self) {
        let edges = self.mesh.edge_indices().collect::<Vec<_>>();

        // Maps old face IDs to replacement face IDs
        let mut mapping = UnionFind::new(self.normals.len());

        for edge in edges {
            if let Some((s, t)) = self.mesh.edge_endpoints(edge) {
                if let Some(opposite) = self.mesh.find_edge(t, s) {
                    // Remove edge if faces have the same normal and material.
                    // Assumes the mesh is manifold possibly with boundary
                    let f1 = self.mesh[edge];
                    let f2 = self.mesh[opposite];

                    if self.materials[f1] == self.materials[f2]
                        && (self.normals[f1] - self.normals[f2]).magnitude() < Self::EPSILON
                    {
                        self.mesh.remove_edge(edge);
                        self.mesh.remove_edge(opposite);
                        mapping.union(f1, f2);
                    }
                }
            }
        }

        let edges = self.mesh.edge_indices().collect::<Vec<_>>();

        // Fix face IDs
        for edge in edges {
            self.mesh[edge] = mapping.find(self.mesh[edge]);
        }

        // Remove lone vertices
        self.mesh.retain_nodes(|graph, n| graph.degree(n) > 0);
    }

    fn tetrahedralize_vertices(&mut self) -> DelaunayTetrahedralization {
        DelaunayTetrahedralization::new(self.mesh.node_weights_mut().map(|pos| *pos).collect())
    }

    fn recover_edges(&mut self, dt: &mut DelaunayTetrahedralization) {
        // Just subdivide each missing edge until it exists in the DT.
        // Edges here are undirected and thus their vertices are sorted by index.
        let mut edges = self
            .mesh
            .edge_references()
            .map(|e| {
                let (s, t) = (e.source().index(), e.target().index());
                if s < t {
                    (s, t)
                } else {
                    (t, s)
                }
            })
            .collect::<FnvHashSet<_>>();

        let mut missing = edges
            .difference(&dt.tetrahedron_edges().collect())
            .copied()
            .collect::<FnvHashSet<_>>();

        while let Some((s, t)) = missing.iter().next().copied() {
            missing.remove(&(s, t));
            let ns = NodeIndex::new(s);
            let nt = NodeIndex::new(t);

            // Subdivide
            let pos = (self.mesh[ns] + self.mesh[nt]) / 2.0;
            let v_new = self.mesh.add_node(pos);

            // At least one of these edges exists
            for (ns, nt) in vec![(ns, nt), (nt, ns)] {
                if let Some(edge) = self.mesh.find_edge(ns, nt) {
                    let face = self.mesh.remove_edge(edge).unwrap();
                    self.mesh.add_edge(ns, v_new, face);
                    self.mesh.add_edge(v_new, nt, face);
                }
            }

            edges.remove(&(s, t));
            edges.insert((s, v_new.index()));
            edges.insert((t, v_new.index()));
            missing.insert((s, v_new.index()));
            missing.insert((t, v_new.index()));

            let (removed, added) = dt.add_point(pos);
            for edge in removed {
                if edges.contains(&edge) {
                    missing.insert(edge);
                }
            }
            for edge in added {
                if edges.contains(&edge) {
                    missing.remove(&edge);
                }
            }
        }
    }

    /// Tetrahedralizes the mesh and returns the
    /// vertex positions and tetrahedrons.
    fn tetrahedralize(mut self) -> (Vec<Vec3>, Vec<([usize; 4], MaterialID)>) {
        let mut dt = self.tetrahedralize_vertices();
        self.recover_edges(&mut dt);

        // Recover faces

        todo!()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use petgraph::algo;
    use petgraph::data::{Element, FromElements};

    fn create_graph<N, I, E>(
        vertices: Vec<N>,
        edges: Vec<(usize, usize, I)>,
        mut edge_fn: impl FnMut(I) -> E,
    ) -> Graph<N, E> {
        Graph::from_elements(
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

    fn create_mesh(positions: Vec<f64>, indexes: Vec<u32>) -> MaterialMesh {
        MaterialMesh::new(
            MeshBuilder::<MaterialID>::new()
                .with_positions(positions)
                .with_indices(indexes)
                .build()
                .expect("Invalid mesh"),
        )
    }

    #[test]
    fn test_new() {
        let mesh = create_mesh(
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            vec![1, 0, 2, 2, 0, 3, 3, 0, 1, 1, 2, 3],
        );
        let graph = create_graph(
            vec![
                vec3(0.0, 0.0, 0.0),
                vec3(1.0, 0.0, 0.0),
                vec3(0.0, 1.0, 0.0),
                vec3(0.0, 0.0, 1.0),
            ],
            vec![
                (1, 0, 0),
                (0, 2, 0),
                (2, 1, 0),
                (2, 0, 1),
                (0, 3, 1),
                (3, 2, 1),
                (3, 0, 2),
                (0, 1, 2),
                (1, 3, 2),
                (1, 2, 3),
                (2, 3, 3),
                (3, 1, 3),
            ],
            |x| x,
        );
        let plc = PiecewiseLinearComplex::new(mesh);

        assert!(algo::is_isomorphic_matching(
            &plc.mesh,
            &graph,
            |x, y| x == y,
            |x, y| x == y
        ));
        assert_eq!(plc.normals.len(), 4);
        assert_eq!(plc.materials, vec![MaterialID::default(); 4]);
    }

    #[test]
    fn test_dissolve() {
        let mesh = create_mesh(
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            vec![0, 1, 2, 2, 3, 0],
        );
        let mut plc = PiecewiseLinearComplex::new(mesh);
        plc.dissolve();

        assert_eq!(plc.mesh.edge_count(), 4);
        let mut iter = plc.mesh.edge_references().map(|e| e.weight());
        let first = iter.next().unwrap();
        assert!(iter.all(|i| i == first));
    }

    #[test]
    #[ignore = "manual"]
    fn test_schonhardt_edges() {
        let mesh = create_mesh(
            vec![
                1.0,
                0.0,
                0.0,
                -0.5,
                0.0,
                -0.75f64.sqrt(),
                -0.5,
                0.0,
                0.75f64.sqrt(),
                0.75f64.sqrt(),
                1.5,
                -0.5,
                -0.75f64.sqrt(),
                1.5,
                -0.5,
                0.0,
                1.5,
                1.0,
            ],
            vec![
                2, 1, 0, 2, 0, 3, 3, 5, 2, 0, 1, 4, 4, 3, 0, 1, 2, 5, 5, 4, 1, 3, 4, 5,
            ],
        );
        let mut plc = PiecewiseLinearComplex::new(mesh);
        let mut dt = plc.tetrahedralize_vertices();
        plc.recover_edges(&mut dt);
        dt.export_debug_obj("assets/debug/dt_test3.obj");
    }
}
