use float_ord::FloatOrd;
use fnv::FnvHashMap;
use petgraph::graph::Edges;
use petgraph::prelude::*;
use std::fs;
use std::num::NonZeroU32;
use std::path::Path;
use tri_mesh::mesh_builder;
use tri_mesh::prelude::*;

use crate::util::{GraphEx, HashVec2, Vec2};

/// The ID type for a material
/// 0 is reserved for the absence of a material
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct MaterialID(NonZeroU32);

impl Default for MaterialID {
    fn default() -> Self {
        Self::new(1)
    }
}

impl MaterialID {
    pub fn new(id: u32) -> Self {
        Self(NonZeroU32::new(id).expect("Material ID can't be 0"))
    }
}

/// A triangle mesh with material
#[derive(Debug)]
pub struct MaterialMesh {
    mesh: Mesh<MaterialID>,
}
#[derive(Clone, Debug, Default)]
struct Intermediate {
    vertex_ids: FnvHashMap<VertexID, usize>,
    indexes: Vec<u32>,
    tags: Vec<MaterialID>,
}

#[derive(Copy, Clone, Debug)]
struct EdgeRange {
    min: f64,
    max: f64,
    halfedge_id: HalfEdgeID,
    /// Whether the edge is a triangulation edge
    triangulation: bool,
}

#[derive(Clone, Debug)]
struct FaceRange {
    min: f64,
    max: f64,
    face_proj_area: f64,
}

impl MaterialMesh {
    const EPSILON: f64 = 1e-5;

    pub fn new(mesh: Mesh<MaterialID>) -> Self {
        Self { mesh }
    }

    pub fn mesh(&self) -> &Mesh<MaterialID> {
        &self.mesh
    }

    pub fn export_debug_obj<P: AsRef<Path>>(&self, path: P) {
        let obj = self.mesh.parse_as_obj();
        fs::write(path, obj).expect("Could not debug obj");
    }

    /// Constructs a material mesh with 1 material from an OBJ string
    pub fn from_obj_1_material(source: String) -> Result<Self, mesh_builder::Error> {
        let mesh = MeshBuilder::new()
            .with_obj(source)
            .with_default_tag(MaterialID::new(1))
            .build()?;

        Ok(Self { mesh })
    }

    /// Dissolve an unnecessary boundary vertex
    fn dissolve_boundary_vertex(&mut self, vertex: VertexID) {
        let flippable_fn = |mesh: &Mesh<MaterialID>, e: HalfEdgeID| {
            if mesh.is_edge_on_boundary(e) {
                return false;
            }

            let mut walker = mesh.walker_from_halfedge(e);
            let e_dir = mesh.edge_vector(e);
            let dir1 = mesh.edge_vector(walker.next_id().unwrap());
            walker.as_twin();
            let dir0 = mesh.edge_vector(walker.previous_id().unwrap());

            // Check that dir0 → dir1 isn't a concave turn
            dir0.cross(dir1).dot(dir0.cross(-e_dir)) > 0.0
        };

        let inner = self
            .mesh
            .vertex_halfedge_iter(vertex)
            .filter(|e| !self.mesh.is_edge_on_boundary(*e))
            .collect::<Vec<_>>();

        let mut inner_count = inner.len();
        let mut flippable = inner
            .into_iter()
            .filter(|e| flippable_fn(&self.mesh, *e))
            .collect::<Vec<_>>();

        // Flip edges safely until there's no more non-boundary edges to flip
        while inner_count > 0 {
            if let Some(halfedge_id) = flippable.pop() {
                if self.mesh.flip_edge(halfedge_id).is_err() {
                    return;
                }

                // Check neighboring edges in 1-ring
                // Note that edge flips are counterclockwise.
                let mut walker = self.mesh.walker_from_halfedge(halfedge_id);
                let prev = walker.previous_id().unwrap();
                let next = walker.as_next().twin_id().unwrap();
                for edge in vec![prev, next] {
                    if !flippable.contains(&edge) && flippable_fn(&self.mesh, edge) {
                        flippable.push(edge);
                    }
                }

                inner_count -= 1;
            } else {
                return;
            }
        }

        // Now dissolve the vertex.
        self.mesh.remove_manifold_vertex(vertex);
    }

    /// Removes unnecessary vertices in the mesh.
    /// A vertex is unnecessary if either:
    /// * it's not a boundary vertex and either
    ///   * its adjacent faces are coplanar and of the same material
    ///   * there are 2 edges that split the vertex such that the second condition holds for each copy
    /// * its a boundary vertex and
    ///   * its adjacent faces are coplanar and of the same material and its adjacent boundary edges are collinear
    ///
    /// TODO: Non-manifold vertices
    fn decimate(&mut self) {
        let vertex_ids = self.mesh.vertex_iter().collect::<Vec<_>>();

        for vertex_id in vertex_ids {
            let normal = self.mesh.vertex_normal(vertex_id);
            // If min_dot is 1, the vertex is flat.
            let materials = self
                .mesh
                .vertex_halfedge_iter(vertex_id)
                .flat_map(|e| self.mesh.walker_from_halfedge(e).face_id())
                .map(|f| self.mesh.face_tag(f))
                .collect::<Vec<_>>();
            let uniform = materials.windows(2).all(|m| m[1] == m[0]);

            let min_dot = self
                .mesh
                .vertex_halfedge_iter(vertex_id)
                .flat_map(|e| self.mesh.walker_from_halfedge(e).face_id())
                .map(|f| FloatOrd(self.mesh.face_normal(f).dot(normal)))
                .min();

            if let Some(min_dot) = min_dot {
                let min_dot = min_dot.0;

                if self.mesh.is_vertex_on_boundary(vertex_id) {
                    // Boundary vertex. Faces must be coplanar, boundary edges must be collinear
                    if uniform && min_dot > 1.0 - Self::EPSILON {
                        let boundary = self
                            .mesh
                            .vertex_halfedge_iter(vertex_id)
                            .filter(|e| self.mesh.is_edge_on_boundary(*e))
                            .collect::<Vec<_>>();
                        let boundary_dirs = boundary
                            .iter()
                            .map(|e| self.mesh.edge_vector(*e).normalize())
                            .collect::<Vec<_>>();

                        if boundary_dirs[0].dot(boundary_dirs[1]) < -1.0 + Self::EPSILON {
                            self.dissolve_boundary_vertex(vertex_id);
                        }
                    }
                } else {
                    // TODO: Decide if bend and flat vertices should be decimated
                    //let bend = self
                    //    .mesh
                    //    .vertex_halfedge_iter(vertex_id)
                    //    .filter(|e| {
                    //        let mut walker = self.mesh.walker_from_halfedge(*e);
                    //        let faces = [
                    //            walker.face_id().unwrap(),
                    //            walker.as_twin().face_id().unwrap(),
                    //        ];

                    //        self.mesh.face_tag(faces[0]) != self.mesh.face_tag(faces[1])
                    //            || self
                    //                .mesh
                    //                .face_normal(faces[0])
                    //                .dot(self.mesh.face_normal(faces[1]))
                    //                < 1.0 - Self::DOT_EPSILON
                    //    })
                    //    .collect::<Vec<_>>();

                    //let bend_dirs = bend
                    //    .iter()
                    //    .map(|e| self.mesh.edge_vector(*e).normalize())
                    //    .collect::<Vec<_>>();

                    //if bend.len() == 2 && bend_dirs[0].dot(bend_dirs[1]) < -1.0 + Self::DOT_EPSILON
                    //{
                    //    // Bend vertex. Faces on each side must be coplanar, bend edges must be collinear (√)
                    //    let mut faces: Vec<Vec<FaceID>> = vec![vec![], vec![]];
                    //    let mut walker = self.mesh.walker_from_halfedge(bend[0]);

                    //    while walker.halfedge_id().unwrap() != bend[1] {
                    //        walker.as_twin().as_next();
                    //        faces[0].push(walker.face_id().unwrap());
                    //    }
                    //    while walker.halfedge_id().unwrap() != bend[0] {
                    //        walker.as_twin().as_next();
                    //        faces[1].push(walker.face_id().unwrap());
                    //    }

                    //    let normals = faces
                    //        .iter()
                    //        .map(|arr| {
                    //            arr.iter()
                    //                .map(|f| self.mesh.face_normal(*f))
                    //                .sum::<Vec3>()
                    //                .normalize()
                    //        })
                    //        .collect::<Vec<_>>();
                    //    let min_dots = faces
                    //        .iter()
                    //        .zip(normals.iter())
                    //        .map(|(arr, normal)| {
                    //            arr.iter()
                    //                .map(|f| FloatOrd(self.mesh.face_normal(*f).dot(*normal)))
                    //                .min()
                    //                .unwrap()
                    //                .0
                    //        })
                    //        .collect::<Vec<_>>();

                    //    if min_dots.into_iter().all(|d| d > 1.0 - Self::DOT_EPSILON) {
                    //        // TODO: Remove vertex
                    //        println!("Redundant bend vertex: {:?}", vertex_id);
                    //    }
                    //} else if bend.len() == 0 {
                    //    // Flat vertex. Faces on each side must be coplanar.
                    //    if uniform && min_dot > 1.0 - Self::DOT_EPSILON {
                    //        // TODO: Remove vertex
                    //        println!("Redundant flat vertex: {:?}", vertex_id);
                    //    }
                    //}
                }
            }
        }
    }

    /// Assumes the edge points in the +axis direction
    /// Returns half-edges that may need to be split.
    /// The first one returned, if any, is the new half-edge resulting
    /// from the split. The others are triangulation edges.
    fn split_edge(&mut self, axis: Axis, slice_coord: f64, range: EdgeRange) -> Vec<HalfEdgeID> {
        let EdgeRange {
            min,
            max,
            halfedge_id,
            ..
        } = range;
        let axis_id = axis as usize;

        let mut pos = self.mesh.edge_positions(halfedge_id);
        let vertices = self.mesh.edge_vertices(halfedge_id);
        // We assume the edge points in the +axis direction, as that was enforced

        // Detect degenerate cases
        if (slice_coord - min).abs() < Self::EPSILON {
            pos.0[axis_id] = slice_coord;
            self.mesh.move_vertex_to(vertices.0, pos.0);
        }

        if (slice_coord - max).abs() < Self::EPSILON {
            pos.1[axis_id] = slice_coord;
            self.mesh.move_vertex_to(vertices.1, pos.1);
        }

        if pos.0[axis_id] == slice_coord || pos.1[axis_id] == slice_coord {
            // Degenerate case; abort
            return if pos.1[axis_id] != slice_coord {
                vec![halfedge_id]
            } else {
                vec![]
            };
        }

        let t = (slice_coord - min) / (max - min);
        let mut inter = vec3(
            slice_coord,
            pos.0[(axis_id + 1) % 3] * (1.0 - t) + pos.1[(axis_id + 1) % 3] * t,
            pos.0[(axis_id + 2) % 3] * (1.0 - t) + pos.1[(axis_id + 2) % 3] * t,
        );

        // Rotate vector properly
        for _ in 0..axis_id {
            inter = vec3(inter.z, inter.x, inter.y);
        }

        let (vertex_id, split_halfedge_id) = self.mesh.split_edge(halfedge_id, inter);
        let mut vec = self
            .mesh
            .vertex_halfedge_iter(vertex_id)
            .filter(|h| {
                let pos = self.mesh.edge_positions(*h);
                pos.1[axis_id] > pos.0[axis_id]
            })
            .collect::<Vec<_>>();

        vec.sort_unstable_by_key(|h| *h != split_halfedge_id);
        vec
    }

    /// Draws contours on the mesh along evenly spaced axis-aligned planes.
    /// One of the planes crosses the origin.
    pub fn axis_contour(&mut self, axis: Axis, spacing: f64, min_slice: f64, max_slice: f64) {
        let axis_id = axis as usize;
        let min = min_slice;
        let max = max_slice;

        // Obtain extreme coordinates of edges
        let mut ranges = self
            .mesh
            .edge_iter()
            .map(|mut halfedge_id| {
                let mut pos = self.mesh.edge_positions(halfedge_id);

                // We want the edge to point in the +axis direction
                if pos.0[axis_id] > pos.1[axis_id] {
                    pos = (pos.1, pos.0);
                    halfedge_id = self
                        .mesh
                        .walker_from_halfedge(halfedge_id)
                        .as_twin()
                        .halfedge_id()
                        .expect("Half-edge doesn't have a twin");
                }

                EdgeRange {
                    min: pos.0[axis_id],
                    max: pos.1[axis_id],
                    halfedge_id,
                    triangulation: false,
                }
            })
            .collect::<Vec<_>>();

        ranges.sort_by_key(|range| FloatOrd(range.min));
        ranges.reverse();

        // Edges that are currently being contoured
        let mut edges = vec![];
        // Triangulation edges that are currently being contoured
        let mut tri_edges = vec![];

        let mut slice_coord = min + spacing;

        while !ranges.is_empty() {
            // Add epsilons so the edge-slicing code can deal with
            // edges that come EXTEREMELY close to slice planes but don't quite reach
            while ranges
                .last()
                .map_or(false, |r| r.min - Self::EPSILON < slice_coord)
            {
                if let Some(range) = ranges.pop() {
                    if range.max + Self::EPSILON > slice_coord {
                        if range.triangulation {
                            &mut tri_edges
                        } else {
                            &mut edges
                        }
                        .push(range)
                    }
                }
            }

            // Split the edges
            for range in edges.drain(..) {
                for (i, new_halfedge_id) in self
                    .split_edge(axis, slice_coord, range)
                    .into_iter()
                    .enumerate()
                {
                    // Edge may still need more splitting
                    ranges.push(EdgeRange {
                        min: slice_coord,
                        max: self.mesh.edge_positions(new_halfedge_id).1[axis_id],
                        halfedge_id: new_halfedge_id,
                        triangulation: i != 0,
                    });
                }
            }

            // Rotate triangulation edges to avoid clutter
            for range in tri_edges.drain(..) {
                self.mesh
                    .flip_edge(range.halfedge_id)
                    .expect("Could not flip triangulation edge");
            }

            slice_coord += spacing;
        }
    }

    /// Slices the mesh into regions based on the contours.
    /// The lesser-coordinate slicing plane is also returned for each slice.
    fn contour_slice(
        &self,
        axis: Axis,
        spacing: f64,
        min_slice: f64,
        max_slice: f64,
    ) -> Vec<(f64, MaterialMesh)> {
        let axis_id = axis as usize;

        let min = min_slice;
        let max = max_slice;

        let mut imms = vec![Intermediate::default(); ((max - min) / spacing) as usize];

        for face_id in self.mesh.face_iter() {
            // Find slice the face is in
            let center = self.mesh.face_center(face_id)[axis_id];
            let slice = ((center - min) / spacing).floor() as usize;

            // Insert face into the slice
            let tag = self.mesh.face_tag(face_id);
            if slice as f64 * spacing + min < center {
                let imm = &mut imms[slice];

                let vertices = self.mesh.face_vertices(face_id);

                for vertex in vec![vertices.0, vertices.1, vertices.2] {
                    let len = imm.vertex_ids.len();
                    let index = *imm.vertex_ids.entry(vertex).or_insert(len);
                    imm.indexes.push(index as u32);
                }
                imm.tags.push(tag);
            }
        }

        imms.into_iter()
            .enumerate()
            .filter(|(i, imm)| imm.vertex_ids.len() > 0)
            .map(|(i, imm)| {
                let mut positions = vec![0.0; imm.vertex_ids.len() * 3];
                for (vertex, index) in imm.vertex_ids {
                    let position = self.mesh.vertex_position(vertex);
                    positions[index * 3 + 0] = position[0];
                    positions[index * 3 + 1] = position[1];
                    positions[index * 3 + 2] = position[2];
                }

                (
                    min + i as f64 * spacing,
                    MaterialMesh::new(
                        MeshBuilder::new()
                            .with_positions(positions)
                            .with_indices(imm.indexes)
                            .with_tags(imm.tags)
                            .build()
                            .expect("Invalid mesh"),
                    ),
                )
            })
            .collect()
    }

    /// Slices the mesh along evenly spaced axis-aligned planes.
    /// One of the planes crosses the origin.
    /// Slices are ordered from minimium coordinate to maximum coordinate.
    /// Does not use the mesh's split_at_intersection() method because
    /// the runtime can be faster in this case.
    ///
    /// Also returns the lesser slice coordinate for each slice
    pub fn axis_slice(mut self, axis: Axis, spacing: f64) -> Vec<(f64, MaterialMesh)> {
        let extreme = self.mesh.extreme_coordinates();
        let min = (extreme.0[axis as usize] / spacing - Self::EPSILON).floor() * spacing;
        let max = (extreme.1[axis as usize] / spacing + Self::EPSILON).ceil() * spacing;

        self.axis_contour(axis, spacing, min, max);
        let mut slices = self.contour_slice(axis, spacing, min, max);
        for (_, slice) in slices.iter_mut() {
            slice.decimate();
        }
        slices
    }

    /// Calculates the ranges that the faces take up along some axes
    /// and calculates the in-out gradient of each range.
    /// The in-out gradient is
    /// * -1 if travelling along the +axis direction goes from outside to inside
    /// * 0 if there is no change
    /// * 1 if travelling along the +axis direction goes from inside to outside
    pub fn axis_ranges_and_in_out_gradients(
        &self,
        axis: Axis,
        cross_section_area: f64,
    ) -> Vec<(f64, f64, i32)> {
        let axis_id = axis as usize;
        let axis_vec = axis.unit_dir();

        let mut face_ranges = self
            .mesh
            .face_iter()
            .map(|f| {
                let pos = self.mesh.face_positions(f);

                FaceRange {
                    min: pos.0[axis_id].min(pos.1[axis_id]).min(pos.2[axis_id]),
                    max: pos.0[axis_id].max(pos.1[axis_id]).max(pos.2[axis_id]),
                    face_proj_area: (pos.1 - pos.0).cross(pos.2 - pos.0).dot(axis_vec) / 2.0,
                }
            })
            .collect::<Vec<_>>();

        face_ranges.sort_by_key(|range| FloatOrd(range.min));

        let mut ranges: Vec<Interval> = vec![];

        // Build the ranges and accumulate projection areas
        for face_range in face_ranges {
            if ranges.is_empty() || face_range.min > ranges.last().unwrap().max {
                // New interval necessary
                ranges.push(Interval {
                    min: face_range.min,
                    max: face_range.max,
                    proj_area: face_range.face_proj_area,
                });
            } else {
                // Expand old interval
                let range = ranges.last_mut().unwrap();
                range.max = range.max.max(face_range.max);
                range.proj_area += face_range.face_proj_area;
            }
        }

        ranges
            .into_iter()
            .map(|range| {
                (
                    range.min,
                    range.max,
                    (range.proj_area / cross_section_area).round() as i32,
                )
            })
            .collect()
    }

    /// Moves any vertices that are very close to slice planes
    /// onto the slice planes.
    /// This should be called before getting the ranges of the faces
    /// to avoid close calls of complex voxels that should be pure voxels.
    pub fn align_with_slice_planes(&mut self, axis: Axis, spacing: f64) {
        for vertex_id in self.mesh.vertex_iter() {
            let mut pos = self.mesh.vertex_position(vertex_id);
            let slice_plane = (pos[axis as usize] / spacing).round() * spacing;

            if (slice_plane - pos[axis as usize]).abs() < Self::EPSILON {
                pos[axis as usize] = slice_plane;
                self.mesh.move_vertex_to(vertex_id, pos);
            }
        }
    }

    /// Gets a graph of the boundary, with correct
    /// winding direction on the edges.
    /// Lone vertices are removed.
    fn boundary_graph(&self) -> Graph<Vec3, ()> {
        let mut graph = Graph::new();
        let mut indexes = FnvHashMap::default();

        for vertex in self.mesh.vertex_iter() {
            if self.mesh.is_vertex_on_boundary(vertex) {
                let pos = self.mesh.vertex_position(vertex);
                let index = graph.add_node(pos);
                indexes.insert(vertex, index);
            }
        }

        for halfedge in self.mesh.halfedge_iter() {
            if self.mesh.is_edge_on_boundary(halfedge)
                && self.mesh.walker_from_halfedge(halfedge).face_id().is_some()
            {
                let vtx = self.mesh.edge_vertices(halfedge);
                graph.add_edge(indexes[&vtx.0], indexes[&vtx.1], ());
            }
        }

        graph.retain_nodes(|graph, n| graph.degree(n) > 0);

        graph
    }

    /// Combine vertices with equal positions on the boundary graph.
    /// Also removes resulting self-loops.
    fn combine_equal_vertices(boundary: &Graph<Vec2, ()>) -> Graph<Vec2, ()> {
        let mut res = Graph::new();

        let mut position_map = FnvHashMap::default();

        for node in boundary.node_indices() {
            position_map
                .entry(HashVec2(boundary[node]))
                .or_insert_with(|| res.add_node(boundary[node]));
        }

        for edge in boundary.edge_indices() {
            let nodes = boundary.edge_endpoints(edge).unwrap();
            let index0 = position_map[&HashVec2(boundary[nodes.0])];
            let index1 = position_map[&HashVec2(boundary[nodes.1])];
            if index0 != index1 {
                res.update_edge(index0, index1, ());
            }
        }

        res
    }

    // Returns true iff there was enough information to compute the intersection.
    fn intersect_center_unit_square_on_graph(boundary: &mut Graph<Vec2, ()>) -> bool {
        *boundary = Self::combine_equal_vertices(boundary);

        // Remove slits. They could go either way and can't be used
        // to determine the intersection.
        boundary.retain_edges(|graph, e| {
            let nodes = graph.edge_endpoints(e).unwrap();
            graph.find_edge(nodes.1, nodes.0).is_none()
        });

        // Ignore parts of the boundary that are also part of the boundary *of* the square
        // and go counterclockwise around it.
        // Boundary must be clockwise so it can be filled with a counterclockwise region.
        let mut ignored_ccw = false;
        boundary.retain_edges(|sub, e| {
            let (v0, v1) = sub.edge_endpoints(e).unwrap();
            let pos0 = sub[v0];
            let pos1 = sub[v1];

            // Edge is on boundary of face if at least 1 coordinate is ±0.5 and the same
            if ((pos0.x.abs() == 0.5 && pos0.x == pos1.x)
                || (pos0.y.abs() == 0.5 && pos0.y == pos1.y))
                && pos0.perp_dot(pos1) > 0.0
            {
                ignored_ccw = true;
                false
            } else {
                true
            }
        });

        // There may be lone vertices. Remove those
        boundary.retain_nodes(|sub, n| sub.indegree(n) + sub.outdegree(n) > 0);

        // Now for clockwise square-boundary edges.
        // Importantly, don't remove lone vertices afterward
        // to prevent 0-area holes from showing up.
        let mut ignored_cw = false;
        boundary.retain_edges(|sub, e| {
            let (v0, v1) = sub.edge_endpoints(e).unwrap();
            let pos0 = sub[v0];
            let pos1 = sub[v1];

            // Edge is on boundary of face if at least 1 coordinate is ±0.5 and the same
            if (pos0.x.abs() == 0.5 && pos0.x == pos1.x)
                || (pos0.y.abs() == 0.5 && pos0.y == pos1.y)
            {
                ignored_cw = true;
                false
            } else {
                true
            }
        });

        // If there's nothing left and there were counterclockwise edges on the face's boundary,
        // the square is not part of the intersection at all.
        if ignored_ccw && !ignored_cw && boundary.edge_count() == 0 {
            return true;
        }

        let mut node_map = boundary
            .node_indices()
            .map(|n| (HashVec2(boundary[n]), n))
            .collect::<FnvHashMap<_, _>>();

        // Find and order boundary points, including square corners
        let mut points = vec![
            vec2(-0.5, -0.5),
            vec2(-0.5, 0.5),
            vec2(0.5, 0.5),
            vec2(0.5, -0.5),
        ]
        .into_iter()
        .chain(
            boundary
                .node_weights_mut()
                .filter(|pos| pos.x.abs() == 0.5 || pos.y.abs() == 0.5)
                .map(|pos| *pos),
        )
        .collect::<Vec<_>>();

        points.sort_by_key(|pos| {
            FloatOrd(if pos.x == -0.5 {
                0.5 + pos.y
            } else if pos.y == 0.5 {
                1.5 + pos.x
            } else if pos.x == 0.5 {
                2.5 - pos.y
            } else if pos.y == -0.5 {
                3.5 - pos.x
            } else {
                unreachable!()
            })
        });
        points.dedup_by_key(|pos| HashVec2(*pos));

        // Finds the minimum dot product between the given vec and given edges
        let dot_fn = |graph: &Graph<Vec2, ()>, vec: Vec2, edges: Edges<(), Directed>| {
            edges
                .map(|edge| {
                    FloatOrd(
                        (graph[edge.target()] - graph[edge.source()])
                            .normalize()
                            .dot(vec),
                    )
                })
                .min()
                .map(|f| f.0)
        };

        // Find where to start drawing the square.
        // Look for an odd-degree vertex that needs an outgoing edge as a priority.
        //
        // If it's an even-degree vertex, there are only even-degree vertices.
        // Do a special check to determine whether a square should be drawn.
        //
        // If there's no edges left and there were clockwise edges on the face's boundary,
        // the entire square is the intersection.
        let index = points
            .iter()
            .position(|pos| {
                node_map
                    .get(&HashVec2(*pos))
                    .map(|n| boundary.outdegree(*n) < boundary.indegree(*n))
                    .unwrap_or(false)
            })
            .or_else(|| {
                points.iter().enumerate().position(|(i, pos)| {
                    node_map
                        .get(&HashVec2(*pos))
                        .map(|n| {
                            let next = points[(i + 1) % points.len()];
                            let diff = next - *pos;

                            boundary.outdegree(*n) == boundary.indegree(*n)
                                && dot_fn(
                                    &boundary,
                                    diff,
                                    boundary.edges_directed(*n, Direction::Incoming),
                                )
                                .unwrap_or(-2.0)
                                    < dot_fn(&boundary, -diff, boundary.edges(*n)).unwrap_or(-1.0)
                        })
                        .unwrap_or(false)
                })
            })
            .or(
                if boundary.edge_count() == 0 && ignored_cw && !ignored_ccw {
                    Some(0)
                } else {
                    None
                },
            );

        //for (i, point) in points.iter().enumerate() {
        //    println!("{}: {:?}", i, point);
        //}
        if let Some(index) = index {
            // Draw the square around the boundary, but only the parts that are in the intersection.
            let mut i = index;
            let mut inside = true;

            // Basically a do-while loop
            while {
                let j = (i + 1) % points.len();

                let new_inside = node_map
                    .get(&HashVec2(points[j]))
                    .map(|n| boundary.degree(*n) % 2 != 0)
                    .unwrap_or(false)
                    != inside;

                //println!("i: {}, j: {}, inside: {}, new_inside: {}", i, j, inside, new_inside);

                if inside {
                    let node_i = *node_map
                        .entry(HashVec2(points[i]))
                        .or_insert_with(|| boundary.add_node(points[i]));
                    let node_j = *node_map
                        .entry(HashVec2(points[j]))
                        .or_insert_with(|| boundary.add_node(points[j]));

                    boundary.add_edge(node_i, node_j, ());
                }

                inside = new_inside;

                i = j;
                i != index
            } {}
            true
        } else if boundary.edge_count() > 0 {
            // At least there are hole(s). Use them to determine whether a square should be drawn.
            // First, look for an outer hole.

            // TODO: Handle holes with slits

            // Obtain bottommost vertex. In case of tie, get rightmost vertex
            let node = boundary
                .node_indices()
                .min_by_key(|n| {
                    let pos = boundary[*n];
                    (FloatOrd(pos.y), FloatOrd(-pos.x))
                })
                .expect("There should be a node here");

            // The vertex is guaranteed to be convex and at an outer hole.
            // Determine which way it goes.
            let square = dot_fn(&boundary, -Vec2::unit_x(), boundary.edges(node))
                < dot_fn(
                    &boundary,
                    Vec2::unit_x(),
                    boundary.edges_directed(node, Direction::Incoming),
                );

            if square {
                let mut indexes = points
                    .into_iter()
                    .map(|pos| boundary.add_node(pos))
                    .collect::<Vec<_>>();
                indexes.push(indexes[0]);

                for ii in indexes.windows(2) {
                    boundary.add_edge(ii[0], ii[1], ());
                }
            }
            true
        } else {
            false
        }
    }

    fn intersect_center_unit_square_with_context(&self, boundary: &mut Graph<Vec2, ()>) {
        // Find the volume under the mesh to determine
        // whether the square is outside or inside.
        let volume = self
            .mesh
            .face_iter()
            .map(|f| {
                let center = self.mesh.face_center(f);
                let pos = self.mesh.face_positions(f);
                // Technically need to divide by 2, but it doesn't matter
                (center.z - 1.0) * (pos.1 - pos.0).cross(pos.2 - pos.0).dot(Vec3::unit_z())
            })
            .sum::<f64>();

        if volume < 0.0 {
            let mut indexes = vec![
                vec2(-0.5, -0.5),
                vec2(-0.5, 0.5),
                vec2(0.5, 0.5),
                vec2(0.5, -0.5),
            ]
            .into_iter()
            .map(|pos| boundary.add_node(pos))
            .collect::<Vec<_>>();
            indexes.push(indexes[0]);

            for ii in indexes.windows(2) {
                boundary.add_edge(ii[0], ii[1], ());
            }
        }
    }

    fn intersect_center_unit_square(
        mesh_fn: impl FnOnce() -> Self,
        mut boundary: Graph<Vec2, ()>,
    ) -> Vec<[Vec2; 3]> {
        if !Self::intersect_center_unit_square_on_graph(&mut boundary) {
            mesh_fn().intersect_center_unit_square_with_context(&mut boundary);
        }

        todo!()
    }

    /// Gets the intersection of a unit cube
    /// and a manifold mesh potentially with boundary.
    /// It is assumed that the mesh's boundary is entirely
    /// on the surface of the cube and that no triangles
    /// are coplanar with a cube face.
    pub fn intersect_unit_cube(mut self, cube_min: Vec3) -> Self {
        self.mesh.translate(-cube_min - vec3(0.5, 0.5, 0.5));
        // Cube center is at origin
        let boundary = self.boundary_graph();

        let mut triangles = self
            .mesh
            .face_iter()
            .map(|f| {
                let pos = self.mesh.face_positions(f);
                [pos.0, pos.1, pos.2]
            })
            .collect::<Vec<_>>();

        // Fill in all 6 cube faces appropriately
        for normal in vec![
            Vec3::unit_x(),
            -Vec3::unit_x(),
            Vec3::unit_y(),
            -Vec3::unit_y(),
            Vec3::unit_z(),
            -Vec3::unit_z(),
        ] {
            // Transform to unit square with (0.0, 0.0) as center
            let tangent = Transform::<Point3<f64>>::transform_vector(
                &Mat3::from_cols(Vec3::unit_y(), Vec3::unit_z(), Vec3::unit_x()),
                normal,
            );
            let square_transform = Mat4::from_cols(
                tangent.extend(0.0),
                normal.cross(tangent).extend(0.0),
                normal.extend(0.0),
                (normal * 0.5).extend(1.0),
            );
            let inv_square_transform = square_transform.invert().unwrap();

            // Get part of boundary on this face
            let mut sub: Graph<Vec2, ()> = boundary.filter_map(
                |_, node| {
                    if node.dot(normal) == 0.5 {
                        Some(
                            inv_square_transform
                                .transform_point(Point3::from_homogeneous(node.extend(1.0)))
                                .to_vec()
                                .truncate(),
                        )
                    } else {
                        None
                    }
                },
                |_, e| Some(*e),
            );

            triangles.extend(
                Self::intersect_center_unit_square(
                    || MaterialMesh::new(self.mesh.transformed(inv_square_transform)),
                    sub,
                )
                .into_iter()
                .map(|tri_2d| {
                    let mut tri_3d = [Vec3::zero(); 3];
                    for i in 0..3 {
                        tri_3d[i] = square_transform
                            .transform_point(Point3::from_homogeneous(
                                tri_2d[i].extend(0.0).extend(1.0),
                            ))
                            .to_vec();
                    }
                    tri_3d
                }),
            );
        }

        self
    }
}

struct Interval {
    min: f64,
    max: f64,
    proj_area: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Axis {
    X,
    Y,
    Z,
}

impl Axis {
    pub fn unit_dir(self) -> Vec3 {
        let mut vec = Vec3::zero();
        vec[self as usize] = 1.0;
        vec
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use petgraph::algo;
    use petgraph::data::{Element, FromElements};

    fn graph_from_mesh(
        mesh: &MaterialMesh,
        vertices: Vec<usize>,
        edges: Vec<(usize, usize)>,
    ) -> Graph<Vec3, ()> {
        let v = mesh.mesh.vertex_iter().collect::<Vec<_>>();

        Graph::from_elements(
            vertices
                .into_iter()
                .map(|vertex| Element::Node {
                    weight: mesh.mesh.vertex_position(v[vertex]),
                })
                .chain(edges.into_iter().map(|(s, t)| Element::Edge {
                    source: s,
                    target: t,
                    weight: (),
                })),
        )
    }

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

    fn create_mesh(positions: Vec<f64>, indexes: Vec<u32>) -> MaterialMesh {
        MaterialMesh {
            mesh: MeshBuilder::<MaterialID>::new()
                .with_positions(positions)
                .with_indices(indexes)
                .build()
                .expect("Invalid mesh"),
        }
    }

    #[test]
    fn test_dissolve_boundary_vertex_simple() {
        let mut mesh = create_mesh(
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.5, 1.0, 0.0],
            vec![0, 1, 3, 3, 1, 2],
        );

        let vertex = mesh.mesh.vertex_iter().collect::<Vec<_>>()[1];
        mesh.dissolve_boundary_vertex(vertex);

        assert_eq!(mesh.mesh.num_vertices(), 3);
        assert_eq!(mesh.mesh.num_faces(), 1);
        assert!(!mesh
            .mesh
            .vertex_iter()
            .collect::<Vec<_>>()
            .contains(&vertex));
    }

    #[test]
    fn test_dissolve_boundary_vertex_multiple_inner() {
        let mut mesh = create_mesh(
            vec![
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.5, 1.0, 0.0, 1.0, 1.0, 0.0,
            ],
            vec![0, 1, 4, 4, 1, 3, 3, 1, 2],
        );

        let vertex = mesh.mesh.vertex_iter().collect::<Vec<_>>()[1];
        mesh.dissolve_boundary_vertex(vertex);

        assert_eq!(mesh.mesh.num_vertices(), 4);
        assert_eq!(mesh.mesh.num_faces(), 2);
        assert!(!mesh
            .mesh
            .vertex_iter()
            .collect::<Vec<_>>()
            .contains(&vertex));
    }

    //#[test]
    //fn test_dissolve_boundary_vertex_different_materials() {
    //    let mut mesh = MaterialMesh { mesh:
    //        MeshBuilder::<MaterialID>::new()
    //            .with_positions(vec![
    //                0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.5, 1.0, 0.0, 1.0, 1.0, 0.0
    //            ])
    //            .with_indices(vec![0, 1, 4, 4, 1, 3, 3, 1, 2])
    //            .with_tags(vec![MaterialID::new(1), MaterialID::new(1), MaterialID::new(2)])
    //            .build()
    //            .expect("Invalid mesh")
    //    };

    //    let vertex = mesh.mesh.vertex_iter().collect::<Vec<_>>()[1];
    //    mesh.dissolve_boundary_vertex(vertex);

    //    // Nothing should have happened.
    //    assert_eq!(mesh.mesh.num_vertices(), 5);
    //    assert_eq!(mesh.mesh.num_faces(), 3);
    //}

    #[test]
    fn test_dissolve_boundary_vertex_concave() {
        let mut mesh = create_mesh(
            vec![
                0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 6.0, 0.0, 0.0, 4.0, 1.0, 0.0, 3.0, 3.0, 0.0, 2.0,
                1.0, 0.0,
            ],
            vec![0, 1, 5, 5, 1, 4, 4, 1, 3, 3, 1, 2],
        );

        let v = mesh.mesh.vertex_iter().collect::<Vec<_>>();
        mesh.dissolve_boundary_vertex(v[1]);

        assert_eq!(mesh.mesh.num_vertices(), 5);
        assert_eq!(mesh.mesh.num_faces(), 3);
        assert!(!mesh.mesh.vertex_iter().collect::<Vec<_>>().contains(&v[1]));
        assert!(
            mesh.mesh
                .vertex_halfedge_iter(v[3])
                .map(|e| mesh.mesh.edge_vertices(e).1)
                .collect::<Vec<_>>()
                .contains(&v[5]),
            "Triangulation doesn't respect concave vertices"
        );
    }

    #[test]
    fn test_boundary_graph_triangle() {
        let mesh = create_mesh(
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            vec![0, 1, 2],
        );

        let graph = mesh.boundary_graph();
        let expected = graph_from_mesh(&mesh, vec![0, 1, 2], vec![(0, 1), (2, 0), (1, 2)]);
        assert!(algo::is_isomorphic_matching(
            &graph,
            &expected,
            |x, y| x == y,
            |x, y| x == y
        ));
    }

    #[test]
    fn test_boundary_graph_square() {
        let mesh = create_mesh(
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            vec![0, 1, 2, 3, 2, 1],
        );

        let graph = mesh.boundary_graph();
        let expected = graph_from_mesh(
            &mesh,
            vec![0, 1, 2, 3],
            vec![(0, 1), (1, 3), (3, 2), (2, 0)],
        );
        assert!(algo::is_isomorphic_matching(
            &graph,
            &expected,
            |x, y| x == y,
            |x, y| x == y
        ));
    }

    #[test]
    fn test_boundary_graph_poked_square() {
        let mesh = create_mesh(
            vec![
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.5, 0.5, 0.0,
            ],
            vec![0, 4, 2, 2, 4, 3, 3, 4, 1, 1, 4, 0],
        );

        let graph = mesh.boundary_graph();
        let expected = graph_from_mesh(
            &mesh,
            vec![0, 1, 2, 3],
            vec![(0, 1), (1, 3), (3, 2), (2, 0)],
        );
        assert!(algo::is_isomorphic_matching(
            &graph,
            &expected,
            |x, y| x == y,
            |x, y| x == y
        ));
    }

    #[test]
    fn test_boundary_graph_no_boundary() {
        let mesh = create_mesh(
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            vec![1, 0, 2, 2, 0, 3, 3, 0, 1, 1, 2, 3],
        );

        let graph = mesh.boundary_graph();
        let expected = graph_from_mesh(&mesh, vec![], vec![]);
        assert!(algo::is_isomorphic_matching(
            &graph,
            &expected,
            |x, y| x == y,
            |x, y| x == y
        ));
    }

    #[test]
    fn test_combine_equal_vertices_none() {
        let graph = create_graph(
            vec![
                vec2(0.0, 0.0),
                vec2(0.5, 0.0),
                vec2(1.0, 0.0),
                vec2(1.5, 0.0),
            ],
            vec![(0, 1), (2, 3), (3, 2)],
        );

        let expected = graph.clone();
        let result = MaterialMesh::combine_equal_vertices(&graph);

        assert!(algo::is_isomorphic_matching(
            &result,
            &expected,
            |x, y| x == y,
            |x, y| x == y
        ));
    }

    #[test]
    fn test_combine_equal_vertices_one() {
        let graph = create_graph(
            vec![
                vec2(0.0, 0.0),
                vec2(0.5, 0.0),
                vec2(0.5, 0.0),
                vec2(1.0, 0.0),
            ],
            vec![(0, 1), (1, 2), (2, 3), (3, 2)],
        );

        let expected = create_graph(
            vec![vec2(0.0, 0.0), vec2(0.5, 0.0), vec2(1.0, 0.0)],
            vec![(0, 1), (1, 2), (2, 1)],
        );
        let result = MaterialMesh::combine_equal_vertices(&graph);

        assert!(algo::is_isomorphic_matching(
            &result,
            &expected,
            |x, y| x == y,
            |x, y| x == y
        ));
    }

    #[test]
    fn test_combine_equal_vertices_edge_collapse() {
        let graph = create_graph(
            vec![
                vec2(0.0, 0.0),
                vec2(0.5, 0.0),
                vec2(0.5, 0.0),
                vec2(0.0, 0.0),
            ],
            vec![(0, 1), (3, 2)],
        );

        let expected = create_graph(vec![vec2(0.0, 0.0), vec2(0.5, 0.0)], vec![(0, 1)]);
        let result = MaterialMesh::combine_equal_vertices(&graph);

        assert!(algo::is_isomorphic_matching(
            &result,
            &expected,
            |x, y| x == y,
            |x, y| x == y
        ));
    }

    #[test]
    fn test_boundary_square_intersect_edge_across() {
        let mut graph = create_graph(vec![vec2(-0.5, -0.25), vec2(0.5, -0.25)], vec![(0, 1)]);
        let expected = create_graph(
            vec![
                vec2(-0.5, -0.5),
                vec2(0.5, -0.5),
                vec2(-0.5, -0.25),
                vec2(0.5, -0.25),
            ],
            vec![(3, 1), (1, 0), (0, 2), (2, 3)],
        );

        MaterialMesh::intersect_center_unit_square_on_graph(&mut graph);
        assert!(algo::is_isomorphic_matching(
            &graph,
            &expected,
            |x, y| x == y,
            |x, y| x == y
        ));
    }

    #[test]
    fn test_boundary_square_intersect_multiple_edges() {
        let mut graph = create_graph(
            vec![
                vec2(-0.5, -0.25),
                vec2(-0.25, -0.5),
                vec2(0.25, -0.5),
                vec2(0.5, -0.25),
                vec2(0.5, 0.25),
                vec2(0.25, 0.5),
                vec2(-0.25, 0.5),
                vec2(-0.5, 0.25),
                vec2(0.0, -0.5),
                vec2(0.0, 0.5),
            ],
            vec![(0, 1), (3, 2), (5, 4), (6, 7), (8, 9)],
        );
        let expected = create_graph(
            vec![
                vec2(-0.5, -0.25),
                vec2(-0.25, -0.5),
                vec2(0.25, -0.5),
                vec2(0.5, -0.25),
                vec2(0.5, 0.25),
                vec2(0.25, 0.5),
                vec2(-0.25, 0.5),
                vec2(-0.5, 0.25),
                vec2(0.0, -0.5),
                vec2(0.0, 0.5),
                vec2(-0.5, -0.5),
                vec2(-0.5, 0.5),
            ],
            vec![
                (0, 1),
                (1, 10),
                (10, 0),
                (6, 7),
                (7, 11),
                (11, 6),
                (8, 9),
                (9, 5),
                (5, 4),
                (4, 3),
                (3, 2),
                (2, 8),
            ],
        );

        MaterialMesh::intersect_center_unit_square_on_graph(&mut graph);
        assert!(algo::is_isomorphic_matching(
            &graph,
            &expected,
            |x, y| x == y,
            |x, y| x == y
        ));
    }

    #[test]
    fn test_boundary_square_intersect_corner() {
        // Boundary hits a corner.
        let mut graph = create_graph(vec![vec2(-0.5, -0.5), vec2(0.5, 0.5)], vec![(0, 1)]);
        let expected = create_graph(
            vec![vec2(-0.5, -0.5), vec2(0.5, 0.5), vec2(0.5, -0.5)],
            vec![(0, 1), (1, 2), (2, 0)],
        );

        MaterialMesh::intersect_center_unit_square_on_graph(&mut graph);
        assert!(algo::is_isomorphic_matching(
            &graph,
            &expected,
            |x, y| x == y,
            |x, y| x == y
        ));
    }

    #[test]
    fn test_boundary_square_intersect_degree_2_inner() {
        // Boundary has a degree-2 vertex on square boundary. No outer square.
        let mut graph = create_graph(
            vec![
                vec2(-0.5, 0.0),
                vec2(0.0, -0.5),
                vec2(0.5, 0.0),
                vec2(0.0, 0.5),
            ],
            vec![(0, 3), (3, 2), (2, 1), (1, 0)],
        );
        let expected = graph.clone();

        MaterialMesh::intersect_center_unit_square_on_graph(&mut graph);
        assert!(algo::is_isomorphic_matching(
            &graph,
            &expected,
            |x, y| x == y,
            |x, y| x == y
        ));
    }

    #[test]
    fn test_boundary_square_intersect_degree_2_outer() {
        // Boundary has a degree-2 vertex on square boundary. Yes outer square.
        let mut graph = create_graph(
            vec![
                vec2(-0.5, 0.0),
                vec2(0.0, -0.5),
                vec2(0.5, 0.0),
                vec2(0.0, 0.5),
            ],
            vec![(0, 1), (1, 2), (2, 3), (3, 0)],
        );
        let expected = create_graph(
            vec![
                vec2(-0.5, 0.0),
                vec2(0.0, -0.5),
                vec2(0.5, 0.0),
                vec2(0.0, 0.5),
                vec2(-0.5, -0.5),
                vec2(0.5, -0.5),
                vec2(0.5, 0.5),
                vec2(-0.5, 0.5),
            ],
            vec![
                (0, 1),
                (1, 4),
                (4, 0),
                (1, 2),
                (2, 5),
                (5, 1),
                (2, 3),
                (3, 6),
                (6, 2),
                (3, 0),
                (0, 7),
                (7, 3),
            ],
        );

        MaterialMesh::intersect_center_unit_square_on_graph(&mut graph);
        assert!(algo::is_isomorphic_matching(
            &graph,
            &expected,
            |x, y| x == y,
            |x, y| x == y
        ));
    }

    #[test]
    fn test_boundary_square_intersect_degree_3() {
        // Boundary has a degree-3 vertex on square boundary.
        let mut graph = create_graph(
            vec![
                vec2(-0.5, 0.0),
                vec2(0.5, -0.5),
                vec2(0.5, 0.0),
                vec2(0.5, 0.5),
            ],
            vec![(0, 1), (2, 0), (0, 3)],
        );
        let expected = create_graph(
            vec![
                vec2(-0.5, 0.0),
                vec2(0.5, -0.5),
                vec2(0.5, 0.0),
                vec2(0.5, 0.5),
                vec2(-0.5, -0.5),
            ],
            vec![(0, 1), (1, 4), (4, 0), (0, 3), (3, 2), (2, 0)],
        );

        MaterialMesh::intersect_center_unit_square_on_graph(&mut graph);
        assert!(algo::is_isomorphic_matching(
            &graph,
            &expected,
            |x, y| x == y,
            |x, y| x == y
        ));
    }

    #[test]
    fn test_boundary_square_intersect_boundary_clockwise() {
        // Boundary is only on square boundary. Edges go clockwise.
        let mut graph = create_graph(
            vec![vec2(-0.5, 0.5), vec2(0.0, 0.5), vec2(0.5, 0.5)],
            vec![(0, 1), (1, 2)],
        );
        let expected = create_graph(
            vec![
                vec2(-0.5, 0.5),
                vec2(0.0, 0.5),
                vec2(0.5, 0.5),
                vec2(0.5, -0.5),
                vec2(-0.5, -0.5),
            ],
            vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)],
        );

        MaterialMesh::intersect_center_unit_square_on_graph(&mut graph);
        assert!(algo::is_isomorphic_matching(
            &graph,
            &expected,
            |x, y| x == y,
            |x, y| x == y
        ));
    }

    #[test]
    fn test_boundary_square_intersect_boundary_counterclockwise() {
        // Boundary is only on square boundary. Edges go counterclockwise.
        let mut graph = create_graph(
            vec![vec2(-0.5, 0.5), vec2(0.0, 0.5), vec2(0.5, 0.5)],
            vec![(2, 1), (1, 0)],
        );
        let expected = create_graph(vec![], vec![]);

        MaterialMesh::intersect_center_unit_square_on_graph(&mut graph);
        assert!(algo::is_isomorphic_matching(
            &graph,
            &expected,
            |x, y| x == y,
            |x, y| x == y
        ));
    }

    #[test]
    fn test_boundary_square_intersect_hole_clockwise() {
        // Boundary is completely inside square. Hole goes clockwise.
        let mut graph = create_graph(
            vec![vec2(-0.25, -0.25), vec2(-0.25, 0.25), vec2(0.25, 0.25)],
            vec![(0, 1), (1, 2), (2, 0)],
        );
        let expected = graph.clone();

        MaterialMesh::intersect_center_unit_square_on_graph(&mut graph);
        assert!(algo::is_isomorphic_matching(
            &graph,
            &expected,
            |x, y| x == y,
            |x, y| x == y
        ));
    }

    #[test]
    fn test_boundary_square_intersect_hole_counterclockwise() {
        // Boundary is completely inside square. Hole goes counterclockwise.
        let mut graph = create_graph(
            vec![vec2(-0.25, -0.25), vec2(-0.25, 0.25), vec2(0.25, 0.25)],
            vec![(0, 2), (2, 1), (1, 0)],
        );
        let expected = create_graph(
            vec![
                vec2(-0.25, -0.25),
                vec2(-0.25, 0.25),
                vec2(0.25, 0.25),
                vec2(-0.5, -0.5),
                vec2(0.5, -0.5),
                vec2(0.5, 0.5),
                vec2(-0.5, 0.5),
            ],
            vec![(0, 2), (2, 1), (1, 0), (3, 6), (6, 5), (5, 4), (4, 3)],
        );

        MaterialMesh::intersect_center_unit_square_on_graph(&mut graph);
        assert!(algo::is_isomorphic_matching(
            &graph,
            &expected,
            |x, y| x == y,
            |x, y| x == y
        ));
    }

    #[test]
    fn test_boundary_square_intersect_nothing() {
        let mut graph = create_graph(vec![], vec![]);
        let expected = graph.clone();

        assert!(!MaterialMesh::intersect_center_unit_square_on_graph(
            &mut graph
        ));
        assert!(algo::is_isomorphic_matching(
            &graph,
            &expected,
            |x, y| x == y,
            |x, y| x == y
        ));
    }

    #[test]
    fn test_boundary_square_context_tet_outside() {
        let mesh = create_mesh(
            vec![
                -0.25, -0.25, -0.75, 0.25, 0.25, -0.75, 0.25, -0.25, -0.25, -0.25, 0.25, -0.25,
            ],
            vec![0, 1, 2, 2, 3, 0, 1, 0, 3, 3, 2, 1],
        );
        let mut graph = Graph::new();
        let expected = graph.clone();

        mesh.intersect_center_unit_square_with_context(&mut graph);
        assert!(algo::is_isomorphic_matching(
            &graph,
            &expected,
            |x, y| x == y,
            |x, y| x == y
        ));
    }

    #[test]
    fn test_boundary_square_context_tet_inside() {
        let mesh = create_mesh(
            vec![
                -0.25, -0.25, -0.75, 0.25, 0.25, -0.75, 0.25, -0.25, -0.25, -0.25, 0.25, -0.25,
            ],
            vec![0, 2, 1, 2, 0, 3, 1, 3, 0, 3, 1, 2],
        );
        let mut graph = Graph::new();
        let expected = create_graph(
            vec![
                vec2(-0.5, -0.5),
                vec2(0.5, -0.5),
                vec2(0.5, 0.5),
                vec2(-0.5, 0.5),
            ],
            vec![(0, 3), (3, 2), (2, 1), (1, 0)],
        );

        mesh.intersect_center_unit_square_with_context(&mut graph);
        assert!(algo::is_isomorphic_matching(
            &graph,
            &expected,
            |x, y| x == y,
            |x, y| x == y
        ));
    }
}
