use fnv::FnvHashMap;
use std::num::NonZeroU32;
use tri_mesh::mesh_builder;
use tri_mesh::prelude::*;
use std::path::Path;
use std::fs;
use float_ord::FloatOrd;

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

impl MaterialMesh {
    fn new(mesh: Mesh<MaterialID>) -> Self {
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

    /// Assumes the edge points in the +axis direction
    /// Returns half-edges that may need to be split.
    /// The first one returned, if any, is the new half-edge resulting
    /// from the split. The others are triangulation edges.
    fn split_edge(&mut self, axis: Axis, slice_coord: f64, range: EdgeRange) -> Vec<HalfEdgeID> {
        let EdgeRange { min, max, halfedge_id , triangulation } = range;
        let axis_id = axis as usize;

        let pos = self.mesh.edge_positions(halfedge_id);
        // We assume the edge points up, as that was enforced
        let t = (slice_coord - min) / (max - min);
        let mut inter = vec3(slice_coord,
            pos.0[(axis_id + 1) % 3] * (1.0 - t) + pos.1[(axis_id + 1) % 3] * t,
            pos.0[(axis_id + 2) % 3] * (1.0 - t) + pos.1[(axis_id + 2) % 3] * t,
        );

        // Rotate vector properly
        for _ in 0..axis_id {
            inter = vec3(inter.z, inter.x, inter.y);
        }

        let (vertex_id, split_halfedge_id) = self.mesh.split_edge(halfedge_id, inter);
        let mut vec = self.mesh.vertex_halfedge_iter(vertex_id)
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
    pub fn axis_contour(mut self, axis: Axis, spacing: f64) -> MaterialMesh {
        let axis_id = axis as usize;
        let extreme = self.mesh.extreme_coordinates();
        let min = (extreme.0[axis_id] / spacing + 0.5).floor() * spacing - spacing;
        let max = (extreme.1[axis_id] / spacing + 0.5).floor() * spacing + spacing;

        // Obtain extreme coordinates of edges
        let mut ranges = self.mesh.edge_iter().map(|mut halfedge_id| {
            let mut pos = self.mesh.edge_positions(halfedge_id);

            // We want the edge to point in the +axis direction
            if pos.0[axis_id] > pos.1[axis_id] {
                pos = (pos.1, pos.0);
                halfedge_id = self.mesh.walker_from_halfedge(halfedge_id).as_twin().halfedge_id().expect("Half-edge doesn't have a twin");
            } 

            EdgeRange {
                min: pos.0[axis_id],
                max: pos.1[axis_id],
                halfedge_id,
                triangulation: false,
            }
        }).collect::<Vec<_>>();

        ranges.sort_by_key(|range| FloatOrd(range.min));
        ranges.reverse();

        // Edges that are currently being contoured
        let mut edges = vec![];
        // Triangulation edges that are currently being contoured
        let mut tri_edges = vec![];

        let mut slice_coord = min + spacing;

        // We start contouring bottom edges and move up
        while !ranges.is_empty() {
            while ranges.last().map_or(false, |r| r.min < slice_coord) {
                if let Some(range) = ranges.pop() {
                    if range.max > slice_coord {
                        if range.triangulation {
                            &mut tri_edges
                        } else {
                            &mut edges
                        }.push(range)
                    }
                }
            }

            // Split the edges
            for range in edges.drain(..) {
                for (i, new_halfedge_id) in self.split_edge(axis, slice_coord, range).into_iter().enumerate() {
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
                self.mesh.flip_edge(range.halfedge_id).expect("Could not flip triangulation edge");
            }

            slice_coord += spacing;
        }

        self
    }

    /// Slices the mesh along evenly spaced axis-aligned planes.
    /// One of the planes crosses the origin.
    /// Slices are ordered from minimium coordinate to maximum coordinate.
    /// Does not use the mesh's split_at_intersection() method because
    /// the runtime can be faster in this case.
    pub fn axis_slice(mut self, axis: Axis, spacing: f64) -> Vec<MaterialMesh> {
        self = self.axis_contour(axis, spacing);
        self.export_debug_obj("assets/debug/contour.obj");
        todo!()
    }

    ///// Slices the mesh along evenly spaced axis-aligned planes.
    ///// One of the planes crosses the origin.
    ///// Slices are ordered from minimium coordinate to maximum coordinate.
    ///// Does not use the mesh's split_at_intersection() method because
    ///// the runtime can be faster in this case.
    //pub fn axis_slice(&self, axis: Axis, spacing: f64) -> Vec<MaterialMesh> {
    //    let (coarse, slice_min) = self.coarse_slice(axis, spacing);

    //    let extreme = self.mesh.extreme_coordinates();
    //    let plane_mins = [extreme.0[(axis as usize + 1) % 3], extreme.0[(axis as usize + 2) % 3]];
    //    let plane_maxs = [extreme.1[(axis as usize + 1) % 3], extreme.1[(axis as usize + 2) % 3]];

    //    // Rotates the axes so the X axis moves to the Y axis
    //    let axis_rotation = Mat3::from_cols(
    //        vec3(0.0, 1.0, 0.0),
    //        vec3(0.0, 0.0, 1.0),
    //        vec3(1.0, 0.0, 0.0)
    //    );
    //    
    //    let mut positions = Mat3::from_cols(
    //        vec3(slice_min, plane_mins[0] - 1.0, plane_mins[1] - 1.0),
    //        vec3(slice_min, 2.0 * plane_maxs[0] - plane_mins[0] + 1.0, plane_mins[1] - 1.0),
    //        vec3(slice_min, plane_mins[0] - 1.0, 2.0 * plane_maxs[1] - plane_mins[1] + 1.0),
    //    );

    //    let mut spacing_vec = Vec3::unit_x() * spacing;

    //    for _ in 0..axis as usize {
    //        positions = axis_rotation * positions;
    //        spacing_vec = axis_rotation * spacing_vec;
    //    }

    //    // Construct slicing planes
    //    let mut planes = MeshBuilder::<()>::new().with_positions(
    //        AsRef::<[f64; 9]>::as_ref(&positions).iter().copied().collect()
    //    ).with_indices(vec![0, 1, 2]).build().expect("Cannot build slice plane");

    //    planes.append(&planes.translated(spacing_vec));
    //    println!("{}", planes);
    //    let mut slice_coord = slice_min;
    //    let mut slices = vec![];

    //    // Slicing time!
    //    for mut mesh in coarse {
    //        let (split, _) = mesh.mesh.split_at_intersection(&mut planes.clone());
    //        
    //        // Combine parts between the slices into 1 mesh
    //        let mut slice = MeshBuilder::<MaterialID>::new().with_positions(vec![]).build().unwrap();
    //        for piece in split {
    //            let ext = piece.extreme_coordinates();
    //            let average = (ext.0[axis as usize] + ext.1[axis as usize]) / 2.0;
    //            if average > slice_coord && average < slice_coord + spacing {
    //                println!("Accepted: Vertices: {}, Faces; {}", piece.num_vertices(), piece.num_faces());
    //                slice.append(&piece);
    //            } else {
    //                println!("Rejected: Vertices: {}, Faces; {}", piece.num_vertices(), piece.num_faces());
    //            }
    //        }
    //        println!("Vertices: {}, Faces; {}", slice.num_vertices(), slice.num_faces());
    //        println!();

    //        slices.push(MaterialMesh::new(slice));

    //        planes.translate(spacing_vec);
    //        slice_coord += spacing;
    //    }

    //    slices
    //}
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Axis {
    X,
    Y,
    Z,
}
