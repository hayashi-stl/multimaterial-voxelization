use fnv::FnvHashMap;
use std::num::NonZeroU32;
use tri_mesh::mesh_builder;
use tri_mesh::prelude::*;
use std::path::Path;
use std::fs;

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

impl MaterialMesh {
    fn new(mesh: Mesh<MaterialID>) -> Self {
        Self { mesh }
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

        println!("{}", mesh);

        Ok(Self { mesh })
    }

    /// Coarsely slices the mesh into regions to be sliced with a plane later.
    /// The minimum-coordinate slicing plane is also returned.
    fn coarse_slice(&self, axis: Axis, spacing: f64) -> (Vec<MaterialMesh>, f64) {
        let axis_id = axis as usize;

        let extreme = self.mesh.extreme_coordinates();
        let min = (extreme.0[axis_id] / spacing + 0.5).floor() * spacing - spacing;
        let max = (extreme.1[axis_id] / spacing + 0.5).floor() * spacing + spacing;

        let mut imms = vec![Intermediate::default(); ((max - min) / spacing) as usize];

        for face_id in self.mesh.face_iter() {
            // Find range of slices the face is in
            let vertices = self.mesh.face_vertices(face_id);
            let positions = (
                self.mesh.vertex_position(vertices.0),
                self.mesh.vertex_position(vertices.1),
                self.mesh.vertex_position(vertices.2),
            );

            let v_min = positions.0[axis_id]
                .min(positions.1[axis_id])
                .min(positions.2[axis_id]);
            let v_max = positions.0[axis_id]
                .max(positions.1[axis_id])
                .max(positions.2[axis_id]);
            let begin = ((v_min - min) / spacing).floor() as usize;
            let end = ((v_max - min) / spacing).ceil() as usize;

            // Insert face into the appropriate slices
            let tag = self.mesh.face_tag(face_id);
            for i in begin..end {
                let imm = &mut imms[i];

                for vertex in vec![vertices.0, vertices.1, vertices.2] {
                    let len = imm.vertex_ids.len();
                    let index = *imm.vertex_ids.entry(vertex).or_insert(len);
                    imm.indexes.push(index as u32);
                }
                imm.tags.push(tag);
            }
        }

        (
            imms.into_iter()
                .map(|imm| {
                    let mut positions = vec![0.0; imm.vertex_ids.len() * 3];
                    for (vertex, index) in imm.vertex_ids {
                        let position = self.mesh.vertex_position(vertex);
                        positions[index * 3 + 0] = position[0];
                        positions[index * 3 + 1] = position[1];
                        positions[index * 3 + 2] = position[2];
                    }

                    MaterialMesh::new(
                        MeshBuilder::new()
                            .with_positions(positions)
                            .with_indices(imm.indexes)
                            .with_tags(imm.tags)
                            .build()
                            .expect("Invalid mesh"),
                    )
                })
                .collect(),
            min,
        )
    }

    /// Slices the mesh along evenly spaced axis-aligned planes.
    /// One of the planes crosses the origin.
    /// Slices are ordered from minimium coordinate to maximum coordinate.
    /// Does not use the mesh's split_at_intersection() method because
    /// the runtime can be faster in this case.
    pub fn axis_slice(&self, axis: Axis, spacing: f64) -> Vec<MaterialMesh> {
        let coarse = self.coarse_slice(axis, spacing);

        for (i, slice) in coarse.0.iter().enumerate() {
            slice.export_debug_obj(format!("assets/debug/{:03}.obj", i));
        }

        todo!()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Axis {
    X,
    Y,
    Z,
}
