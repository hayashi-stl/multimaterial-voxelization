use fnv::FnvHashMap;
use tri_mesh::prelude::*;
use std::path::Path;

use crate::material_mesh::{Axis, MaterialID, MaterialMesh};

pub type Vec3i = Vector3<i32>;
pub type Vec3f = Vector3<f32>;

#[derive(Debug)]
struct DebugMeshBuilder {
    positions: Vec<f64>,
    indexes: Vec<u32>,
    materials: Vec<MaterialID>,
}

impl DebugMeshBuilder {
    fn new() -> Self {
        Self {
            positions: vec![],
            indexes: vec![],
            materials: vec![],
        }
    }

    /// Add a cube given min coordinates, side length, and material
    fn add_cube(&mut self, min: Vec3, length: f64, material: MaterialID) {
        let offset = self.positions.len() / 3;
        self.indexes.extend(vec![
            2, 6, 4, 4, 0, 2, 6, 7, 5, 5, 4, 6, 7, 3, 1, 1, 5, 7,
            3, 2, 0, 0, 1, 3, 0, 4, 5, 5, 1, 0, 2, 3, 7, 7, 6, 2,
        ].into_iter().map(|i| offset as u32 + i));

        for z in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    self.positions.extend(vec![
                        min.x + length * x as f64,
                        min.y + length * y as f64,
                        min.z + length * z as f64,
                    ]);
                }
            }
        }

        self.materials.extend(vec![material; 12]);
    }

    fn build(self) -> MaterialMesh {
        MaterialMesh::new(
            MeshBuilder::new()
                .with_positions(self.positions)
                .with_indices(self.indexes)
                .with_tags(self.materials)
                .build()
                .expect("Invalid mesh")
        )
    }
}

/// The voxelization. Includes chunks, pure voxels, and complex voxels.
#[derive(Clone, Debug, Default)]
pub struct Voxels {
    chunks: FnvHashMap<Vec3i, Chunk>,
}

impl Voxels {
    fn new() -> Self {
        Self::default()
    }

    /// Export this voxelization as an obj for debugging
    pub fn export_debug_obj<P: AsRef<Path>>(&self, path: P) {
        let mut builder = DebugMeshBuilder::new();
        
        for (pos, chunk) in &self.chunks {
            match chunk {
                Chunk::Uniform(material) =>
                    builder.add_cube(pos.cast().unwrap() * Chunk::SIZE as f64, Chunk::SIZE as f64, *material)
            }
        }

        builder.build().export_debug_obj(path)
    }

    fn fill_uniform_chunks(&mut self, ranges_yz: Vec<(f64, f64, Vec<(f64, f64, i32)>)>) {
        for (y, z, ranges) in &ranges_yz {
            let chunk_y = *y as i32 / Chunk::SIZE as i32;
            let chunk_z = *z as i32 / Chunk::SIZE as i32;
            let mut inside = 0;
            let mut start = 0;

            for (min, max, grad) in ranges {
                if inside != 0 {
                    let end = (*min / Chunk::SIZE as f64).floor() as i32;

                    for chunk_x in start..end {
                        self.chunks.insert(
                            vec3(chunk_x, chunk_y, chunk_z),
                            // Placeholder until multimaterial works
                            Chunk::Uniform(MaterialID::new(1)),
                        );
                    }
                }

                inside = (inside + grad).rem_euclid(2);
                start = (*max / Chunk::SIZE as f64).ceil() as i32;
            }
        }
    }
}

impl From<MaterialMesh> for Voxels {
    fn from(mesh: MaterialMesh) -> Self {
        let slices = mesh.axis_slice(Axis::Z, Chunk::SIZE as f64);
        let slices = slices
            .into_iter()
            .flat_map(|(z, slice)| {
                slice
                    .axis_slice(Axis::Y, Chunk::SIZE as f64)
                    .into_iter()
                    .map(move |(y, slice)| (y, z, slice))
            })
            .collect::<Vec<_>>();

        // Obtain ranges as a map from (y, z) coords to a vector of (min, max, in-out gradient) tuples
        let ranges_yz = slices
            .into_iter()
            .map(|(y, z, slice)| {
                (
                    y,
                    z,
                    slice.axis_ranges_and_in_out_gradients(
                        Axis::X,
                        (Chunk::SIZE * Chunk::SIZE) as f64,
                    ),
                )
            })
            .collect::<Vec<_>>();

        let mut voxels = Self::new();

        voxels.fill_uniform_chunks(ranges_yz);

        voxels
    }
}

/// A chunk. Can be uniform or complex
#[derive(Clone, Debug)]
pub enum Chunk {
    Uniform(MaterialID),
}

impl Chunk {
    pub const SIZE: usize = 16;
}

/// A complex chunk. Contains a grid of voxels.
#[derive(Clone, Debug)]
pub struct ComplexChunk {
    voxels: [Voxel; Chunk::SIZE * Chunk::SIZE * Chunk::SIZE],
    complex: Vec<ComplexVoxel>,
}

/// A complex chunk entry. Can be a pure voxel or an index to a complex voxel
#[derive(Copy, Clone, Debug)]
pub enum Voxel {
    Pure(MaterialID),
    Complex(u32),
}

/// A complex voxel, including inner vertices and hulls.
/// Each hull contains indexes to points.
/// Inner vertices have indexes 8 and greater.
/// Corner point (x, y, z) where x, y, z ∈ {0, 1} has index x + 2y + 4z
#[derive(Clone, Debug)]
pub struct ComplexVoxel {
    /// Vertices that are not corners of the cube.
    inner_vertices: Vec<Vec3f>,
    hulls: Vec<([u32; Self::MAX_HULL_SIZE], MaterialID)>,
}

impl ComplexVoxel {
    pub const MAX_HULL_SIZE: usize = 8;
}
