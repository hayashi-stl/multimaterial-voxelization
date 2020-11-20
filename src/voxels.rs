use cgmath::prelude::*;
use cgmath::Vector3;
use fnv::FnvHashMap;

use crate::material_mesh::{Axis, MaterialID, MaterialMesh};

pub type Vec3i = Vector3<i32>;
pub type Vec3f = Vector3<f32>;

/// The voxelization. Includes chunks, pure voxels, and complex voxels.
#[derive(Clone, Debug, Default)]
pub struct Voxels {
    chunks: FnvHashMap<Vec3i, Chunk>,
}

impl Voxels {
    fn new() -> Self {
        Self::default()
    }
}

impl From<MaterialMesh> for Voxels {
    fn from(mesh: MaterialMesh) -> Self {
        let (slices, min_z) = mesh.axis_slice(Axis::Z, Chunk::SIZE as f64);

        for (i, slice) in slices.iter().enumerate() {
            slice.export_debug_obj(format!("assets/debug/slice_{:03}.obj", i));
        }
        Self::default()
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
