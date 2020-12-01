use fnv::FnvHashMap;
use rayon::prelude::*;
use std::path::Path;
use tri_mesh::prelude::*;

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
        self.indexes.extend(
            vec![
                2, 6, 4, 4, 0, 2, 6, 7, 5, 5, 4, 6, 7, 3, 1, 1, 5, 7, 3, 2, 0, 0, 1, 3, 0, 4, 5, 5,
                1, 0, 2, 3, 7, 7, 6, 2,
            ]
            .into_iter()
            .map(|i| offset as u32 + i),
        );

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
                .expect("Invalid mesh"),
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

        for (chunk_pos, chunk) in &self.chunks {
            match chunk {
                Chunk::Uniform(material) => builder.add_cube(
                    chunk_pos.cast().unwrap() * Chunk::SIZE as f64,
                    Chunk::SIZE as f64,
                    *material,
                ),

                Chunk::Complex(chunk) => chunk.add_to_debug_mesh(*chunk_pos, &mut builder),
            }
        }

        let mut output = String::from("o object\n");

        let positions = &builder.positions;
        let len = positions.len() / 3;
        for i in 0..builder.positions.len() / 3 {
            if (i + 1) * 100 / len > i * 100 / len {
                //println!("Vertices: {}%", (i + 1) * 100 / len);
            }
            output += &format!(
                "v {} {} {}\n",
                positions[i * 3],
                positions[i * 3 + 1],
                positions[i * 3 + 2]
            );
        }

        let indexes = &builder.indexes;
        let len = indexes.len() / 3;
        for i in 0..builder.indexes.len() / 3 {
            if (i + 1) * 100 / len > i * 100 / len {
                //println!("Faces: {}%", (i + 1) * 100 / len);
            }

            let mut face = String::new();
            for j in 0..3 {
                let index = indexes[i * 3 + j] + 1;
                face += &format!(" {}", index);
            }
            output += &format!("f{}\n", face);
        }

        std::fs::write(path, output).expect("Could not debug obj");

        //builder.build().export_debug_obj(path)
    }

    fn fill_uniform_chunks(&mut self, ranges_yz: Vec<(f64, f64, Vec<(f64, f64, i32)>)>) {
        for (y, z, ranges) in &ranges_yz {
            let chunk_y = (*y as i32).div_euclid(Chunk::SIZE as i32);
            let chunk_z = (*z as i32).div_euclid(Chunk::SIZE as i32);
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

    fn add_complex_voxels_to_chunks(
        &self,
        slices: Vec<(Vec3, MaterialMesh)>,
        chunks: FnvHashMap<Vec3i, ComplexChunk>,
    ) -> FnvHashMap<Vec3i, ComplexChunk> {
        for (pos, slice) in slices {
            let mesh = slice.intersect_unit_cube(pos);
            mesh.export_debug_obj(format!(
                "assets/debug/complex/cube_{:04}_{:04}_{:04}.obj",
                pos.x, pos.y, pos.z
            ));
        }

        chunks
    }

    fn ranges_to_complex_chunks(
        &self,
        ranges_yz: Vec<(f64, f64, Vec<(f64, f64, i32)>)>,
    ) -> FnvHashMap<Vec3i, ComplexChunk> {
        let mut chunks: FnvHashMap<Vec3i, ComplexChunk> = FnvHashMap::default();
        let mut dummy = ComplexChunk::new();
        let chunk_size = Chunk::SIZE as i32;

        for (y, z, ranges) in &ranges_yz {
            let chunk_y = (*y as i32).div_euclid(chunk_size);
            let chunk_z = (*z as i32).div_euclid(chunk_size);
            let mod_y = (*y as i32).rem_euclid(chunk_size);
            let mod_z = (*z as i32).rem_euclid(chunk_size);

            let mut inside = 0;
            let mut start = 0i32;

            let mut chunk = &mut dummy;

            for (min, max, grad) in ranges {
                if inside != 0 {
                    let end = min.floor() as i32;

                    let mut x = start;

                    while x < end {
                        // Chunk change!
                        if x == start || x.rem_euclid(chunk_size) == 0 {
                            // Skip over uniform chunks
                            if self.chunks.contains_key(&vec3(
                                x.div_euclid(chunk_size),
                                chunk_y,
                                chunk_z,
                            )) {
                                x = (x + chunk_size).div_euclid(chunk_size) * chunk_size;
                                continue;
                            }

                            chunk = chunks
                                .entry(vec3(x.div_euclid(chunk_size), chunk_y, chunk_z))
                                .or_insert(ComplexChunk::new());
                        }

                        *chunk.voxel_mut(vec3(x.rem_euclid(chunk_size), mod_y, mod_z)) =
                            Voxel::Pure(Some(MaterialID::new(1)));
                        x += 1;
                    }
                }

                inside = (inside + grad).rem_euclid(2);
                start = max.ceil() as i32;
            }
        }

        chunks
    }
}

impl From<MaterialMesh> for Voxels {
    fn from(mesh: MaterialMesh) -> Self {
        // Uniform chunks
        let slices = mesh.axis_slice(Axis::Z, Chunk::SIZE as f64);
        let mut slices: Vec<(f64, f64, MaterialMesh)> = slices
            .into_par_iter()
            .flat_map_iter(|(z, slice)| {
                slice
                    .axis_slice(Axis::Y, Chunk::SIZE as f64)
                    .into_iter()
                    .map(move |(y, slice)| (y, z, slice))
            })
            .collect::<Vec<_>>();

        // Obtain ranges as a map from (y, z) coords to a vector of (min, max, in-out gradient) tuples
        let ranges_yz = slices
            // Using mutable reference only because MaterialMesh is not Sync
            .par_iter_mut()
            .map(|(y, z, slice)| {
                slice.align_with_slice_planes(Axis::X, Chunk::SIZE as f64);
                (
                    *y,
                    *z,
                    slice.axis_ranges_and_in_out_gradients(
                        Axis::X,
                        (Chunk::SIZE * Chunk::SIZE) as f64,
                    ),
                )
            })
            .collect::<Vec<_>>();

        let mut voxels = Self::new();

        voxels.fill_uniform_chunks(ranges_yz);

        // Pure/complex voxels
        let complex_chunks = slices
            .into_par_iter()
            .flat_map_iter(|(chunk_y, chunk_z, slice)| {
                let mut slices: Vec<(f64, f64, MaterialMesh)> = slice
                    .axis_slice(Axis::Z, 1.0)
                    .into_iter()
                    .flat_map(|(z, slice)| {
                        slice
                            .axis_slice(Axis::Y, 1.0)
                            .into_iter()
                            .map(move |(y, slice)| (y, z, slice))
                    })
                    .collect::<Vec<_>>();

                let ranges_yz = slices
                    // Using mutable reference only because MaterialMesh is not Sync
                    .iter_mut()
                    .map(|(y, z, slice)| {
                        slice.align_with_slice_planes(Axis::X, 1.0);
                        (*y, *z, slice.axis_ranges_and_in_out_gradients(Axis::X, 1.0))
                    })
                    .collect::<Vec<_>>();

                let chunks = voxels.ranges_to_complex_chunks(ranges_yz);

                let slices = slices
                    .into_iter()
                    .flat_map(|(y, z, slice)| {
                        slice
                            .axis_slice(Axis::X, 1.0)
                            .into_iter()
                            .map(move |(x, slice)| (vec3(x, y, z), slice))
                    })
                    .collect::<Vec<_>>();

                let chunks = voxels.add_complex_voxels_to_chunks(slices, chunks);

                chunks.into_iter().map(|(k, v)| (k, Chunk::Complex(v)))
            })
            .collect::<Vec<_>>();

        voxels.chunks.extend(complex_chunks.into_iter());

        //let objs = std::fs::read_dir("assets/debug/complex")
        //    .unwrap()
        //    .flat_map(|entry| {
        //        let path = entry.unwrap().path();
        //        if path.is_file() {
        //            Some(std::fs::read_to_string(path).unwrap())
        //        } else {
        //            None
        //        }
        //    }).collect::<Vec<_>>();
        //let len = objs.len();
        //let mesh = objs
        //    .into_iter()
        //    .enumerate()
        //    .fold(MeshBuilder::<MaterialID>::new().with_positions(vec![]).build().unwrap(), |mut mesh, (i, obj)| {
        //        let m = MeshBuilder::new().with_obj(obj).build().unwrap();
        //        mesh.append(&m);
        //        println!("{}/{}", i, len);
        //        mesh
        //    });
        //MaterialMesh::new(mesh).export_debug_obj("assets/debug/complex_voxels_2.obj");

        voxels
    }
}

/// A chunk. Can be uniform or complex
#[derive(Clone, Debug)]
pub enum Chunk {
    Uniform(MaterialID),
    Complex(ComplexChunk),
}

impl Chunk {
    pub const SIZE: usize = 16;

    fn is_uniform(&self) -> bool {
        if let Chunk::Uniform(_) = self {
            true
        } else {
            false
        }
    }
}

/// A complex chunk. Contains a grid of voxels.
#[derive(Clone, Debug)]
pub struct ComplexChunk {
    voxels: [Voxel; Chunk::SIZE * Chunk::SIZE * Chunk::SIZE],
    complex: Vec<ComplexVoxel>,
}

impl ComplexChunk {
    fn new() -> Self {
        Self {
            voxels: [Voxel::Pure(None); Chunk::SIZE * Chunk::SIZE * Chunk::SIZE],
            complex: vec![],
        }
    }

    fn add_to_debug_mesh(&self, chunk_pos: Vec3i, builder: &mut DebugMeshBuilder) {
        for z in 0..Chunk::SIZE as i32 {
            for y in 0..Chunk::SIZE as i32 {
                for x in 0..Chunk::SIZE as i32 {
                    if let Voxel::Pure(Some(mat)) = self.voxel(vec3(x, y, z)) {
                        // Cheaply remove landlocked voxels
                        if vec![
                            vec3(1, 0, 0),
                            vec3(-1, 0, 0),
                            vec3(0, 1, 0),
                            vec3(0, -1, 0),
                            vec3(0, 0, 1),
                            vec3(0, 0, -1),
                        ]
                        .into_iter()
                        .any(|vec| {
                            let vec = vec + vec3(x, y, z);
                            vec.x < 0
                                || vec.x >= Chunk::SIZE as i32
                                || vec.y < 0
                                || vec.y >= Chunk::SIZE as i32
                                || vec.z < 0
                                || vec.z >= Chunk::SIZE as i32
                                || self.voxel(vec).is_empty()
                        }) {
                            builder.add_cube(
                                (chunk_pos * Chunk::SIZE as i32 + vec3(x, y, z))
                                    .cast()
                                    .unwrap(),
                                1.0,
                                mat,
                            );
                        }
                    }
                }
            }
        }
    }

    fn offset_to_index(offset: Vec3i) -> usize {
        (offset.z as usize * Chunk::SIZE + offset.y as usize) * Chunk::SIZE + offset.x as usize
    }

    /// Get the voxel at a certain offset in the chunk
    pub fn voxel(&self, offset: Vec3i) -> Voxel {
        self.voxels[Self::offset_to_index(offset)]
    }

    /// Get a mutable reference to the voxel at a certain offset in the chunk
    pub fn voxel_mut(&mut self, offset: Vec3i) -> &mut Voxel {
        &mut self.voxels[Self::offset_to_index(offset)]
    }
}

/// A complex chunk entry. Can be a pure voxel or an index to a complex voxel
#[derive(Copy, Clone, Debug)]
pub enum Voxel {
    Pure(Option<MaterialID>),
    Complex(u32),
}

impl Voxel {
    /// Checks if a voxel is empty
    pub fn is_empty(self) -> bool {
        match self {
            Voxel::Pure(optional) => optional.is_none(),
            Voxel::Complex(_) => false,
        }
    }
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
