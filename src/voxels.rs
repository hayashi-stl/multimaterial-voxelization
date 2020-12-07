use fnv::{FnvHashMap, FnvHashSet};
use float_ord::FloatOrd;
use rayon::prelude::*;
use std::path::Path;
use tri_mesh::prelude::*;
use combination::combine;
use bvh::bvh::BVH;
use bvh::ray::Ray;
use bvh::nalgebra::{Point3 as NPoint3, Vector3 as NVec3};

use crate::material_mesh::{Axis, MaterialID, MaterialMesh, BvhTriangle};
use crate::plc::PiecewiseLinearComplex;
use crate::util::HashVec3;

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

    fn add_mesh(&mut self, positions: Vec<Vec3>, triangles: Vec<([usize; 3], MaterialID)>) {
        let offset = self.positions.len() / 3;

        for ([v0, v1, v2], material) in triangles {
            self.indexes.extend(vec![v0, v1, v2].into_iter().map(|v| (v + offset) as u32));
            self.materials.push(material);
        }

        self.positions.extend(positions.into_iter().flat_map(|p| vec![p.x, p.y, p.z].into_iter()));
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

/// Approximation to closest material. Checks only in orthodirections.
fn closest_material(bvh: &BVH, triangles: &[BvhTriangle], point: Vec3) -> MaterialID {
    vec![Vec3::unit_x(), Vec3::unit_y(), Vec3::unit_z(), -Vec3::unit_x(), -Vec3::unit_y(), -Vec3::unit_z()]
        .into_iter()
        .flat_map(|dir| {
            let ray = Ray::new(
                NPoint3::new(point.x as f32, point.y as f32, point.z as f32),
                NVec3::new(dir.x as f32, dir.y as f32, dir.z as f32),
            );

            bvh.traverse(&ray, triangles)
                .into_iter()
                .map(|tri| (
                    tri.material(),
                    tri.intersection_time(point, dir)
                ))
                .filter_map(|(mat, t)| Some((mat, t?)))
                .min_by_key(|(_, t)| FloatOrd(*t))
        })
        .min_by_key(|(_, t)| FloatOrd(*t))
        .unwrap_or_default()
        .0
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
    pub fn export_debug_obj<P: AsRef<Path> + Clone>(&self, path: P) {
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

        let mut indexes = builder.indexes.chunks_exact(3).enumerate().collect::<Vec<_>>();
        indexes.sort_by_cached_key(|(i, _)| builder.materials[*i].0.get() - 1);
        let indexes = indexes.into_iter().flat_map(|(_, i)| vec![i[0], i[1], i[2]]).collect::<Vec<_>>();
        let len = indexes.len() / 3;

        let mut sorted_mats = builder.materials.clone();
        sorted_mats.sort();

        let mut materials = vec![];
        for i in 0..indexes.len() / 3 {
            if (i + 1) * 100 / len > i * 100 / len {
                //println!("Faces: {}%", (i + 1) * 100 / len);
            }

            if i == 0 || sorted_mats[i] != sorted_mats[i - 1] {
                let mat = sorted_mats[i].0.get() - 1;
                output += &format!("usemtl mat{}\n", mat);
                materials.push(mat);
            }

            let mut face = String::new();
            for j in 0..3 {
                let index = indexes[i * 3 + j] + 1;
                face += &format!(" {}", index);
            }
            output += &format!("f{}\n", face);
        }

        let mut mtl = String::new();

        let wheel = vec![vec3(1.0, 0.0, 0.0), vec3(1.0, 1.0, 0.0), vec3(0.0, 1.0, 0.0),
            vec3(0.0, 1.0, 1.0), vec3(0.0, 0.0, 1.0), vec3(1.0, 0.0, 1.0), vec3(1.0, 0.0, 0.0)];
        
        let len = materials.len();
        for (i, mat) in materials.into_iter().enumerate() {
            mtl += &format!("newmtl mat{}\n", mat);
            
            let hue = i as f64 / len as f64;
            let index = (6.0 * hue).floor() as usize;
            let frac = 6.0 * hue - index as f64;
            let color = wheel[index].lerp(wheel[index + 1], frac);
            mtl += &format!("Kd {} {} {}", color.x, color.y, color.z);
            mtl += "\n";
        }

        std::fs::write(path.clone(), output).expect("Could not debug obj");
        let path = path.as_ref();
        std::fs::write(path.with_extension("mtl"), mtl).expect("Could not debug mtl");

        //builder.build().export_debug_obj(path)
    }

    fn fill_uniform_chunks(&mut self, ranges_yz: Vec<(f64, f64, Vec<(f64, f64, i32)>)>, bvh: &BVH, tris: &[BvhTriangle]) {
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
                            Chunk::Uniform(closest_material(bvh, tris, 
                                vec3(chunk_x, chunk_y, chunk_z).cast::<f64>().unwrap() * Chunk::SIZE as f64
                                    + vec3(Chunk::SIZE / 2, Chunk::SIZE / 2, Chunk::SIZE / 2).cast::<f64>().unwrap()
                            )),
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
        mut chunks: FnvHashMap<Vec3i, ComplexChunk>,
    ) -> FnvHashMap<Vec3i, ComplexChunk> {
        let chunk_size = Chunk::SIZE as i32;

        for (pos, slice) in slices {
            let mut mesh = slice.intersect_unit_cube(pos);
            // Collapse 0-area edges
            //mesh.mesh_mut().collapse_small_faces(f64::MIN_POSITIVE);
            mesh.collapse_small_edges();

            //mesh.export_debug_obj(format!(
            //    "assets/debug/complex/cube_{:04}_{:04}_{:04}.obj",
            //    pos.x, pos.y, pos.z
            //));

            let mut plc = PiecewiseLinearComplex::new(MaterialMesh::new(mesh.mesh().translated(-pos)));
            plc.dissolve();
            let hulls = match plc.tetrahedralize() {
                Ok(tets) => tets.convex_hulls(),
                Err(error) => {
                    eprintln!("Fallback to single convex hull at {:?} because {:?}", pos, error);
                    vec![ mesh.mesh().vertex_iter().map(|v| mesh.mesh().vertex_position(v) - pos).collect() ]
                }
            };

            let pos = pos.cast::<i32>().unwrap();
            let chunk_pos = vec3(pos.x.div_euclid(chunk_size), pos.y.div_euclid(chunk_size), pos.z.div_euclid(chunk_size));
            let in_pos = vec3(pos.x.rem_euclid(chunk_size), pos.y.rem_euclid(chunk_size), pos.z.rem_euclid(chunk_size));

            let chunk = chunks.entry(chunk_pos).or_insert(ComplexChunk::new());
            *chunk.voxel_mut(in_pos) = Voxel::Complex(chunk.complex.len() as u32);
            chunk.complex.push(ComplexVoxel::new(hulls));
        }

        chunks
    }

    fn ranges_to_complex_chunks(
        &self,
        ranges_yz: Vec<(f64, f64, Vec<(f64, f64, i32)>)>,
        bvh: &BVH,
        tris: &[BvhTriangle],
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
                            Voxel::Pure(Some(closest_material(
                                bvh,
                                tris,
                                vec3(x as f64 + 0.5, *y + 0.5, *z + 0.5)
                            )));
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
        let (bvh, triangles) = mesh.bvh();

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

        voxels.fill_uniform_chunks(ranges_yz, &bvh, &triangles);

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

                let chunks = voxels.ranges_to_complex_chunks(ranges_yz, &bvh, &triangles);

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
                    match self.voxel(vec3(x, y, z)) {
                        Voxel::Pure(Some(mat)) =>
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
                            },

                        Voxel::Pure(None) => {} // empty voxel

                        Voxel::Complex(index) => {
                            let boundary = self.complex[index as usize].boundary(chunk_pos * Chunk::SIZE as i32 + vec3(x, y, z));
                            builder.add_mesh(boundary.0, boundary.1);
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
    inner_vertices: Vec<Vec3>,
    hulls: Vec<(Vec<u32>, MaterialID)>,
}

impl ComplexVoxel {
    pub const MAX_HULL_SIZE: usize = 8;

    fn new(hulls: Vec<Vec<Vec3>>) -> Self {
        let vertices = hulls.iter().flat_map(|hull| hull.iter().copied().map(HashVec3))
            .collect::<FnvHashSet<_>>();
        
        // including corners
        let mut all_vertices = (0..2).flat_map(|z|
            (0..2).flat_map(move |y|
                (0..2).map(move |x| HashVec3(vec3(x as f64, y as f64, z as f64)))))
            .collect::<Vec<_>>();
        all_vertices.extend(vertices.difference(&all_vertices.iter().copied().collect::<FnvHashSet<_>>()));

        let index_map = all_vertices.iter().enumerate()
            .map(|(i, v)| (*v, i))
            .collect::<FnvHashMap<_, _>>();
                
        let hulls = hulls
            .into_iter()
            .map(|hull| (hull
                .into_iter()
                .map(|pos| index_map[&HashVec3(pos)] as u32)
                .collect::<FnvHashSet<_>>().into_iter().collect(), MaterialID::new(1)))
            .collect();
                
        Self {
            inner_vertices: all_vertices[8..].iter().map(|h| h.0).collect(),
            hulls
        }
    }

    fn toggle_face(edges: &mut FnvHashMap<[usize; 2], usize>, face: [usize; 3]) {
        if edges.get(&[face[1], face[0]]).unwrap_or(&face[0]) == &face[2] {
            for [i, j] in vec![[1, 0], [2, 1], [0, 2]] {
                edges.remove(&[face[i], face[j]]);
            }
        } else {
            for [i, j, k] in vec![[0, 1, 2], [1, 2, 0], [2, 0, 1]] {
                edges.insert([face[i], face[j]], face[k]);
            }
        }
    }

    fn convex_hull(points: Vec<Vec3>) -> Vec<[Vec3; 3]> {
        // Maps edges to opposite vertex
        let mut edges = FnvHashMap::default();
        let mut angle_check = vec![];

        let mut first_tet = [0, 1, 2, 3];
        for tet in combine::combine_vec(&(0..points.len()).collect(), 4) {
            let det = (points[1] - points[0]).cross(points[2] - points[0]).dot(points[3] - points[0]);
            if det.abs() >= 1e-5 {
                first_tet = if det >= 0.0 {
                    [tet[0], tet[1], tet[3], tet[2]]
                } else {
                    [tet[0], tet[1], tet[2], tet[3]]
                };
                break;
            }
        }
        for [i, j, k] in vec![[0, 1, 2], [3, 2, 1], [2, 3, 0], [1, 0, 3]] {
            Self::toggle_face(&mut edges, [first_tet[i], first_tet[j], first_tet[k]]);
        }

        let mut added = first_tet.iter().copied().collect::<FnvHashSet<_>>();

        for (i, pos) in points.iter().enumerate() {
            if !added.contains(&i) {
                if let Some([v0, v1, v2]) = edges.iter().find(|([v0, v1], v2)| {
                    (points[*v1] - points[*v0]).cross(points[**v2] - points[*v0]).dot(pos - points[*v0]) > 1e-5
                }).map(|([v0, v1], v2)| [*v0, *v1, *v2]) {
                    added.insert(i);

                    // Point is above convex hull; add it.
                    Self::toggle_face(&mut edges, [v0, v2, v1]);
                    Self::toggle_face(&mut edges, [v0, i, v2]);
                    Self::toggle_face(&mut edges, [v2, i, v1]);
                    Self::toggle_face(&mut edges, [v1, i, v0]);

                    // Edges that could be concave
                    angle_check.extend(vec![[v0, v2], [v2, v1], [v1, v0]]);
                }

                while let Some([v0, v1]) = angle_check.pop() {
                    // Check for concavity
                    if let (Some(v2), Some(v3)) = (edges.get(&[v0, v1]).copied(), edges.get(&[v1, v0]).copied()) {
                        // Epsilon to avoid infinite loop
                        if (points[v1] - points[v0]).cross(points[v2] - points[v0]).dot(points[v3] - points[v0]) > 1e-10 {
                            // Flip!
                            Self::toggle_face(&mut edges, [v0, v2, v1]);
                            Self::toggle_face(&mut edges, [v0, v1, v3]);
                            Self::toggle_face(&mut edges, [v1, v2, v3]);
                            Self::toggle_face(&mut edges, [v0, v3, v2]);

                            angle_check.extend(vec![[v0, v3], [v3, v1], [v1, v2], [v2, v0]]);
                        }
                    }
                }
            }
        }

        edges.into_iter()
            .filter(|([v0, v1], v2)| v0 < v1 && v0 < v2)
            .map(|([v0, v1], v2)| [points[v0], points[v1], points[v2]])
            .collect()
    }

    fn boundary(&self, offset: Vec3i) -> (Vec<Vec3>, Vec<([usize; 3], MaterialID)>) {
        let positions = (0..2).flat_map(|z|
            (0..2).flat_map(move |y|
                (0..2).map(move |x| vec3(x, y, z).cast::<f64>().unwrap())))
            .chain(self.inner_vertices.iter().copied())
            .map(|pos| pos + offset.cast::<f64>().unwrap())
            .collect::<Vec<_>>();
        let index_map = positions.iter().enumerate().map(|(i, p)| (HashVec3(*p), i))
            .collect::<FnvHashMap<_, _>>();

        let mut faces = FnvHashMap::default();
                                
        for (index, hull) in self.hulls.iter().enumerate() {
            let convex = Self::convex_hull(
                hull.0.iter().map(|i| positions[*i as usize]).collect::<Vec<_>>()
            );

            if offset == vec3(2, 3, -7) {
                let mesh = MaterialMesh::new(MeshBuilder::new()
                    .with_positions(convex.iter().flat_map(|p|
                        vec![p[0].x, p[0].y, p[0].z, p[1].x, p[1].y, p[1].z, p[2].x, p[2].y, p[2].z]).collect())
                    .build().unwrap());
                mesh.export_debug_obj(format!("assets/debug/voxels_hull_{}.obj", index));
            }

            for face in convex {
                // Canonicalize face
                let face = face.iter().map(|pos| index_map[&HashVec3(*pos)]).collect::<Vec<_>>();
                let i_min = face.iter().enumerate().min_by_key(|(i, v)| **v).unwrap().0;
                let face = [face[i_min], face[(i_min + 1) % 3], face[(i_min + 2) % 3]];

                // Face cancels with face of opposite orientation
                if faces.remove(&[face[0], face[2], face[1]]).is_none() {
                    faces.insert(face, hull.1);
                }
            }
        }

        (positions, faces.into_iter().collect())
    }
}
