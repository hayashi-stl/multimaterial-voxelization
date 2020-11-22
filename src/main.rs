use std::fs;
use voxelization::material_mesh::MaterialMesh;
use voxelization::voxels::Voxels;

fn main() {
    let source = fs::read_to_string("assets/test.obj").expect("File not read");
    let mesh = MaterialMesh::from_obj_1_material(source).expect("Invalid mesh");

    let voxels = Voxels::from(mesh);
    voxels.export_debug_obj("assets/debug/voxels.obj");
}
