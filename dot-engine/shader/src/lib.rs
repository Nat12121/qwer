#![cfg_attr(target_arch = "spirv", no_std)]

use spirv_std::{
	glam::{vec3, UVec3, Vec3Swizzles},
	spirv,
};

#[spirv(compute(threads(1, 1)))]
pub fn main(
	#[spirv(num_workgroups)] work_groups: UVec3,
	#[spirv(global_invocation_id)] id: UVec3,
	#[spirv(descriptor_set = 0, binding = 0)] image: &spirv_std::Image!(2D, type=f32, sampled=false, depth=false),
) {
	unsafe {
		image.write(
			id.xy(),
			vec3(
				id.x as f32 / work_groups.x as f32,
				id.y as f32 / work_groups.y as f32,
				0.0,
			),
		);
	}
}