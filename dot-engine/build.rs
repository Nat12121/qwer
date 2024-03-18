use std::error::Error;

use spirv_builder::{MetadataPrintout, SpirvBuilder};

fn main() -> Result<(), Box<dyn Error>> {
	SpirvBuilder::new("shader", "spirv-unknown-vulkan1.0")
		.capability(spirv_builder::Capability::StorageImageWriteWithoutFormat)
		.extension("SPV_KHR_storage_buffer_storage_class")
		.release(true)
		.print_metadata(MetadataPrintout::Full)
		.build()?;
	Ok(())
}