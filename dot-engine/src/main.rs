use vulkano::{pipeline::Pipeline, sync::GpuFuture};
use winit::{
	event::{Event, WindowEvent},
	event_loop::ControlFlow,
};

fn main() -> Result<(), impl std::error::Error> {
	let event_loop = winit::event_loop::EventLoop::new().unwrap();

	let window = std::sync::Arc::new(
		winit::window::WindowBuilder::new()
			.with_title("Dot")
			.build(&event_loop)
			.unwrap(),
	);

	let required_extensions = vulkano::swapchain::Surface::required_extensions(&event_loop);

	let shader_spv =
		vulkano::shader::spirv::bytes_to_words(include_bytes!(env!("shader.spv"))).unwrap();

	let mut renderer = Renderer::new(
		window.clone(),
		required_extensions,
		vulkano::Version::major_minor(0, 1),
		&shader_spv,
	);

	event_loop.run(move |event, elwt| {
		elwt.set_control_flow(ControlFlow::Poll);

		match event {
			Event::WindowEvent { event, .. } => {
				match event {
					WindowEvent::CloseRequested => elwt.exit(),
					WindowEvent::Resized(_) => renderer.recreate_swapchain(true),
					WindowEvent::RedrawRequested => {
						let image_extent: [u32; 2] = window.inner_size().into();
						if image_extent.contains(&0) {
							return;
						}
						renderer.run(image_extent, None)
					},
					_ => (),
				}
			},
			Event::AboutToWait => window.request_redraw(),
			_ => (),
		}
	})
}

struct Renderer {
	device: std::sync::Arc<vulkano::device::Device>,
	queue: std::sync::Arc<vulkano::device::Queue>,
	swapchain: std::sync::Arc<vulkano::swapchain::Swapchain>,
	images: Vec<std::sync::Arc<vulkano::image::Image>>,
	compute_pipeline: std::sync::Arc<vulkano::pipeline::ComputePipeline>,
	recreate_swapchain: bool,
	previous_frame_end: Option<Box<dyn GpuFuture>>,
	memory_allocator: std::sync::Arc<
		vulkano::memory::allocator::GenericMemoryAllocator<
			vulkano::memory::allocator::FreeListAllocator,
		>,
	>,
	descriptor_set_allocator: vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator,
	command_buffer_allocator: vulkano::command_buffer::allocator::StandardCommandBufferAllocator,
	buffer_allocator: vulkano::buffer::allocator::SubbufferAllocator,
}

impl Renderer {
	fn new(
		window: std::sync::Arc<winit::window::Window>,
		required_extensions: vulkano::instance::InstanceExtensions,
		app_version: vulkano::Version,
		shader_spv: &[u32],
	) -> Self {
		let instance = vulkano::instance::Instance::new(
			vulkano::VulkanLibrary::new().unwrap(),
			vulkano::instance::InstanceCreateInfo {
				application_version: app_version,
				engine_version: vulkano::Version::major_minor(0, 1),
				engine_name: Some("Dot".to_owned()),
				flags: vulkano::instance::InstanceCreateFlags::ENUMERATE_PORTABILITY,
				enabled_extensions: required_extensions,
				..Default::default()
			},
		)
		.unwrap();

		let surface =
			vulkano::swapchain::Surface::from_window(instance.clone(), window.clone()).unwrap();

		let device_extensions = vulkano::device::DeviceExtensions {
			khr_swapchain: true,
			khr_storage_buffer_storage_class: true,
			khr_vulkan_memory_model: true,
			..vulkano::device::DeviceExtensions::empty()
		};
		let (physical_device, queue_family_index) = instance
			.enumerate_physical_devices()
			.unwrap()
			.filter(|p| p.supported_extensions().contains(&device_extensions))
			.filter_map(|p| {
				p.queue_family_properties()
					.iter()
					.enumerate()
					.position(|(i, q)| {
						q.queue_flags
							.intersects(vulkano::device::QueueFlags::COMPUTE)
							&& p.surface_support(i as u32, &surface).unwrap_or(false)
					})
					.map(|i| (p, i as u32))
			})
			.min_by_key(|(p, _)| {
				match p.properties().device_type {
					vulkano::device::physical::PhysicalDeviceType::DiscreteGpu => 0,
					vulkano::device::physical::PhysicalDeviceType::IntegratedGpu => 1,
					vulkano::device::physical::PhysicalDeviceType::VirtualGpu => 2,
					vulkano::device::physical::PhysicalDeviceType::Cpu => 3,
					vulkano::device::physical::PhysicalDeviceType::Other => 4,
					_ => 5,
				}
			})
			.expect("no suitable physical device found");

		println!(
			"Using device: {} (type: {:?})",
			physical_device.properties().device_name,
			physical_device.properties().device_type,
		);

		let (device, mut queues) = vulkano::device::Device::new(
			physical_device,
			vulkano::device::DeviceCreateInfo {
				enabled_extensions: device_extensions,
				queue_create_infos: vec![vulkano::device::QueueCreateInfo {
					queue_family_index,
					..Default::default()
				}],
				enabled_features: vulkano::device::Features {
					vulkan_memory_model: true,
					..vulkano::device::Features::empty()
				},
				..Default::default()
			},
		)
		.unwrap();

		let queue = queues.next().unwrap();

		let (swapchain, images) = {
			let surface_capabilities = device
				.physical_device()
				.surface_capabilities(&surface, Default::default())
				.unwrap();

			vulkano::swapchain::Swapchain::new(
				device.clone(),
				surface,
				vulkano::swapchain::SwapchainCreateInfo {
					min_image_count: surface_capabilities.min_image_count.max(2),

					image_format: vulkano::format::Format::B8G8R8A8_UNORM,

					image_extent: window.inner_size().into(),

					image_usage: vulkano::image::ImageUsage::STORAGE,

					composite_alpha: vulkano::swapchain::CompositeAlpha::Opaque,

					..Default::default()
				},
			)
			.unwrap()
		};

		let compute_pipeline = {
			let shader = {
				unsafe {
					vulkano::shader::ShaderModule::new(
						device.clone(),
						vulkano::shader::ShaderModuleCreateInfo::new(&shader_spv),
					)
				}
				.unwrap()
				.entry_point("main")
				.unwrap()
			};

			let stage = vulkano::pipeline::PipelineShaderStageCreateInfo::new(shader);

			let layout = vulkano::pipeline::PipelineLayout::new(
				device.clone(),
				vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo::from_stages([
					&stage,
				])
				.into_pipeline_layout_create_info(device.clone())
				.unwrap(),
			)
			.unwrap();

			vulkano::pipeline::ComputePipeline::new(
				device.clone(),
				None,
				vulkano::pipeline::compute::ComputePipelineCreateInfo::stage_layout(stage, layout),
			)
			.unwrap()
		};

		let recreate_swapchain = false;

		let previous_frame_end = Some(vulkano::sync::now(device.clone()).boxed());

		let memory_allocator = std::sync::Arc::new(
			vulkano::memory::allocator::StandardMemoryAllocator::new_default(device.clone()),
		);
		let descriptor_set_allocator =
			vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator::new(
				device.clone(),
				Default::default(),
			);
		let command_buffer_allocator =
			vulkano::command_buffer::allocator::StandardCommandBufferAllocator::new(
				device.clone(),
				Default::default(),
			);
		let buffer_allocator = vulkano::buffer::allocator::SubbufferAllocator::new(
			memory_allocator.clone(),
			vulkano::buffer::allocator::SubbufferAllocatorCreateInfo {
				buffer_usage: vulkano::buffer::BufferUsage::STORAGE_BUFFER,
				memory_type_filter: vulkano::memory::allocator::MemoryTypeFilter::PREFER_DEVICE
					| vulkano::memory::allocator::MemoryTypeFilter::HOST_RANDOM_ACCESS,
				..Default::default()
			},
		);
		Self {
			device,
			queue,
			swapchain,
			images,
			compute_pipeline,
			recreate_swapchain,
			previous_frame_end,
			memory_allocator,
			descriptor_set_allocator,
			command_buffer_allocator,
			buffer_allocator,
		}
	}

	fn run(
		&mut self,
		image_extent: [u32; 2],
		additional_set: Option<std::sync::Arc<vulkano::descriptor_set::PersistentDescriptorSet>>,
	) {
		self.previous_frame_end.as_mut().unwrap().cleanup_finished();

		if self.recreate_swapchain {
			let (new_swapchain, new_images) = self
				.swapchain
				.recreate(vulkano::swapchain::SwapchainCreateInfo {
					image_extent,
					..self.swapchain.create_info()
				})
				.expect("failed to recreate swapchain");
			self.images = new_images;
			self.swapchain = new_swapchain;

			self.recreate_swapchain = false;
		}

		let (image_index, suboptimal, acquire_future) =
			match vulkano::swapchain::acquire_next_image(self.swapchain.clone(), None)
				.map_err(vulkano::Validated::unwrap)
			{
				Ok(r) => r,
				Err(vulkano::VulkanError::OutOfDate) => {
					self.recreate_swapchain = true;
					return;
				},
				Err(e) => panic!("failed to acquire next image: {e}"),
			};

		if suboptimal {
			self.recreate_swapchain = true;
		}

		let view =
			vulkano::image::view::ImageView::new_default(self.images[image_index as usize].clone())
				.unwrap();

		let layout = self.compute_pipeline.layout().set_layouts().get(0).unwrap();
		let set = vulkano::descriptor_set::PersistentDescriptorSet::new(
			&self.descriptor_set_allocator,
			layout.clone(),
			[vulkano::descriptor_set::WriteDescriptorSet::image_view(
				0, view,
			)],
			[],
		)
		.unwrap();

		let sets = if let Some(additional_set) = additional_set {
			vec![set, additional_set]
		} else {
			vec![set]
		};

		let mut builder = vulkano::command_buffer::AutoCommandBufferBuilder::primary(
			&self.command_buffer_allocator,
			self.queue.queue_family_index(),
			vulkano::command_buffer::CommandBufferUsage::OneTimeSubmit,
		)
		.unwrap();
		builder
			.bind_pipeline_compute(self.compute_pipeline.clone())
			.unwrap()
			.bind_descriptor_sets(
				vulkano::pipeline::PipelineBindPoint::Compute,
				self.compute_pipeline.layout().clone(),
				0,
				sets,
			)
			.unwrap()
			.dispatch([image_extent[0], image_extent[1], 1])
			.unwrap();
		let command_buffer = builder.build().unwrap();

		let future = self
			.previous_frame_end
			.take()
			.unwrap()
			.join(acquire_future)
			.then_execute(self.queue.clone(), command_buffer)
			.unwrap()
			.then_swapchain_present(
				self.queue.clone(),
				vulkano::swapchain::SwapchainPresentInfo::swapchain_image_index(
					self.swapchain.clone(),
					image_index,
				),
			)
			.then_signal_fence_and_flush();

		match future.map_err(vulkano::Validated::unwrap) {
			Ok(future) => {
				self.previous_frame_end = Some(future.boxed());
			},
			Err(vulkano::VulkanError::OutOfDate) => {
				self.recreate_swapchain = true;
				self.previous_frame_end = Some(vulkano::sync::now(self.device.clone()).boxed());
			},
			Err(e) => {
				println!("failed to flush future: {e}");
				self.previous_frame_end = Some(vulkano::sync::now(self.device.clone()).boxed());
			},
		}
	}

	fn recreate_swapchain(&mut self, value: bool) {
		self.recreate_swapchain = value;
	}
}