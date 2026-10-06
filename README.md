# bevy_slang

`bevy_slang` loads [Slang](https://github.com/shader-slang/slang) source files as ordinary Bevy `Shader` assets. It is based on the integration approach in [bevy_mod_slang](https://github.com/DGriffin91/bevy_mod_slang), with configurable compiler invocation, compiler diagnostics, per-asset metadata, source-root handling, and WGSL output support.

## Requirements

Install a Slang binary distribution so `slangc` is available on `PATH`, or set `SLANGC` to the executable path. The official releases include `slangc`, the compiler library, and headers for Windows, Linux, and macOS.

SPIR-V is the default output and needs Bevy's `shader_format_spirv` feature:

```toml
[dependencies]
bevy = { version = "0.19.1", features = ["3d", "shader_format_spirv"] }
bevy_slang = "0.1"
```

## Usage

Add the plugin alongside Bevy's normal plugins:

```rust
use bevy::{prelude::*, shader::Shader};
use bevy_slang::SlangPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(SlangPlugin::default())
        .add_systems(Startup, load_shader)
        .run();
}

fn load_shader(asset_server: Res<AssetServer>) {
    let _shader: Handle<Shader> = asset_server.load("shaders/example.slang");
}
```

The default source root is `assets`, matching Bevy's default asset directory. If the app uses a different asset directory or Slang include directory, configure it explicitly:

```rust
App::new()
    .add_plugins(DefaultPlugins)
    .add_plugins(
        SlangPlugin::new()
            .with_compiler_path("tools/slangc")
            .with_source_root("game_assets")
            .with_include_path("game_assets/shaders/include"),
    );
```

Use the resulting `Handle<Shader>` anywhere Bevy accepts a `ShaderRef`, including custom materials. The Slang shader should expose the entry points and resource layout expected by that Bevy pipeline.

## Per-shader settings

Embedded/generated shader pipelines can use `SlangCompiler::compile_source`
or `compile_bundle`. The latter accepts an entry filename and `(path, source)`
pairs, preserving relative includes in a temporary source tree. Both return an
ordinary Bevy `Shader`, report compiler diagnostics, and compile synchronously;
invoke them during initialization or on a task pool, never every rendered frame.

For native Vulkan pipelines that use extensions outside Naga's SPIR-V importer,
enable `bevy_slang`'s `spirv_passthrough` feature and enable
`WgpuFeatures::PASSTHROUGH_SHADERS` on the render device. Bevy then passes Slang's
SPIR-V directly to Vulkan. Slang validates emitted SPIR-V by default; the runtime
does not validate passthrough code. This mode is for trusted application shaders.

Settings are read from a standard Bevy sidecar file at `example.slang.meta`:

```ron
(
    meta_format_version: "1.0",
    asset: Load(
        loader: "bevy_slang::SlangLoader",
        settings: (
            target: "spirv",
            profile: Some("glsl_460+spirv_1_5"),
            entry_points: ["vertexMain", "fragmentMain"],
            include_paths: ["assets/shaders/include"],
            defines: ["USE_FOG=1"],
            matrix_layout: "column-major",
            debug_info: true,
        ),
    ),
)
```

Most shaders can omit `entry_points` and `stage` when their functions use Slang's `[shader("vertex")]`, `[shader("fragment")]`, or `[shader("compute")]` attributes. `extra_args` is available for newer Slang options that have not yet gained a typed setting.

For `target: "wgsl"`, specify `entry_points` and `stage` explicitly: Slang's CLI cannot infer entry points from `[shader(...)]` attributes for every non-SPIR-V target. For WebGPU/WebAssembly builds, precompile those assets on a native development machine; runtime compiler invocation is intentionally unavailable on WebAssembly.

## Descriptor bindings

Wgpu 29's Vulkan backend packs bindings densely in layout-entry order. When using
SPIR-V passthrough with a sparse layout, call `remap_spirv_bindings` with
`SpirvBindingRemap` entries before adding the shader to `Assets<Shader>`. Each
mapping identifies its descriptor set, logical Slang binding, and physical
layout-entry index. This adjusts decorations only; it preserves entry points,
shader code, and buffer layouts. Dense layouts and Naga-translated shaders do not
need relocation.

## Development

```text
cargo fmt --check
cargo test
cargo check
```

When using Bevy's file watcher in a development build, editing a directly loaded `.slang` file triggers the normal asset reload path and recompiles it. Imported source modules should be kept in the configured include tree and may require an explicit reload of the entry shader when using a custom asset reader.

An end-to-end loader example is available with:

```text
cargo run --example load_shader
```

## License

Licensed under either of:

* Apache License, Version 2.0
* MIT License
