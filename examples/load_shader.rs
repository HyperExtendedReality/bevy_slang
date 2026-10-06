//! Loads a Slang shader through Bevy's asset system.

use bevy::{
    asset::{AssetPlugin, AssetServer},
    prelude::{App, Handle, MinimalPlugins, Res, Startup},
    shader::Shader,
};
use bevy_slang::SlangPlugin;

fn main() {
    App::new()
        .add_plugins((
            MinimalPlugins,
            AssetPlugin::default(),
            SlangPlugin::default(),
        ))
        .add_systems(Startup, load_shader)
        .run();
}

fn load_shader(asset_server: Res<AssetServer>) {
    let _: Handle<Shader> = asset_server.load("shaders/example.slang");
}
