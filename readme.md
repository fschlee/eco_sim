The recommended way to get rust is to use the rustup tool:
https://rustup.rs/

    cd gui

for vulkan backend (or metal backend on OS X, untested):

    cargo run

If on Linux and using x11 instead of Wayland:

    cargo run --fetures=x11
 
for other backends:

    cargo run --feature="backend" --no-default-features

with "backend" replaced by "dx11", "dx12" or "gl".  

Left click to select entity, ctrl + left click to see inferred mental states for other entities, hover and tab to circle through multiple entities in the same square.
Press t to toggle between sight range and threat mode for selected entity, and m to toggle showing map knowledge. Right click to instruct the seleced enityy
 to move to the clicked square. Space or pause button toggle the pause state, F5 reloads the map.
