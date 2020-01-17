## Installation

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

### Install as Python library
Install maturin (https://github.com/PyO3/maturin), then go to `eco_sim_py\ffi` and from within a virtualenv run

    maturin develop 

Alternatively to install globally

    maturin build  

## Usage (GUI)
Left click to select entity, ctrl + left click to see inferred mental states for other entities, hover and tab to circle through multiple entities in the same square.
Press t to toggle between sight range and threat mode for selected entity, and m to toggle showing map knowledge. Right click to instruct the seleced enityy
 to move to the clicked square. Space or pause button toggle the pause state, +/- adjust speed, F5 reloads the map.

## Project Structure

`sim` contains the simulation itself, including agent behavior and mental state inference.

`gui` contains the GUI.

`enum_macros` contains helper macros used in both `sim` and `gui`.

`eco_sim_py/ffi` contains the Python bindings exposing `sim` as reinforcement learning environment, with optional use of `gui` for visualization.

`eco_sim_py/drl` uses the environment exposed in `eco_sim_py/ffi` and pytorch to define and train a simple NN. The NN has two heads, 
one predicting the rewards associated with actions, the second predicting the current mental states of other agents. The value head is used for Q-learning.