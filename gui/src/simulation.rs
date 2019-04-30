use crate::ui::Action;

use eco_sim::{SimState, EntityType, Storage};
use crate::renderer::con_back::{UiVertex};
use std::ops::Range;

pub struct RenderData{
    pub vertices: Vec<UiVertex>,
    pub indices: Vec<u32>,
    pub commands: Vec<Command>
}

pub struct Command {
    pub range: Range<u32>,
    pub x_offset: f32,
    pub y_offset: f32,


}

pub enum RenderUpdate{
    New,
    Delete,
    Transform,
    Once(RenderData)
}

pub struct GameState{
    eco_sim: eco_sim::SimState,
    cache: Storage<Command>,
    paused: bool,

}

const SIM_STEP : f32 = 0.1;

impl GameState {
    pub fn new() -> GameState{
        let eco_sim = SimState::new(SIM_STEP);
        println!("init");
        GameState{eco_sim, cache: Storage::new(), paused: true }
    }
    pub fn update(&mut self, actions: Vec<Action>, time_step: f32) {
        for a in &actions {
            match a {
                Action::Pause => self.paused = true,
                Action::Unpause => self.paused = false,
                Action::Reset(pause_state) => {
                    self.paused = *pause_state;
                    self.eco_sim = SimState::new(SIM_STEP);
                },
            }
        }
        if !self.paused {
            self.eco_sim.advance(time_step);
        }

    }
    pub fn get_render_data(&mut self) -> RenderData {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        {
            let v = |p0, p1, c| UiVertex {
                pos: [p0, p1],
                uv: [p0, p1],
                mode: 0,
                color: c
            };
            {
                let grey = 0xff808080;
                vertices.append(&mut vec![v(0.0, 0.0, grey), v(1.0, 0.0, grey), v(1.0, 1.0, grey), v(0.0, 1.0, grey)]);
                indices.append(&mut vec![0, 1, 3, 3, 1, 2]);
            }
            {
                let dg = 0xff008000;
                let base = vertices.len() as u32;
                vertices.append(&mut vec![v(0.5, 0.5, dg), v(0.8, 0.5, dg), v(0.7, 0.7, dg), v(0.5, 0.8, dg), v(0.3, 0.7, dg), v(0.2, 0.5, dg), v(0.3, 0.3, dg), v(0.5, 0.2, dg), v(0.7, 0.3, dg)]);

                for i in 1..8 {
                    indices.push(base);
                    indices.push(base + i);
                    indices.push(base + i + 1);
                }
                indices.push(base);
                indices.push(base + 8);
                indices.push(base + 1);
            }
            let lg = 0xff00ff00;
            vertices.append(& mut vec![v(0.0, 0.0, lg), v(1.0, 0.0, lg), v(1.0, 1.0, lg), v(0.0, 1.0, lg)]);
            indices.append(&  mut vec![13, 14, 16, 16, 14, 15]);
            {
                let dg = 0xff404040;
                let base = vertices.len() as u32;
                vertices.append(&mut vec![v(0.5, 0.5, dg), v(0.8, 0.5, dg), v(0.7, 0.7, dg), v(0.5, 0.8, dg), v(0.3, 0.7, dg), v(0.2, 0.5, dg), v(0.3, 0.3, dg), v(0.5, 0.2, dg), v(0.7, 0.3, dg)]);

                for i in 1..8 {
                    indices.push(base);
                    indices.push(base + i);
                    indices.push(base + i + 1);
                }
                indices.push(base);
                indices.push(base + 8);
                indices.push(base + 1);
            }
            {
                let grey = 0xff202020;
                let base = vertices.len() as u32;
                vertices.append(&mut vec![v(0.0, 0.0, grey), v(1.0, 0.0, grey), v(1.0, 1.0, grey), v(0.0, 1.0, grey)]);
                indices.append(&mut vec![base + 0, base + 1, base + 3, base + 3, base + 1, base + 2]);
            }

        }
        let mut commands = Vec::new();
        let cell_width = 48.0;
        let cell_height = 48.0;
        fn lookup(et: EntityType) -> Range<u32> {
            use EntityType::*;
            // print!("{:?}", et);
            match et {
                Tree=> 6..30,
                Grass => 30..36,
                Rabbit=> 36..60,
                Rock => 60..66,
            }
        }
        for (i, j, dat) in self.eco_sim.get_view(0..eco_sim::MAP_WIDTH, 0..eco_sim::MAP_HEIGHT){
            let x_offset = cell_width * i as f32;
            let y_offset = cell_height * j as f32;
            commands.push(Command{range: 0..6, x_offset, y_offset });
            if let Some(ents) = dat {
                for ent in ents {
                    commands.push(Command{range: lookup(ent), x_offset, y_offset });
                }
            }


        }
         RenderData{
             vertices,
             indices,
             commands,
         }
    }
    pub fn get_view(
        &self,
    ) -> impl Iterator<Item = (usize, usize, eco_sim::ViewData)> + '_ {
        self.eco_sim.get_view(0..eco_sim::MAP_WIDTH, 0..eco_sim::MAP_HEIGHT)
    }
}

