use crate::ui::Action;

use eco_sim::{SimState, Storage, WorldEntity, MentalState, Coord, entity_type::{EntityType, Count}};
use crate::renderer::con_back::{UiVertex};
use std::ops::Range;
use winit::dpi::LogicalPosition;
use std::collections::{HashSet, HashMap};
use itertools::Itertools;
use crate::renderer::memory::{Id, Tex};

pub struct BaseData {
    pub vertices: Vec<UiVertex>,
    pub indices: Vec<u32>,
}

pub enum Update<'a> {
    Replace(& 'a BaseData),
    // Writes,
    // Transformations,
}
#[derive(Clone, Copy, Debug)]
pub struct Instance {
    pub x_offset: f32,
    pub y_offset: f32,
    pub highlight: u32,
    pub z: f32,
}
pub struct Model {
    pub range: Range<u32>,
    pub texture: Option<Id<Tex>>,
    pub vec: Vec<Instance>,

}
impl Default for Model {
    fn default() -> Self {
        Model { range: 0..0, texture: None, vec: Vec::new() }
    }
}
pub struct RenderData <'a> {
    pub update: Option<Update<'a>>,
    pub models: & 'a [Model],
}

pub struct GameState{
    eco_sim: eco_sim::SimState,
    base_data: BaseData,
    re_register: bool,
    models: [Model; EntityType::COUNT + Self::NON_ENTITY_MODELS],
    paused: bool,
    highlighted: HashSet<(usize, usize)>,
    highlight_visible: Option<WorldEntity>,
    threat_mode: bool,
    cell_width: f32,
    cell_height: f32,
    margin: f32,
}

const SIM_STEP : f32 = 0.1;

impl GameState {
    pub const NON_ENTITY_MODELS : usize = 1;
    const SQUARE : usize = EntityType::COUNT;
    pub fn new() -> GameState{
        let eco_sim = SimState::new(SIM_STEP);
        let (base_data, models) = Self::init_models();
        GameState{
            eco_sim,
            base_data,
            re_register: true,
            models,
            paused: true,
            highlighted: HashSet::new(),
            highlight_visible: None,
            cell_width : 80.0,
            cell_height: 80.0,
            margin: 80.0,
            threat_mode: false,
        }
    }
    fn init_models() -> (BaseData, [Model; EntityType::COUNT + Self::NON_ENTITY_MODELS]) {
        let mut models : [Model; EntityType::COUNT + Self::NON_ENTITY_MODELS] = Default::default();
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        {
            let v = |p0, p1, c| UiVertex {
                pos: [p0, p1],
                uv: [p0, p1],
                mode: 0,
                color: c
            };

            let full_block = |col| vec![v(0.0, 0.0, col), v(1.0, 0.0, col), v(1.0, 1.0, col), v(0.0, 1.0, col)];

            {

                vertices.append( & mut full_block(0xff808080));
                indices.append(&mut vec![0, 1, 3, 3, 1, 2]);
                models[Self::SQUARE].range = 0..6;
            }
            {
                let dg = 0xff008000;
                models[EntityType::Tree.idx()].range = ball(dg, 1.0, & mut vertices, & mut indices);
                models[EntityType::Clover.idx()].range  = block(dg, & mut vertices, & mut indices);
            }
            let lg = 0xff00ff00;
            models[EntityType::Grass.idx()].range  = block(lg, & mut vertices, & mut indices);

            {
                let dg = 0xff404040;
                models[EntityType::Rabbit.idx()].range = ball(dg, 0.7, &mut vertices, &mut indices);
                models[EntityType::Wolf.idx()].range = ball(dg, 0.9, &mut vertices, &mut indices);
                let brown = 0xff002060;
                models[EntityType::Deer.idx()].range = ball(brown, 0.85, &mut vertices, &mut indices);
            }
            {
                let grey = 0xff202020;
                models[EntityType::Rock.idx()].range = block(grey, & mut vertices, & mut indices);
                models[EntityType::Burrow.idx()].range = ball(grey, 0.75, & mut vertices, & mut indices);
            }

        }

        (BaseData { vertices, indices}, models)
    }
    pub fn update(&mut self, actions: impl Iterator<Item=Action>, time_step: f32) {
        for a in actions {
            match a {
                Action::Pause => self.paused = true,
                Action::Unpause => self.paused = false,
                Action::Reset(pause_state) => {
                    self.paused = pause_state;
                    self.eco_sim = SimState::new(SIM_STEP);
                },
                Action::UpdateMentalState(mental_state) => {
                    self.eco_sim.update_mental_state(mental_state);
                }
                Action::Hover(pos) => {
                    let coords= self.logical_position_to_coords(pos);
                    if self.highlight_visible.is_none() {
                        self.highlighted.clear();
                    }
                    self.highlighted.insert(coords);
                }
                Action::Move(entity, pos) => {
                    if let Some(ms) = self.eco_sim.get_mental_state(&entity) {
                        let (x, y) = self.logical_position_to_coords(pos);
                        let sim_pos = eco_sim::Position { x: x as Coord, y: y as Coord };

                        let mut new_ms = ms.clone();
                        new_ms.current_action = eco_sim::Action::Idle;
                        new_ms.current_behavior = Some(eco_sim::Behavior::Travel(sim_pos));
                        self.eco_sim.update_mental_state(new_ms);
                    }
                }
                Action::ClearHighlight => self.highlight_visible = None,
                Action::HighlightVisibility(ent) => self.highlight_visible = Some(ent),
                Action::ToggleThreatMode => self.threat_mode = !self.threat_mode,
            }
        }
        if !self.paused {
            self.eco_sim.advance(time_step);
        }

    }
    pub fn get_render_data(&mut self) -> RenderData {
        for m in self.models.iter_mut() {
            m.vec.clear()
        }
        let danger =  if let Some(ent) = self.highlight_visible {
            self.highlighted.clear();
            for eco_sim::Position{x, y} in  self.eco_sim.get_visibility(&ent) {
                self.highlighted.insert((x as usize, y as usize));
            }
            if self.threat_mode && self.highlight_visible.is_some() {
                self.eco_sim.threat_map(&ent)
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };
        for (i, j, dat) in self.eco_sim.get_view(0..eco_sim::MAP_WIDTH, 0..eco_sim::MAP_HEIGHT){
            let x_offset = self.cell_width * i as f32;
            let y_offset = self.cell_height * j as f32;
            let idx = eco_sim::MAP_WIDTH * j + i;
            let col = match self.highlighted.contains(&(i, j)){
                x if danger.len() > idx && self.threat_mode => x as u32 * 256 | 512 | ((danger[idx] * 10.0).floor() as u32).min(255),
                true => 256,
                false => 0,
            };
            let mut z = 0.01;
            self.models[Self::SQUARE].vec.push(Instance { x_offset, y_offset, highlight:  col, z });
            for ent in dat {
                z += 0.01;
                self.models[ent.e_type().idx()].vec.push(Instance{ x_offset, y_offset, highlight: 0, z});
            }
        }
        let update = if self.re_register {
            self.re_register = false;
            Some(Update::Replace(& self.base_data))
        } else {
            None
        };

         RenderData{
             update,
             models: &self.models,
         }
    }
    pub fn get_view(
        &self,
    ) -> impl Iterator<Item = (usize, usize, &[eco_sim::ViewData])> + '_ {
        self.eco_sim.get_view(0..eco_sim::MAP_WIDTH, 0..eco_sim::MAP_HEIGHT)
    }
    pub fn get_editable_entity(&self, position: LogicalPosition) -> Option<WorldEntity> {
        let sim_pos = self.logical_to_sim_position(position);
        self.eco_sim.entities_at(sim_pos).iter()
            .find(|e| { self.eco_sim.get_mental_state(*e).is_some()}).copied()
            .or(self.eco_sim.entities_at(sim_pos).iter().next().copied())

    }
    pub fn get_editable_index(&self, sim_pos: eco_sim::Position) -> usize {
        self.eco_sim.entities_at(sim_pos).iter().enumerate().find_map(|(i, e)| {
            match self.eco_sim.get_mental_state(e) {
                Some(_) => Some(i),
                None => None,
            }
        }).unwrap_or(0)
    }
    pub fn get_nth_entity(&self, n: usize, sim_pos: eco_sim::Position) -> Option<WorldEntity> {
        self.eco_sim.entities_at(sim_pos).iter().cycle().dropping(n).next().copied()
    }
    pub fn get_mental_state(&self, entity: &WorldEntity) -> Option<&MentalState> {
        self.eco_sim.get_mental_state(entity)
    }
    pub fn get_mental_model(&self, entity: &WorldEntity) -> Option<impl Iterator<Item = & impl std::fmt::Display>> {
        self.eco_sim.get_mental_model(entity)
    }
    pub fn get_physical_state(&self, entity: &WorldEntity) -> Option<&eco_sim::PhysicalState> {
        self.eco_sim.get_physical_state(entity)
    }
    pub fn logical_to_sim_position(&self, position: LogicalPosition) -> eco_sim::Position {
        let (x, y) = self.logical_position_to_coords(position);
        eco_sim::Position{ x: x as Coord, y: y as Coord}
    }
    fn logical_position_to_coords(&self, position: LogicalPosition) -> (usize, usize) {
        let x = ((position.x as f32 - self.margin) / self.cell_width).floor() as usize;
        let y = ((position.y as f32 - self.margin) / self.cell_height).floor() as usize;
        (x, y)
    }
    pub fn is_within(& self, position: LogicalPosition) -> bool {
        position.x as f32 >= self.margin
            && position.y as f32 >= self.margin
            && position.x as f32 <= self.margin + self.cell_width * eco_sim::MAP_WIDTH as f32
            && position.y as f32 <= self.margin + self.cell_height * eco_sim::MAP_HEIGHT as f32
    }
}

pub fn block(col: u32, vertices: &mut Vec<UiVertex>, indices: & mut Vec<u32>) -> Range<u32> {
    let v = |p0, p1, c| UiVertex {
        pos: [p0, p1],
        uv: [p0, p1],
        mode: 0,
        color: c
    };
    let base = vertices.len() as u32;
    let idx = indices.len();
    vertices.append(&mut vec![v(0.1, 0.1, col), v(0.9, 0.1, col), v(0.9, 0.9, col), v(0.1, 0.9, col)]);
    indices.append(&mut vec![base + 0, base + 1, base + 3, base + 3, base + 1, base + 2]);
    idx as u32.. (idx as u32 + 6)
}

pub fn ball(col: u32, radius: f32, vertices: &mut Vec<UiVertex>, indices: & mut Vec<u32>) -> Range<u32> {
    let v = |p0, p1, c| UiVertex {
        pos: [p0, p1],
        uv: [p0, p1],
        mode: 0,
        color: c
    };
    let base = vertices.len() as u32;
    let idx = indices.len();
    let base = vertices.len() as u32;
    let l = radius;
    let m = 0.5 + (radius - 0.5)  / 1.5;


    vertices.append(&mut vec![v(0.5, 0.5, col), v(l, 0.5, col),
                              v(m, m, col), v(0.5, l, col),
                              v(1.0 - m, m, col), v(1.0 - l, 0.5, col),
                              v(1.0 - m, 1.0 - m, col), v(0.5, 1.0 - l, col),
                              v(m, 1.0 - m, col)]);

    for i in 1..8 {
        indices.push(base);
        indices.push(base + i);
        indices.push(base + i + 1);
    }
    indices.push(base);
    indices.push(base + 8);
    indices.push(base + 1);
    idx as u32..(idx as u32 + 24)
}